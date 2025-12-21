import copy
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import ray
import tqdm

from stochastic_muzero.mcts import StochasticMuZeroMCTS
from stochastic_muzero.model import categorical_to_scalar, scalar_to_categorical

class StochasticMuZeroSPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.rewards = []  # Immediate rewards from env
        self.history = []  # Action history
        self.obs_history = []  # Observation history


class GameHistory:
    def __init__(self, observations, actions, rewards, root_values, policy_probs):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.root_values = root_values
        self.policy_probs = policy_probs
        
    def __len__(self):
        return len(self.actions)


@ray.remote
class StochasticMuZeroSelfPlayWorker:
    def __init__(self, game_cls, model_cls, model_args, mcts_config, games_per_worker):
        self.game = game_cls()
        self.model = model_cls(**model_args)
        self.mcts_config = mcts_config
        self.games_per_worker = games_per_worker
        
    def play_games(self, weights, temperature):
        self.model.load_state_dict(weights)
        self.model.eval()
        
        mcts = StochasticMuZeroMCTS(
            game=self.game,
            model=self.model,
            num_searches=self.mcts_config.get('num_searches', 50),
            c_puct=self.mcts_config.get('c_puct', 1.41),
            dirichlet_epsilon=self.mcts_config.get('dirichlet_epsilon', 0.25),
            dirichlet_alpha=self.mcts_config.get('dirichlet_alpha', 0.3),
            discount=self.mcts_config.get('discount', 0.997),
            use_chance_nodes=self.mcts_config.get('use_chance_nodes', False)
        )
        
        histories = []
        
        for _ in range(self.games_per_worker):
            spg = StochasticMuZeroSPG(self.game)
            obs = self.game.get_encoded_state(spg.state)
            spg.obs_history.append(obs)
            
            active_games = [spg] 
            
            while True:
                current_states = [spg.state for spg in active_games]
                mcts.search(current_states, active_games)
                
                root = spg.root
                action_probs = mcts.get_action_probs(root, temperature)
                action = np.random.choice(len(action_probs), p=action_probs)
                
                spg.history.append(action)
                spg.policy_probs.append(action_probs)
                spg.root_values.append(root.value())
                
                spg.state.apply_action(action)
                
                reward = 0.0
                if hasattr(spg.state, 'reward'):
                    reward = spg.state.reward
                
                _, is_terminal = self.game.get_value_and_terminated(spg.state, 0)
                
                if is_terminal and reward == 0.0:
                    val, _ = self.game.get_value_and_terminated(spg.state, 0)
                    reward = val
                
                spg.rewards.append(reward)
                spg.obs_history.append(self.game.get_encoded_state(spg.state))
                
                if is_terminal:
                    histories.append(GameHistory(
                        spg.obs_history[:-1],
                        spg.history,
                        spg.rewards,
                        spg.root_values,
                        spg.policy_probs,
                    ))
                    break
        
        return histories


class StochasticMuZero:
    def __init__(self, model, optimizer, game_cls, model_cls, model_args,
                 mcts_config, num_parallel_games, temperature, batch_size,
                 num_iterations, num_selfPlay_iterations, num_epochs,
                 unroll_steps=5, td_steps=10, discount=0.997, games_per_worker=10,
                 training_ratio=10, max_buffer_size=10000):
        
        self.policy_name = "StochasticMuZero"
        
        self.model = model
        self.optimizer = optimizer
        self.game_cls = game_cls
        self.model_cls = model_cls
        self.model_args = model_args
        self.mcts_config = mcts_config
        self.num_parallel_games = num_parallel_games
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_selfPlay_iterations = num_selfPlay_iterations
        self.num_epochs = num_epochs
        
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.training_ratio = training_ratio
        self.max_buffer_size = max_buffer_size
        self.support_range = model_args.get("support_range", (-300, 301, 1))
        
        self.replay_buffer = [] # Simple list buffer
        
        # Initialize Ray workers
        self.workers = [
            StochasticMuZeroSelfPlayWorker.remote(
                game_cls, model_cls, model_args, mcts_config, games_per_worker
            ) for _ in range(num_parallel_games)
        ]
        
    def selfPlay(self):
        weights = self.model.state_dict()
        weights_id = ray.put(weights)
        
        # Launch workers
        futures = [w.play_games.remote(weights_id, self.temperature) for w in self.workers]
        
        # Collect results
        new_histories = []
        for history_batch in ray.get(futures):
            new_histories.extend(history_batch)
            
        print(f"Collected {len(new_histories)} new games.")
        self.replay_buffer.extend(new_histories)

        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
            
        return len(new_histories)

    def learn(self):
        for iteration in range(self.num_iterations):
            print(f"--- Iteration {iteration + 1}/{self.num_iterations} ---")
            
            # Self Play
            self.selfPlay()
            
            # Training
            loss = self.train()
            print(f"Loss: {loss:.4f}")
            
            # Checkpoint
            torch.save(self.model.state_dict(), f"model_{self.policy_name}_{iteration}.pt")
            
    def train(self):
        self.model.train()
        total_loss = 0
        
        # Determine number of updates based on ratio
        num_updates = int(len(self.replay_buffer) * self.training_ratio / self.batch_size)
        
        pbar = tqdm.tqdm(range(num_updates), desc="Training")
        for _ in pbar:
            batch = self.sample_batch()
            loss = self.update_weights(batch)
            total_loss += loss
            pbar.set_postfix(loss=loss)
            
        return total_loss / num_updates

    def sample_batch(self):
        game_histories = [random.choice(self.replay_buffer) for _ in range(self.batch_size)]
        
        obs_batch = []
        action_batch = []
        target_value_batch = []
        target_reward_batch = []
        target_policy_batch = []
        obs_batch_targets = []
        
        for game in game_histories:
            game_len = len(game.actions)
            start_index = random.randint(0, game_len - 1)
            
            obs_batch.append(game.observations[start_index])
            
            actions = []
            values = []
            rewards = []
            policies = []
            
            for k in range(self.unroll_steps):
                current_index = start_index + k
                
                if current_index < game_len:
                    actions.append(game.actions[current_index])
                else:
                    actions.append(random.randint(0, self.model.action_size - 1))
                    
                if current_index < game_len:
                    bootstrap_index = current_index + self.td_steps
                    if bootstrap_index < game_len:
                        value = game.root_values[bootstrap_index] * (self.discount ** self.td_steps)
                        for i in range(self.td_steps):
                            value += game.rewards[current_index + i] * (self.discount ** i)
                    else:
                        value = 0 # Terminal
                        for i in range(game_len - current_index):
                            value += game.rewards[current_index + i] * (self.discount ** i)
                             
                    values.append(value)
                    rewards.append(game.rewards[current_index])
                    policies.append(game.policy_probs[current_index])
                else:
                    values.append(0.0)
                    rewards.append(0.0)
                    policies.append(np.zeros(self.model.action_size)) # Masked
                    
            action_batch.append(actions)
            target_value_batch.append(values)
            target_reward_batch.append(rewards)
            target_policy_batch.append(policies)
            
            game_obs = []
            for k in range(self.unroll_steps):
                obs_idx = start_index + k + 1
                if obs_idx < len(game.observations):
                    game_obs.append(game.observations[obs_idx])
                else:
                    # Pad with last observation 
                    game_obs.append(game.observations[-1])
            obs_batch_targets.append(game_obs)
            
        return (torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.model.device),
                torch.tensor(np.array(action_batch), dtype=torch.long, device=self.model.device),
                torch.tensor(np.array(target_value_batch), dtype=torch.float32, device=self.model.device),
                torch.tensor(np.array(target_reward_batch), dtype=torch.float32, device=self.model.device),
                torch.tensor(np.array(target_policy_batch), dtype=torch.float32, device=self.model.device),
                torch.tensor(np.array(obs_batch_targets), dtype=torch.float32, device=self.model.device))
    
    def update_weights(self, batch):
        obs, actions, target_values, target_rewards, target_policies, target_obs = batch
        
        # Initial Inference
        hidden_state, policy_logits, value_logits, reward_logits, _ = self.model.initial_inference(obs)
        
        # Targets for Step 0
        target_value_0 = target_values[:, 0]
        target_policy_0 = target_policies[:, 0]
        
        # Loss 0
        value_loss = self.scalar_loss(value_logits, target_value_0)
        policy_loss = F.cross_entropy(policy_logits, target_policy_0)
        
        # Accumulate Loss
        loss = value_loss + policy_loss
        
        # Metrics for logging
        total_value_loss = value_loss.item()
        total_policy_loss = policy_loss.item()
        total_reward_loss = 0.0
        total_chance_loss = 0.0
        total_afterstate_loss = 0.0
        
        current_hidden = hidden_state
        gradient_scale = 1.0 / self.unroll_steps
        
        for k in range(self.unroll_steps):
            action = actions[:, k]
            
            afterstate, reward_predictions = self.model.dynamics(current_hidden, action)
            
            target_rew = target_rewards[:, k]
            step_reward_loss = self.scalar_loss(reward_predictions, target_rew)
            total_reward_loss += step_reward_loss.item()
            
            step_chance_loss = torch.tensor(0.0, device=self.model.device)
            step_afterstate_value_loss = torch.tensor(0.0, device=self.model.device)
            step_vq_loss = torch.tensor(0.0, device=self.model.device)
            
            if self.model.use_afterstate:
                gt_next_obs = target_obs[:, k]
                
                _, target_indices, step_vq_loss = self.model.encode_chance(gt_next_obs)
                
                chance_logits, afterstate_value_logits = self.model.afterstate_prediction(afterstate)
                
                step_chance_loss = F.cross_entropy(chance_logits, target_indices)
                
                step_afterstate_value_loss = self.scalar_loss(afterstate_value_logits, target_values[:, k])
                
                total_chance_loss += step_chance_loss.item()
                total_afterstate_loss += step_afterstate_value_loss.item()
                
                chance_onehot = F.one_hot(target_indices, self.model.chance_space_size).float()
                next_hidden, _ = self.model.afterstate_dynamics(afterstate, chance_onehot)
                
            else:
                next_hidden = afterstate # Deterministic case
            
            policy_logits, value_logits = self.model.prediction(next_hidden)
            
            target_val = target_values[:, k + 1] 
            target_pol = target_policies[:, k + 1]
            
            step_value_loss = self.scalar_loss(value_logits, target_val)
            step_policy_loss = F.cross_entropy(policy_logits, target_pol)
            
            total_value_loss += step_value_loss.item()
            total_policy_loss += step_policy_loss.item()
            
            step_loss = (step_value_loss + step_policy_loss + step_reward_loss + 
                         step_chance_loss + step_afterstate_value_loss + step_vq_loss)
                         
            loss += step_loss * gradient_scale
            
            current_hidden = next_hidden
            current_hidden.register_hook(lambda grad: grad * 0.5)
            
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()

    def scalar_loss(self, logits, target):
        if target.ndim == 1:
            target_dist = scalar_to_categorical(target, self.support_range)
        else:
            target_dist = target
             
        return -torch.sum(target_dist * F.log_softmax(logits, dim=1), dim=1).mean()
