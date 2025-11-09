"""
Stochastic MuZero Training Loop

MuZero training khác với AlphaZero ở các điểm sau:

1. TARGETS:
   - AlphaZero: chỉ train với (state, policy, value) từ kết quả game
   - MuZero: train với n-step bootstrapped targets:
     + Policy target: từ MCTS
     + Value target: n-step returns (r₀ + γr₁ + ... + γⁿvₙ)
     + Reward target: immediate rewards từ game
     
2. UNROLL:
   - MuZero "unroll" K steps vào future sử dụng dynamics model
   - Train dynamics model để dự đoán đúng next hidden state và rewards
   - Giúp model học được temporal structure
   
3. LOSS FUNCTION:
   - Policy loss: Cross-entropy với MCTS policy
   - Value loss: MSE với n-step returns
   - Reward loss: MSE với actual rewards
   - (Optional) Consistency loss cho stochastic dynamics
"""

import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import copy


class MuZero:
    """
    Stochastic MuZero Learning Algorithm
    """
    def __init__(self, model, optimizer, game, mcts,
                 num_parallel_games, temperature, batch_size,
                 num_iterations, num_selfPlay_iterations, num_epochs,
                 num_unroll_steps=5, td_steps=10, discount=0.997):
        """
        Args:
            model: MuZeroNetwork
            optimizer: Optimizer (Adam, SGD, etc.)
            game: Game environment
            mcts: MuZeroMCTS
            num_parallel_games: Số games chạy parallel trong self-play
            temperature: Temperature cho action sampling
            batch_size: Batch size cho training
            num_iterations: Số iterations tổng cộng
            num_selfPlay_iterations: Số games self-play mỗi iteration
            num_epochs: Số epochs train mỗi iteration
            num_unroll_steps: Số steps unroll trong dynamics (K in paper)
            td_steps: Số steps cho n-step returns (n in paper)
            discount: Discount factor γ
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.mcts = mcts
        
        self.num_parallel_games = num_parallel_games
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_selfPlay_iterations = num_selfPlay_iterations
        self.num_epochs = num_epochs
        
        # MuZero specific parameters
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.discount = discount
    
    def selfPlay(self):
        """
        Self-play phase: AI chơi với chính nó
        
        Thu thập trajectory data:
        - observations: game states thật
        - action_probs: policy từ MCTS
        - values: values từ MCTS
        - rewards: actual rewards từ game
        """
        print("------------------------------------------------------------")
        print("🎮 Bắt đầu self-play phase...")
        
        # Lưu trajectories: mỗi game là một trajectory
        trajectories = []
        
        # Khởi tạo parallel games
        spGames = [SPG(self.game) for _ in range(self.num_parallel_games)]
        player = self.game.get_current_player(self.game.get_initial_state())
        
        total_moves = 0
        completed_games = 0
        
        while len(spGames) > 0:
            states = [spg.state for spg in spGames]
            
            # MCTS search để lấy action probabilities
            self.mcts.search(states, spGames)
            
            # Thu thập data và thực hiện actions
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                # Lấy action probabilities từ visit counts
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)
                
                # Lấy value từ root
                root_value = spg.root.value()
                
                # Lưu vào trajectory
                spg.trajectory.append({
                    'observation': copy.deepcopy(spg.state),
                    'action_probs': action_probs,
                    'value': root_value,
                    'player': player,
                    'reward': 0.0  # Sẽ được update sau
                })
                
                # Sample action với temperature
                temperature_action_probs = action_probs ** (1 / self.temperature)
                if np.sum(temperature_action_probs) == 0:
                    temperature_action_probs = np.ones_like(temperature_action_probs) / len(temperature_action_probs)
                else:
                    temperature_action_probs /= np.sum(temperature_action_probs)
                
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                
                # Lưu action đã chọn
                spg.trajectory[-1]['action'] = action
                
                # Thực hiện action trên environment THẬT
                next_state = self.game.get_next_state(spg.state, action)
                
                # Lấy reward
                reward, is_terminal = self.game.get_value_and_terminated(
                    state=next_state,
                    player=player
                )
                
                # Update reward cho step này
                spg.trajectory[-1]['reward'] = reward
                
                spg.state = next_state
                
                # Kiểm tra terminal
                if is_terminal:
                    completed_games += 1
                    
                    # Compute n-step returns cho toàn bộ trajectory
                    self._compute_targets(spg.trajectory, reward, player)
                    
                    # Lưu trajectory
                    trajectories.append(spg.trajectory)
                    
                    # Remove game
                    del spGames[i]
            
            player = self.game.get_opponent(player)
            total_moves += 1
            
            if total_moves % 10 == 0:
                print(f"📊 Moves: {total_moves} | Games còn lại: {len(spGames)}")
        
        print(f"✅ Self-play hoàn thành!")
        print(f"   Total moves: {total_moves}")
        print(f"   Games completed: {completed_games}")
        print(f"   Trajectories collected: {len(trajectories)}")
        print("------------------------------------------------------------")
        
        return trajectories
    
    def _compute_targets(self, trajectory, final_value, final_player):
        """
        Tính n-step bootstrapped targets cho value
        
        Công thức: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γⁿv_{t+n}
        
        Args:
            trajectory: List of trajectory steps
            final_value: Terminal value của game
            final_player: Player cuối cùng
        """
        # Bootstrap từ cuối trajectory
        bootstrap_value = final_value
        
        for i in range(len(trajectory) - 1, -1, -1):
            step = trajectory[i]
            player = step['player']
            
            # Tính n-step return
            value_target = 0.0
            discount = 1.0
            
            for j in range(i, min(i + self.td_steps, len(trajectory))):
                value_target += discount * trajectory[j]['reward']
                discount *= self.discount
            
            # Thêm bootstrap value nếu chưa terminal
            if i + self.td_steps < len(trajectory):
                value_target += discount * trajectory[i + self.td_steps]['value']
            else:
                # Terminal: dùng final value
                value_target += discount * bootstrap_value
            
            # Flip value nếu player khác
            if player != final_player:
                value_target = self.game.get_opponent_value(value_target)
            
            step['value_target'] = value_target
    
    def train(self, trajectories):
        """
        Training phase: train model với data từ self-play
        
        MuZero training gồm 3 losses:
        1. Policy loss: match MCTS policy
        2. Value loss: match n-step returns  
        3. Reward loss: match actual rewards
        4. (Optional) Dynamics consistency loss
        """
        print("------------------------------------------------------------")
        print("🎓 Bắt đầu training phase...")
        
        # Flatten trajectories thành list of samples
        samples = []
        for traj in trajectories:
            for t, step in enumerate(traj):
                # Mỗi sample gồm:
                # - observation hiện tại
                # - các actions tiếp theo (để unroll)
                # - targets: policy, value, rewards
                sample = {
                    'observation': step['observation'],
                    'action_probs': step['action_probs'],
                    'value_target': step.get('value_target', step['value']),
                }
                
                # Lấy K actions tiếp theo để unroll
                actions = []
                rewards = []
                policies = []
                values = []
                
                for k in range(min(self.num_unroll_steps, len(traj) - t)):
                    future_step = traj[t + k]
                    actions.append(future_step['action'])
                    rewards.append(future_step['reward'])
                    policies.append(future_step['action_probs'])
                    values.append(future_step.get('value_target', future_step['value']))
                
                sample['actions'] = actions
                sample['rewards'] = rewards
                sample['policies'] = policies
                sample['values'] = values
                
                samples.append(sample)
        
        print(f"📦 Tổng số samples: {len(samples)}")
        
        # Training loop
        random.shuffle(samples)
        total_batches = int(np.ceil(len(samples) / self.batch_size))
        
        epoch_losses = []
        
        for batchIdx in range(0, len(samples), self.batch_size):
            batch = samples[batchIdx:min(len(samples), batchIdx + self.batch_size)]
            
            # Skip batches with size 1 (BatchNorm requirement)
            if len(batch) < 2:
                continue
            
            # Compute loss
            loss, loss_dict = self._compute_loss(batch)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping để ổn định training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Logging
            current_batch = batchIdx // self.batch_size + 1
            if current_batch % 10 == 0:
                print(f"📈 Batch {current_batch}/{total_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Policy: {loss_dict['policy_loss']:.4f} | "
                      f"Value: {loss_dict['value_loss']:.4f} | "
                      f"Reward: {loss_dict['reward_loss']:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"✅ Training hoàn thành! Avg loss: {avg_loss:.4f}")
        print("------------------------------------------------------------")
    
    def _compute_loss(self, batch):
        """
        Tính loss cho MuZero
        
        Loss gồm:
        1. Initial step: policy loss + value loss
        2. Unroll steps: policy loss + value loss + reward loss
        """
        batch_size = len(batch)
        device = self.model.device
        
        # === INITIAL INFERENCE ===
        observations = [sample['observation'] for sample in batch]
        
        # Encode observations nếu chưa encoded
        encoded_observations = self.game.get_encoded_state(observations)
        
        # Ensure model is in training mode for BatchNorm
        self.model.train()
        hidden_states, policy_logits, values = self.model.initial_inference(encoded_observations)
        
        # Initial policy loss
        policy_targets = torch.tensor(
            np.array([sample['action_probs'] for sample in batch]),
            dtype=torch.float32, device=device
        )
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        
        # Initial value loss
        value_targets = torch.tensor(
            np.array([[sample['value_target']] for sample in batch]),
            dtype=torch.float32, device=device
        )
        # Normalize value targets về [-1, 1]
        value_targets = value_targets * 2 - 1
        
        # Handle categorical vs scalar values
        if values.shape[-1] > 1:
            # Categorical: convert to scalar first
            from muzero.model import categorical_to_scalar
            value_support_range = getattr(self.model, 'value_support_range', (-300, 301, 1))
            values_scalar = categorical_to_scalar(
                torch.softmax(values, dim=-1),
                value_support_range
            )
            value_loss = F.mse_loss(values_scalar, value_targets)
        else:
            # Scalar values
            value_loss = F.mse_loss(values, value_targets)
        
        # === UNROLL DYNAMICS ===
        reward_loss = 0.0
        
        # Unroll K steps
        for k in range(self.num_unroll_steps):
            # Collect actions cho step k
            actions = []
            for sample in batch:
                if k < len(sample['actions']):
                    actions.append(sample['actions'][k])
                else:
                    actions.append(0)  # Padding (sẽ bị mask)
            
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            
            # Recurrent inference
            hidden_states, rewards, policy_logits, values = self.model.recurrent_inference(
                hidden_states, actions
            )
            
            # Collect targets cho step k
            policy_targets_k = []
            value_targets_k = []
            reward_targets_k = []
            masks = []
            
            for i, sample in enumerate(batch):
                if k < len(sample['policies']):
                    policy_targets_k.append(sample['policies'][k])
                    value_targets_k.append(sample['values'][k])
                    reward_targets_k.append(sample['rewards'][k])
                    masks.append(1.0)
                else:
                    # Padding
                    policy_targets_k.append(np.zeros(self.game.action_size))
                    value_targets_k.append(0.0)
                    reward_targets_k.append(0.0)
                    masks.append(0.0)
            
            policy_targets_k = torch.tensor(
                np.array(policy_targets_k),
                dtype=torch.float32, device=device
            )
            value_targets_k = torch.tensor(
                np.array(value_targets_k).reshape(-1, 1),
                dtype=torch.float32, device=device
            )
            reward_targets_k = torch.tensor(
                np.array(reward_targets_k).reshape(-1, 1),
                dtype=torch.float32, device=device
            )
            masks = torch.tensor(
                np.array(masks).reshape(-1, 1),
                dtype=torch.float32, device=device
            )
            
            # Normalize targets
            value_targets_k = value_targets_k * 2 - 1
            reward_targets_k = reward_targets_k * 2 - 1
            
            # Handle categorical values/rewards
            if values.shape[-1] > 1:
                # Categorical: convert to scalar
                from muzero.model import categorical_to_scalar
                value_support_range = getattr(self.model, 'value_support_range', (-300, 301, 1))
                values_scalar = categorical_to_scalar(
                    torch.softmax(values, dim=-1),
                    value_support_range
                )
            else:
                values_scalar = values
            
            if rewards.shape[-1] > 1:
                # Categorical: convert to scalar
                from muzero.model import categorical_to_scalar
                reward_support_range = getattr(self.model, 'reward_support_range', (-300, 301, 1))
                rewards_scalar = categorical_to_scalar(
                    torch.softmax(rewards, dim=-1),
                    reward_support_range
                )
            else:
                rewards_scalar = rewards
            
            # Compute losses với masking
            policy_loss += (F.cross_entropy(policy_logits, policy_targets_k, reduction='none').mean() * masks).mean()
            value_loss += (F.mse_loss(values_scalar, value_targets_k, reduction='none') * masks).mean()
            reward_loss += (F.mse_loss(rewards_scalar, reward_targets_k, reduction='none') * masks).mean()
        
        # === TOTAL LOSS ===
        # Weight các losses
        total_loss = policy_loss + value_loss + reward_loss
        
        # L2 regularization
        l2_weight = 1e-4
        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                l2_loss += torch.sum(param.pow(2))
        
        total_loss += l2_weight * l2_loss
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss if isinstance(reward_loss, float) else reward_loss.item(),
            'l2_loss': l2_loss.item()
        }
        
        return total_loss, loss_dict
    
    def learn(self):
        """
        Main learning loop cho MuZero
        """
        print("=" * 80)
        print("🚀 BẮT ĐẦU STOCHASTIC MUZERO LEARNING")
        print("=" * 80)
        print(f"📊 Cấu hình:")
        print(f"   - Iterations: {self.num_iterations}")
        print(f"   - Self-play games/iteration: {self.num_selfPlay_iterations}")
        print(f"   - Training epochs/iteration: {self.num_epochs}")
        print(f"   - Unroll steps: {self.num_unroll_steps}")
        print(f"   - TD steps: {self.td_steps}")
        print(f"   - Discount: {self.discount}")
        print("=" * 80)
        
        for iteration in range(self.num_iterations):
            print(f"\n{'='*80}")
            print(f"🔄 ITERATION {iteration + 1}/{self.num_iterations}")
            print(f"{'='*80}")
            
            # Self-play phase
            trajectories = []
            self.model.eval()
            
            for i in range(self.num_selfPlay_iterations // self.num_parallel_games):
                print(f"\n🎮 Self-play batch {i + 1}/{self.num_selfPlay_iterations // self.num_parallel_games}")
                batch_trajectories = self.selfPlay()
                trajectories.extend(batch_trajectories)
            
            print(f"\n✅ Self-play hoàn thành! Collected {len(trajectories)} trajectories")
            
            # Training phase
            self.model.train()
            for epoch in range(self.num_epochs):
                print(f"\n📚 Training epoch {epoch + 1}/{self.num_epochs}")
                self.train(trajectories)
            
            # Save checkpoint
            os.makedirs("checkpoint", exist_ok=True)
            checkpoint_path = f"checkpoint/{self.mcts.name}_{self.game.name}_iteration_{iteration + 1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        
        print("\n" + "=" * 80)
        print("🎉 STOCHASTIC MUZERO LEARNING HOÀN THÀNH!")
        print("=" * 80)


class SPG:
    """
    Self-Play Game: quản lý state của một game đang self-play
    """
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.trajectory = []
        self.root = None
        self.node = None


if __name__ == "__main__":
    print("Stochastic MuZero Training Implementation")
    print("\n🎯 Đặc điểm chính:")
    print("✅ N-step bootstrapped returns")
    print("✅ K-step unroll với dynamics model")
    print("✅ 3 losses: policy, value, reward")
    print("✅ Stochastic dynamics với latent variables")
    print("\n📖 Training flow:")
    print("1. Self-play với MCTS trên learned model")
    print("2. Thu thập trajectories (obs, actions, rewards, MCTS policies)")
    print("3. Compute n-step returns")
    print("4. Train model với unrolled dynamics")
    print("5. Repeat!")

