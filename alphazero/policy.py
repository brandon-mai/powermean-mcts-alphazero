import numpy as np
import torch
import torch.nn.functional as F
import random, os, time, copy
import ray, tqdm

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

@ray.remote(num_gpus=0.1, num_cpus=1) 
class SelfPlayWorker:
    def __init__(self, game_cls, mcts_cls, model_cls, model_args, games_per_worker):
        self.game = game_cls()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Worker initialized on {self.device}")        
        
        self.model = model_cls(**model_args).to("cuda")
        self.model.eval()
        
        self.games_per_worker = games_per_worker 

        self.mcts = mcts_cls(game=self.game, model=self.model)

    def play_game(self, weights, temperature):
        self.model.load_state_dict(weights)
        
        return_memory = []
        
        spgs = [SPG(self.game) for _ in range(self.games_per_worker)]
        
        step = 0
        while spgs: 
            current_states = [spg.state for spg in spgs]
            self.mcts.search(current_states, spgs) 
            
            next_spgs = []
            
            for spg in spgs:
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)
                
                player = self.game.get_current_player(spg.state)
                spg.memory.append((copy.deepcopy(spg.root.state), action_probs, player))
                
                temp_action_probs = action_probs ** (1 / temperature)
                if np.sum(temp_action_probs) == 0: 
                     temp_action_probs = np.ones_like(action_probs) / len(action_probs)
                else:
                    temp_action_probs /= np.sum(temp_action_probs)
                    
                action = np.random.choice(self.game.action_size, p=temp_action_probs)
                
                prev_state = spg.state 
                spg.state = self.game.get_next_state(spg.state, action)
                

                current_player_at_end = self.game.get_opponent(self.game.get_current_player(prev_state))
                value, is_terminal = self.game.get_value_and_terminated(spg.state, current_player_at_end)

                if is_terminal:
                    for hist_state, hist_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == current_player_at_end else self.game.get_opponent_value(value)
                        return_memory.append((hist_state, hist_probs, hist_outcome))
                else:
                    next_spgs.append(spg)
            
            spgs = next_spgs 
            step += 1
            
        return return_memory 

class AlphaZero:
    def __init__(self, model, optimizer, game_cls, mcts_cls, model_cls, model_args,
                 num_parallel_games, temperature, batch_size, 
                 num_iterations, num_selfPlay_iterations, num_epochs, games_per_worker=50):
        self.policy_name = "AlphaZero"
        
        self.game_cls = game_cls
        self.mcts_cls = mcts_cls
        self.model_cls = model_cls
        self.model_args = model_args 

        self.model = model
        self.optimizer = optimizer
        
        self.num_parallel_games = num_parallel_games 
        self.games_per_worker = games_per_worker
        
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_selfPlay_iterations = num_selfPlay_iterations 
        self.num_epochs = num_epochs


        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        print(f"Initializing {num_parallel_games} Ray workers...")
        self.workers = [
            SelfPlayWorker.remote(game_cls, mcts_cls, model_cls, model_args, games_per_worker) 
            for _ in range(num_parallel_games)
        ]

    def selfPlay(self):
            print("------------------------------------------------------------")
            print(f"Starting distributed self-play (Ray)... Target: {self.num_selfPlay_iterations} games")
            
            model_weights = self.model.state_dict()
            cpu_weights = {k: v.cpu() for k, v in model_weights.items()}
            weights_id = ray.put(cpu_weights)
            
            memory = []
            
            games_completed = 0
            games_launched = 0
            worker_futures = {} 

            for i, worker in enumerate(self.workers):
                if games_launched < self.num_selfPlay_iterations:
                    fut = worker.play_game.remote(weights_id, self.temperature)
                    worker_futures[fut] = worker
                    games_launched += self.games_per_worker
            
            with tqdm.tqdm(total=self.num_selfPlay_iterations, desc="Self-Play", unit="game") as pbar:
                while games_completed < self.num_selfPlay_iterations:
                    done_ids, _ = ray.wait(list(worker_futures.keys()), num_returns=1)
                    done_id = done_ids[0]
                    
                    game_memory = ray.get(done_id)
                    memory.extend(game_memory)
                    
                    games_returned = len(game_memory) / self.games_per_worker 
                    pbar.update(self.games_per_worker) 
                    games_completed += self.games_per_worker
                    
                    pbar.set_postfix(samples=len(memory), finished=games_completed)
                    
                    worker = worker_futures.pop(done_id)
                    
                    if games_completed < self.num_selfPlay_iterations:
                        fut = worker.play_game.remote(weights_id, self.temperature)
                        worker_futures[fut] = worker
                    
            return memory

    def train(self, memory):
        print("------------------------------------------------------------")
        print(f"Starting training phase on {len(memory)} samples...")
        random.shuffle(memory)
        batch_losses = []

        total_batches = int(np.ceil(len(memory) / self.batch_size))
        for batchIdx in range(0, len(memory), self.batch_size):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.batch_size)]

            states, policy_targets, value_targets = zip(*sample)
            states, policy_targets, value_targets = states, np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(states)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            l2_weight = 1e-4
            l2_loss = 0.0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and not any(x in name for x in ['bias', 'beta']):
                    l2_loss += torch.sum(param.pow(2))
            
            loss = policy_loss + value_loss + l2_weight * l2_loss
            batch_losses.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_batch = batchIdx // self.batch_size + 1
            if current_batch % 10 == 0:
                print(f"Batch {current_batch}/{total_batches} | Batch size: {len(sample)} | Loss: {loss.item():.6f}")

        avg_loss = np.mean(batch_losses)
        print(f"Training completed.")
        print(f"Average loss across batches: {avg_loss:.6f}")
        print("------------------------------------------------------------")
    
    def learn(self):
        print("============================================================")
        print("Starting AlphaZero learning process...")
        print(f"Total iterations: {self.num_iterations}")
        print(f"Each iteration: {self.num_selfPlay_iterations} self-play games and {self.num_epochs} training epochs")
        print("============================================================")

        for iteration in range(self.num_iterations):
            print(f"\n>>> Iteration {iteration + 1}/{self.num_iterations} started.")
            memory = []
            
            self.model.eval()
            
            memory += self.selfPlay()
            print("All self-play games completed. Starting model training.")
            self.model.train()
            for epoch in range(self.num_epochs):
                print(f"--- Training epoch {epoch + 1}/{self.num_epochs} ---")
                self.train(memory)

            os.makedirs("checkpoint", exist_ok=True)
            torch.save(self.model.state_dict(), 
                        f"checkpoint/policy_{self.policy_name}_"
                        f"game_{self.game_cls.__name__}_"  
                        f"mcts_{self.mcts_cls.func.__name__}_"  
                        f"iter_{iteration + 1}.pt")
            print(f"Model and optimizer checkpoints saved for iteration {iteration + 1}.")

        print("\n============================================================")
        print("AlphaZero learning process finished successfully.")
        print("============================================================")