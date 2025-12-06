import numpy as np
import torch
import torch.nn.functional as F
import random, os, time, copy
import ray, tqdm

class SPG:
    """
    Container for a Single Player Game (SPG) state during self-play.
    """
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

@ray.remote
class SelfPlayWorker:
    """
    Ray Actor responsible for running self-play games in parallel.
    """
    def __init__(self, game_cls, mcts_cls, model_cls, model_args, games_per_worker):
        self.game = game_cls()
        
        # Ray automatically manages resources. Check if GPU is assigned to this worker.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Override model device configuration to match the worker's assigned hardware
        worker_model_args = model_args.copy()
        worker_model_args["device"] = self.device
        
        self.model = model_cls(**worker_model_args)
        self.model.to(self.device) 
        self.model.eval()
        
        self.games_per_worker = games_per_worker 
        self.mcts_cls = mcts_cls

    def play_game(self, weights, temperature):
        """
        Executes a batch of self-play games.
        """
        # Load the latest weights from the driver
        self.model.load_state_dict(weights)
        
        # Instantiate MCTS. 
        # Note: mcts_cls is a 'partial' function with hyperparameters (C, p, gamma...) pre-filled.
        self.mcts = self.mcts_cls(game=self.game, model=self.model)

        return_memory = []
        spgs = [SPG(self.game) for _ in range(self.games_per_worker)]
        
        while spgs: 
            current_states = [spg.state for spg in spgs]
            
            # Execute MCTS Search (Vectorized/Batch processing)
            self.mcts.search(current_states, spgs) 
            
            next_spgs = []
            
            for spg in spgs:
                # Retrieve statistics from the Root Node (Vectorized Optimized Node)
                visits = spg.root.child_visits
                actions = spg.root.valid_actions
                
                # Construct probability vector
                action_probs = np.zeros(self.game.action_size)
                action_probs[actions] = visits
                action_probs /= np.sum(action_probs)
                
                player = self.game.get_current_player(spg.state)
                
                # --- MEMORY OPTIMIZATION ---
                # 1. Store only canonical features (Numpy) instead of full State objects to save RAM.
                canonical_input = self.game.get_encoded_state(spg.state)
                
                # 2. Store probabilities as float16 to further reduce memory footprint.
                spg.memory.append((canonical_input, action_probs.astype(np.float16), player))
                # ---------------------------

                # Select action based on Temperature
                if temperature == 0:
                    # Greedy selection
                    action = actions[np.argmax(visits)]
                else:
                    # Stochastic selection
                    temp_probs = action_probs ** (1 / temperature)
                    prob_sum = np.sum(temp_probs)
                    if prob_sum == 0:
                        temp_probs = np.ones_like(action_probs) / len(action_probs)
                    else:
                        temp_probs /= prob_sum
                    action = np.random.choice(self.game.action_size, p=temp_probs)
                
                # Step the environment
                spg.state = self.game.get_next_state(spg.state, action)
                
                # Check for Terminal State
                opponent = self.game.get_opponent(player)
                value, is_terminal = self.game.get_value_and_terminated(spg.state, opponent)
                
                if is_terminal:
                    # Normalize value from [0, 1] to [-1, 1] for AlphaZero training
                    value = 2 * value - 1
                    
                    # Process game history
                    for hist_state, hist_probs, hist_player in spg.memory:
                        # hist_state is already a numpy array (feature) -> Safe for memory
                        # Flip value perspective based on player turn
                        hist_outcome = value if hist_player == opponent else -value
                        return_memory.append((hist_state, hist_probs, hist_outcome))
                else:
                    next_spgs.append(spg)
            
            spgs = next_spgs 
            
        return return_memory 

class AlphaZero:
    """
    Main AlphaZero Training Pipeline managing distributed Self-Play and Training.
    """
    def __init__(self, model, optimizer, game_cls, mcts_cls, model_cls, model_args,
                 num_parallel_games, temperature, batch_size, 
                 num_iterations, num_selfPlay_iterations, num_epochs, games_per_worker=50):
        self.policy_name = "AlphaZero"
        
        self.game_cls = game_cls
        self.mcts_cls = mcts_cls # This is a partial function
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
        
        num_gpus_available = torch.cuda.device_count()
        # Allocate fractional GPU resources per worker if available
        gpu_per_worker = num_gpus_available / num_parallel_games if num_parallel_games > 0 else 0
            
        self.workers = [
            SelfPlayWorker.options(num_gpus=gpu_per_worker).remote(
                game_cls, mcts_cls, model_cls, model_args, games_per_worker
            ) 
            for _ in range(num_parallel_games)
        ]

    def selfPlay(self):
        """
        Orchestrates distributed self-play using Ray.
        """
        print("------------------------------------------------------------")
        print(f"Starting distributed self-play (Ray)... Target: {self.num_selfPlay_iterations} games")
        
        # Push model weights to Ray's Object Store (Shared Memory) for efficient distribution
        model_weights = self.model.state_dict()
        cpu_weights = {k: v.cpu() for k, v in model_weights.items()}
        weights_id = ray.put(cpu_weights)
        
        memory = []
        
        games_completed = 0
        games_launched = 0
        worker_futures = {} 

        # Launch initial batch of workers
        for i, worker in enumerate(self.workers):
            if games_launched < self.num_selfPlay_iterations:
                fut = worker.play_game.remote(weights_id, self.temperature)
                worker_futures[fut] = worker
                games_launched += self.games_per_worker
        
        # Monitor progress and collect results
        with tqdm.tqdm(total=self.num_selfPlay_iterations, desc="Self-Play", unit="game") as pbar:
            while games_completed < self.num_selfPlay_iterations:
                # Wait for at least 1 worker to finish
                done_ids, _ = ray.wait(list(worker_futures.keys()), num_returns=1)
                done_id = done_ids[0]
                
                # Retrieve result from Object Store
                game_memory = ray.get(done_id)
                memory.extend(game_memory)
                
                # Update progress bar
                pbar.update(self.games_per_worker) 
                games_completed += self.games_per_worker
                pbar.set_postfix(samples=len(memory), finished=games_completed)
                
                # Relaunch worker if target not reached
                worker = worker_futures.pop(done_id)
                if games_completed < self.num_selfPlay_iterations:
                    fut = worker.play_game.remote(weights_id, self.temperature)
                    worker_futures[fut] = worker
                
        return memory

    def train(self, memory):
        """
        Trains the neural network using the collected self-play data.
        """
        print("------------------------------------------------------------")
        print(f"Starting training phase on {len(memory)} samples...")
        random.shuffle(memory)
        batch_losses = []

        total_batches = int(np.ceil(len(memory) / self.batch_size))
        for batchIdx in range(0, len(memory), self.batch_size):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.batch_size)]

            # Unpack sample
            states, policy_targets, value_targets = zip(*sample)
            
            # Convert list of numpy features (from selfPlay) to Tensor batch
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.model.device)
            
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(states)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            
            # L2 Regularization
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
                print(f"Batch {current_batch}/{total_batches} | Loss: {loss.item():.6f}")

        avg_loss = np.mean(batch_losses)
        print(f"Training completed. Avg Loss: {avg_loss:.6f}")
        print("------------------------------------------------------------")
    
    def learn(self):
        """
        Main learning loop: Self-Play -> Train -> Checkpoint.
        """
        print("============================================================")
        print(f"Starting AlphaZero learning. Total Iters: {self.num_iterations}")
        print("============================================================")

        for iteration in range(self.num_iterations):
            print(f"\n>>> Iteration {iteration + 1}/{self.num_iterations}")
            memory = []
            
            # Phase 1: Self-Play
            self.model.eval()
            memory += self.selfPlay()
            
            # Phase 2: Training
            print("Self-play finished. Training model...")
            self.model.train()
            for epoch in range(self.num_epochs):
                print(f"--- Epoch {epoch + 1}/{self.num_epochs} ---")
                self.train(memory)

            # Checkpointing
            os.makedirs("checkpoint", exist_ok=True)
            # Extract the actual function name if mcts_cls is a partial object
            mcts_name = self.mcts_cls.func.__name__ if hasattr(self.mcts_cls, 'func') else self.mcts_cls.__name__
            
            save_path = f"checkpoint/policy_{self.policy_name}_{self.game_cls.__name__}_{mcts_name}_iter_{iteration + 1}.pt"
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

        print("\n============================================================")
        print("Done.")