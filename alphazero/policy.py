import numpy as np
import torch
import torch.nn.functional as F
import random

class AlphaZero:
    def __init__(self, model, optimizer, game, mcts,
                 num_parallel_games, temperature, batch_size,
                 num_iterations, num_selfPlay_iterations, num_epochs):
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

    def selfPlay(self):
        print("------------------------------------------------------------")
        print("Starting self-play phase...")
        return_memory = []
        player = 1
        
        spGames = [SPG(self.game) for _ in range(self.num_parallel_games)]
        total_moves = 0
        completed_games = 0

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                action_probs = np.zeros(self.game.action_size)

                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))
                temperature_action_probs = action_probs ** (1 / self.temperature)

                if np.sum(temperature_action_probs) == 0:
                    temperature_action_probs = np.ones_like(temperature_action_probs) / len(temperature_action_probs)
                else:
                    temperature_action_probs /= np.sum(temperature_action_probs)
                
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                spg.state = self.game.get_next_state(spg.state, action)
                
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)
                if is_terminal:
                    completed_games += 1
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)
            total_moves += 1

            if total_moves % 10 == 0:
                print(f"[Update] Total moves simulated so far: {total_moves} | "
                    f"Remaining active games: {len(spGames)}")

        print(f"Self-play completed.")
        print(f"Total simulated moves: {total_moves}")
        print(f"Training samples generated: {len(return_memory)}")
        print("------------------------------------------------------------")
        return return_memory

                
    def train(self, memory):
        print("------------------------------------------------------------")
        print(f"Starting training phase on {len(memory)} samples...")
        random.shuffle(memory)
        batch_losses = []

        total_batches = int(np.ceil(len(memory) / self.batch_size))
        for batchIdx in range(0, len(memory), self.batch_size):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.batch_size)]

            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            
            loss = policy_loss + value_loss
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
            for i in range(self.num_selfPlay_iterations // self.num_parallel_games):
                print(f"--- Running self-play batch {i + 1}/{self.num_selfPlay_iterations // self.num_parallel_games} ---")
                memory += self.selfPlay()
            
            print("All self-play games completed. Starting model training.")
            self.model.train()
            for epoch in range(self.num_epochs):
                print(f"--- Training epoch {epoch + 1}/{self.num_epochs} ---")
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"{self.mcts.name}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
            print(f"Model and optimizer checkpoints saved for iteration {iteration + 1}.")

        print("\n============================================================")
        print("AlphaZero learning process finished successfully.")
        print("============================================================")


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
