import copy 
import ray
import tqdm
import torch
import numpy as np
import argparse, sys, itertools, time, json, os, random, string
from collections import defaultdict
from functools import partial

# Add project subdirectories to path for module resolution
sys.path.append('powermean-mcts-alphazero/')
sys.path.append('powermean-mcts-alphazero/games')
sys.path.append('powermean-mcts-alphazero/alphazero')
sys.path.append('powermean-mcts-alphazero/mcts')

from games import (
    ConnectFour, Breakthrough, TicTacToe, Havannah, Y, 
    Stochastic_ConnectFour, Stochastic_Breakthrough, 
    Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y
)
from alphazero import ResNet
from mcts import Stochastic_Powermean_UCT, PUCT


class SPG:
    """
    Lightweight container for a single game instance during the tournament.
    
    Optimizations:
    - Uses `__slots__` to restrict attribute creation, significantly reducing 
      memory footprint when instantiating thousands of objects.
    - Speeds up attribute access compared to standard __dict__.
    """
    __slots__ = ['game', 'state', 'memory', 'root'] 
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None

def random_suffix(k=6):
    """Generates a random suffix for unique filename creation."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def get_game_class(game_name):
    """Maps game name strings to their corresponding class definitions."""
    mapping = {
        "ConnectFour": ConnectFour, "Breakthrough": Breakthrough, "TicTacToe": TicTacToe,
        "Havannah": Havannah, "Y": Y, "Stochastic_ConnectFour": Stochastic_ConnectFour, 
        "Stochastic_Breakthrough": Stochastic_Breakthrough, "Stochastic_TicTacToe": Stochastic_TicTacToe, 
        "Stochastic_Havannah": Stochastic_Havannah, "Stochastic_Y": Stochastic_Y
    }
    return mapping[game_name]

def get_model_config_and_class(game_name):
    """Returns the Model Class and specific architecture config based on the game."""
    if "ConnectFour" in game_name:
        config = {"num_resBlocks": 9, "num_hidden": 128}
    elif "Breakthrough" in game_name:
        config = {"num_resBlocks": 12, "num_hidden": 128}
    else:
        config = {"num_resBlocks": 5, "num_hidden": 64} 
    return ResNet, config


@ray.remote
class TournamentWorker:
    """
    Ray Actor responsible for running matches between two agents.
    Designed to persist in GPU memory to avoid reloading models repeatedly.
    """
    def __init__(self, game_cls, model_cls, model_args):
        self.game = game_cls()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model once per worker
        self.model = model_cls(game=self.game, device=self.device, **model_args)
        self.model.eval()
        
        print(f"Tournament Worker initialized on {self.device}")

    def update_weights(self, weights):
        """Updates the worker's model with new weights broadcasted from the driver."""
        self.model.load_state_dict(weights)

    @torch.inference_mode() 
    def run_match(self, num_games, temperature, starting_player, 
                  mcts_cls_A, mcts_args_A, mcts_cls_B, mcts_args_B):
        """
        Executes a batch of games between Agent A and Agent B.
        
        Optimizations:
        - @torch.inference_mode(): Disables gradient tracking for max speed.
        - Vectorized MCTS: Runs searches for all games in the batch simultaneously.
        """

        # Initialize MCTS instances for both agents
        mcts_A = mcts_cls_A(game=self.game, model=self.model, **mcts_args_A)
        mcts_B = mcts_cls_B(game=self.game, model=self.model, **mcts_args_B)
        
        # Map bots to player indices based on starting configuration
        bots = {}
        if starting_player == 0:
            bots[0], bots[1] = mcts_A, mcts_B
        else:
            bots[0], bots[1] = mcts_B, mcts_A

        spgs = [SPG(self.game) for _ in range(num_games)]
        results = []
        
        # Pre-allocate probability vector to avoid re-creation in loop
        action_size = self.game.action_size
        ones_prob = np.ones(action_size) / action_size

        while spgs:
            # Group games by current player to batch MCTS requests
            player_spgs = {0: [], 1: []}
            for spg in spgs:
                player_spgs[self.game.get_current_player(spg.state)].append(spg)
            
            # Execute Batch MCTS Search
            if player_spgs[0]:
                bots[0].search([s.state for s in player_spgs[0]], player_spgs[0])
            if player_spgs[1]:
                bots[1].search([s.state for s in player_spgs[1]], player_spgs[1])

            next_spgs = []
            
            # Process results and step the environment
            for spg in spgs:
                # 1. Extract Action Probabilities from MCTS Root
                if spg.root is not None:

                    # Note: Assumes root.children is populated. 
                    # For optimization, checking child.visit_count is sufficient.
                    counts = np.zeros(action_size)
                    for child in spg.root.children.values(): 
                        counts[child.action_taken] = child.visit_count
                    
                    sum_counts = np.sum(counts)
                    if sum_counts > 0:
                        action_probs = counts / sum_counts
                    else:
                        action_probs = ones_prob
                else:
                    action_probs = ones_prob

                # 2. Apply Temperature for action selection
                # Optimization: Use argmax for low temp to avoid expensive power ops
                if temperature < 1e-3:
                    action = np.argmax(action_probs)
                else:
                    # Safe power operation with normalization
                    action_probs = action_probs ** (1 / temperature)
                    sum_prob = np.sum(action_probs)
                    if sum_prob > 0:
                        action_probs /= sum_prob
                    else:
                        action_probs = ones_prob
                    action = np.random.choice(action_size, p=action_probs)
                
                # 3. Game Step & Terminal Check
                current_player_idx = self.game.get_current_player(spg.state)
                spg.state = self.game.get_next_state(spg.state, action)
                
                # Memory Optimization:
                # Set root to None to allow GC to collect the old tree immediately.
                # Prioritizes memory safety over Tree Reuse for this implementation.
                spg.root = None 
                
                value, is_terminal = self.game.get_value_and_terminated(spg.state, current_player_idx)
                
                if is_terminal:
                    # Determine Winner
                    if value == 1.0:
                        winner = current_player_idx
                    elif value == 0.0:
                        winner = self.game.get_opponent(current_player_idx)
                    else:
                        winner = None # Draw
                    
                    results.append({'winner': winner, 'outcome': 1.0 if winner is not None else 0.5})
                else:
                    next_spgs.append(spg)
            
            spgs = next_spgs

        return results
    
def run_single_tournament_distributed(game_cls, mcts_list, args, run_id, workers, device):
    """
    Orchestrates a single tournament run (round-robin or specific matchups) across distributed workers.
    """
    checkpoint_path = mcts_list[0]["checkpoint_path"]
    
    print(f"  Loading weights from: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location='cpu')
    
    # Put weights into Ray's Object Store for efficient broadcasting to workers
    weights_ref = ray.put(weights) 
    
    print(f"  Broadcasting weights to {len(workers)} workers...")
    ray.get([w.update_weights.remote(weights_ref) for w in workers])
    print("  Weights synchronized.")

    run_results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    matchups = list(itertools.combinations(mcts_list, 2))
    
    all_tasks_metadata = []
    
    print(f"  Generating tasks for {len(matchups)} matchups...")
    
    # Prepare task definitions
    for m1, m2 in matchups:
        cls_A, args_A = m1["mcts_cls"], m1["mcts_args"]
        cls_B, args_B = m2["mcts_cls"], m2["mcts_args"]
        
        # Play both as Player 1 and Player 2 for fairness
        for start_p in [0, 1]: 
            games_remaining = args.num_games_per_pair
            
            while games_remaining > 0:
                batch_size = min(games_remaining, args.games_per_worker)
                
                task_data = {
                    "batch_size": batch_size,
                    "temperature": args.temperature,
                    "starting_player": 0, 
                    "cls_A": cls_A, "args_A": args_A,
                    "cls_B": cls_B, "args_B": args_B,
                    "p1_name": m1["name"] if start_p == 0 else m2["name"], 
                    "p2_name": m2["name"] if start_p == 0 else m1["name"]
                }
                
                if start_p == 1:
                    task_data["cls_A"], task_data["cls_B"] = cls_B, cls_A
                    task_data["args_A"], task_data["args_B"] = args_B, args_A
                
                all_tasks_metadata.append(task_data)
                games_remaining -= batch_size

    total_games_scheduled = sum(t["batch_size"] for t in all_tasks_metadata)
    print(f"  Total tasks created: {len(all_tasks_metadata)} | Total games: {total_games_scheduled}")
    
    # Scheduler Optimization: Use Dictionary for O(1) task lookup
    future_to_task = {} 
    
    worker_index = 0
    num_workers = len(workers)
    
    # Initial Dispatch: Fill workers with tasks
    # Limit initial dispatch to avoid overloading Ray's scheduler if tasks > workers
    initial_batch_size = min(len(all_tasks_metadata), len(workers) * 2) 
    
    for i in range(initial_batch_size):
        task = all_tasks_metadata[i]
        worker = workers[worker_index % num_workers]
        worker_index += 1
        
        fut = worker.run_match.remote(
            task["batch_size"], task["temperature"], task["starting_player"],
            task["cls_A"], task["args_A"], task["cls_B"], task["args_B"]
        )
        # Store task metadata mapped by the future object
        future_to_task[fut] = {"p1": task["p1_name"], "p2": task["p2_name"]}

    next_task_idx = initial_batch_size

    # Event Loop: Process results as they arrive
    with tqdm.tqdm(total=total_games_scheduled, desc=f"Tournament Run #{run_id + 1}") as pbar:
        while future_to_task:
            # 1. Efficiently wait for the first available completion (O(N) -> O(1) with list(keys))
            done_ids, _ = ray.wait(list(future_to_task.keys()), num_returns=1)
            done_id = done_ids[0]
            
            # 2. Retrieve Metadata (O(1) operation)
            task_info = future_to_task.pop(done_id)
            p1_name = task_info["p1"]
            p2_name = task_info["p2"]
            
            # 3. Process Results
            try:
                batch_results = ray.get(done_id)
                for result in batch_results:
                    pbar.update(1) 
                    
                    if result['outcome'] == 1.0:
                        if result['winner'] == 0: 
                            run_results[p1_name]["win"] += 1
                            run_results[p2_name]["loss"] += 1
                        else: 
                            run_results[p2_name]["win"] += 1
                            run_results[p1_name]["loss"] += 1
                    else: 
                        run_results[p1_name]["draw"] += 1
                        run_results[p2_name]["draw"] += 1
            except Exception as e:
                print(f"Error in batch task: {e}")

            # 4. Schedule Next Task (Pipeline approach)
            if next_task_idx < len(all_tasks_metadata):
                task = all_tasks_metadata[next_task_idx]
                next_task_idx += 1
                
                # Round-robin worker assignment
                worker = workers[worker_index % num_workers]
                worker_index += 1
                
                new_fut = worker.run_match.remote(
                    task["batch_size"], task["temperature"], task["starting_player"],
                    task["cls_A"], task["args_A"], task["cls_B"], task["args_B"]
                )
                future_to_task[new_fut] = {"p1": task["p1_name"], "p2": task["p2_name"]}

    return run_results

def aggregate_results(all_runs_results):
    """Aggregates and calculates statistics (Mean/Std) from multiple tournament runs."""
    aggregated = defaultdict(lambda: {"win": [], "loss": [], "draw": []})
    
    for run_results in all_runs_results:
        for name, record in run_results.items():
            aggregated[name]["win"].append(record["win"])
            aggregated[name]["loss"].append(record["loss"])
            aggregated[name]["draw"].append(record["draw"])
    
    final_results = {}
    for name, records in aggregated.items():
        wins = np.array(records["win"])
        losses = np.array(records["loss"])
        draws = np.array(records["draw"])
        total_games = wins + losses + draws
        
        # Calculate win rate safely
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rates = np.where(total_games > 0, (wins / total_games * 100), 0)
        
        final_results[name] = {
            "win_mean": float(np.mean(wins)),
            "win_std": float(np.std(wins)),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "draw_mean": float(np.mean(draws)),
            "draw_std": float(np.std(draws)),
            "win_rate_mean": float(np.mean(win_rates)),
            "win_rate_std": float(np.std(win_rates)),
        }
    return final_results


def run_tournament(args):
    """Main entry point for running the tournament pipeline."""
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        project_root = os.path.abspath('powermean-mcts-alphazero') 
        
        # Ensure workers can import local modules
        module_paths = [
            project_root,
            os.path.join(project_root, 'games'),
            os.path.join(project_root, 'alphazero'),
            os.path.join(project_root, 'mcts'),
        ]
        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": module_paths})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_cls = get_game_class(args.game)
    model_cls, model_config = get_model_config_and_class(args.game)
    
    num_gpus = torch.cuda.device_count()

    # Allocate GPU resources per worker
    gpu_per_worker = (num_gpus / args.num_games_parallel) if num_gpus > 0 else 0
    WorkerRemote = TournamentWorker.options(num_gpus=gpu_per_worker)

    print("Initializing workers...")
    workers = [
        WorkerRemote.remote(
            game_cls=game_cls, 
            model_cls=model_cls,
            model_args=model_config
        ) for _ in range(args.num_games_parallel)
    ]

    mcts_args_base = {
        "C": args.C, 
        "dirichlet_epsilon": args.dirichlet_epsilon, 
        "dirichlet_alpha": args.dirichlet_alpha, 
        "num_searches": args.num_searches
    }

    all_checkpoint_results = {}
    
    # Iterate over checkpoints to evaluate
    for checkpoint_idx, checkpoint_path in enumerate(args.checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"PROCESSING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")
        
        mcts_list = []
        
        # Create configurations for Stochastic PowerMean MCTS (with varied 'p')
        for p in args.p:
            args_stoch = mcts_args_base.copy()
            args_stoch["p"] = p
            args_stoch["gamma"] = args.gamma
            
            mcts_list.append({   
                "name": f"Stochastic_Powermean_UCT_p={p}", 
                "checkpoint_path": checkpoint_path,
                "mcts_cls": Stochastic_Powermean_UCT, 
                "mcts_args": args_stoch 
            })

        # Add Baseline PUCT
        mcts_list.append({
            "name": "PUCT",
            "checkpoint_path": checkpoint_path,
            "mcts_cls": PUCT, 
            "mcts_args": mcts_args_base 
        })
        
        all_runs_results = []
        for run_id in range(args.num_runs):
            results = run_single_tournament_distributed(
                game_cls, mcts_list, args, run_id, workers, device
            )
            all_runs_results.append(results)
            
            print(f"  Run {run_id+1} Summary:")
            for k, v in results.items():
                print(f"    {k}: {v}")

        # Save results
        checkpoint_results = aggregate_results(all_runs_results)
        checkpoint_basename = os.path.basename(checkpoint_path)
        all_checkpoint_results[checkpoint_basename] = checkpoint_results
        
        os.makedirs("individual_result", exist_ok=True)
        suffix = random_suffix()
        fn = f"individual_result/{os.path.splitext(checkpoint_basename)[0]}_{suffix}.json"
        
        save_data = {
            "config": vars(args),
            "results": checkpoint_results
        }
        
        # Helper for JSON serialization of Numpy types
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)

        with open(fn, "w") as f:
            json.dump(save_data, f, indent=4, default=convert)
        print(f"  Saved: {fn}")

    # Final Summary Save
    os.makedirs("evaluate_result", exist_ok=True)
    suffix = random_suffix()
    final_fn = f"evaluate_result/summary_{suffix}.json"
    
    final_save = {
        "config": vars(args),
        "results": all_checkpoint_results
    }
    with open(final_fn, "w") as f:
        json.dump(final_save, f, indent=4, default=convert)
    print(f"\nDone. Final results saved to {final_fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Tournament for AlphaZero models.")
    parser.add_argument("--game", type=str, default="ConnectFour",
                            choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                    "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                    "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y"],
                            help="Game to play.")  
    parser.add_argument("--checkpoint_paths", nargs='+', required=True, help="List of model checkpoints to evaluate.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of repeated runs for statistical significance.")
    parser.add_argument("--num_searches", type=int, default=50, help="Number of MCTS simulations per move.")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant.")
    parser.add_argument("--p", type=float, nargs='+', default=[1.5], help="List of 'p' values for PowerMean UCT.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma factor for PowerMean.")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.0, help="Noise for tournament (usually 0).") 
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--num_games_per_pair", type=int, default=10, help="Total games per matchup per run.")
    
    parser.add_argument("--games_per_worker", type=int, default=10, 
                        help="Batch size: Number of games running in parallel on a single worker.")
    
    parser.add_argument("--num_games_parallel", type=int, default=4, help="Number of Ray workers.")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for move selection.") 
    
    args = parser.parse_args()
    
    run_tournament(args)