import copy 
import ray
import tqdm
import torch
import numpy as np
import argparse, sys, itertools, time, json, os, random, string
from collections import defaultdict
from functools import partial

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'games'))
sys.path.append(os.path.join(os.getcwd(), 'alphazero'))
sys.path.append(os.path.join(os.getcwd(), 'mcts'))
sys.path.append(os.path.join(os.getcwd(), 'stochastic_muzero'))

from games import (
    ConnectFour, Breakthrough, TicTacToe, Havannah, Y, 
    Stochastic_ConnectFour, Stochastic_Breakthrough, 
    Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y,
    Stochastic_MiniGrid_8x8_Empty, Stochastic_MiniGrid_6x6_Empty_Random,
    Stochastic_FrozenLake_4x4_Random_Map, Stochastic_FrozenLake_8x8_Random_Map,
    Taxi_Is_Raining_Fickle_Passenger
)

from stochastic_muzero.model import StochasticMuZeroNetwork
from stochastic_muzero.mcts import StochasticMuZeroMCTS as ClassicMCTS
from stochastic_muzero.powermean_mcts import StochasticMuZeroMCTS as PowerMeanMCTS

class StochasticMuZeroSPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.root = None
        self.total_reward = 0.0

def get_game_class(game_name):
    mapping = {
        "ConnectFour": ConnectFour, "Breakthrough": Breakthrough, "TicTacToe": TicTacToe,
        "Havannah": Havannah, "Y": Y, "Stochastic_ConnectFour": Stochastic_ConnectFour, 
        "Stochastic_Breakthrough": Stochastic_Breakthrough, "Stochastic_TicTacToe": Stochastic_TicTacToe, 
        "Stochastic_Havannah": Stochastic_Havannah, "Stochastic_Y": Stochastic_Y,
        "Stochastic_MiniGrid_8x8_Empty": Stochastic_MiniGrid_8x8_Empty,
        "Stochastic_MiniGrid_6x6_Empty_Random": Stochastic_MiniGrid_6x6_Empty_Random,
        "Stochastic_FrozenLake_4x4_Random_Map": Stochastic_FrozenLake_4x4_Random_Map,
        "Stochastic_FrozenLake_8x8_Random_Map": Stochastic_FrozenLake_8x8_Random_Map,
        "Taxi_Is_Raining_Fickle_Passenger": Taxi_Is_Raining_Fickle_Passenger
    }
    return mapping[game_name]

def get_model_config(game_name, algorithm="StochasticMuZero"):
    config = {}
    
    if "TicTacToe" in game_name:
        config.update({"num_resBlocks": 5, "num_hidden": 64})
    elif "ConnectFour" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "Breakthrough" in game_name:
        config.update({"num_resBlocks": 12, "num_hidden": 128})
    elif "Havannah" in game_name or "Y" in game_name:
        config.update({"num_resBlocks": 20, "num_hidden": 256})
    elif "Stochastic_MiniGrid" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "FrozenLake_4x4" in game_name:
        config.update({"num_resBlocks": 5, "num_hidden": 64})
    elif "FrozenLake_8x8" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "Taxi" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    
    if algorithm == "StochasticMuZero":
        if "Taxi" in game_name:
             min_val, max_val = -220, 30
        else:
             min_val, max_val = -2, 13 
        
        step = 1
        support_size = (max_val - min_val) // step 
        
        config.update({
            "chance_space_size": 12,
            "support_size": support_size, 
            "support_range": (min_val, max_val, step),
            "use_afterstate": True
        })
            
    return config

@ray.remote
class EvaluationWorker:
    def __init__(self, game_cls, model_cls, model_args, games_per_worker):
        self.game = game_cls()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Eval Worker initialized on {self.device}")

        worker_model_args = model_args.copy()
        worker_model_args["device"] = self.device
        
        self.model = model_cls(**worker_model_args)
        self.model.to(self.device)
        self.model.eval()

        self.games_per_worker = games_per_worker
        self._model_build_done = False
        
    def evaluate_batch(self, weights, temperature, mcts_config):
        # Lazy model initialization / Forward pass to build graph if needed
        if not self._model_build_done:
            obs_shape = self.model.observation_shape
            dummy_obs = torch.zeros((1, *obs_shape), device=self.device)
            hidden = self.model.representation(dummy_obs)
            dummy_action = torch.zeros((1,), dtype=torch.long, device=self.device)
            next_hidden, _ = self.model.dynamics(hidden, dummy_action)
            self.model.prediction(hidden)
            
            if self.model.use_afterstate:
                self.model.afterstate_prediction(next_hidden)
                chance_onehot = torch.zeros((1, self.model.chance_space_size), device=self.device)
                self.model.afterstate_dynamics(hidden, chance_onehot)
                
            self._model_build_done = True

        # Load weights
        self.model.load_state_dict(weights)
        self.model.eval()

        # Config extraction
        support_range = mcts_config.get('support_range', (-300, 301, 1))
        mcts_type = mcts_config.get('mcts_type', 'classic')
        
        MCTSClass = PowerMeanMCTS if mcts_type == 'powermean' else ClassicMCTS
        
        mcts_kwargs = {
            "game": self.game,
            "model": self.model,
            "num_searches": mcts_config.get('num_searches', 50),
            "dirichlet_epsilon": mcts_config.get('dirichlet_epsilon', 0.0), 
            "dirichlet_alpha": mcts_config.get('dirichlet_alpha', 0.3),
            "discount": mcts_config.get('discount', 0.997),
            "use_chance_nodes": mcts_config.get('use_chance_nodes', False),
            "support_range": support_range
        }
        
        if mcts_type == 'powermean':
            mcts_kwargs["C"] = mcts_config.get('c_puct', 1.41)
            mcts_kwargs["p"] = mcts_config.get('p', 1.5)
        else:
            mcts_kwargs["c_puct"] = mcts_config.get('c_puct', 1.41)

        mcts = MCTSClass(**mcts_kwargs)

        
        results = []
        
        active_games = []
        for _ in range(self.games_per_worker):
            spg = StochasticMuZeroSPG(self.game)
            active_games.append(spg)
            
        while active_games:
            current_states = [spg.state for spg in active_games]
            mcts.search(current_states, active_games)
            
            next_active_games = []
            
            for spg in active_games:
                root = spg.root
                action_probs = mcts.get_action_probs(root, temperature)
                
                if temperature < 1e-3:
                    action = np.argmax(action_probs)
                else:
                    action = np.random.choice(len(action_probs), p=action_probs)
                
                spg.state = self.game.get_next_state(spg.state, action)
                
                reward = 0.0
                if hasattr(spg.state, 'reward'): 
                    reward = spg.state.reward
                elif hasattr(spg.state, 'custom_reward'): 
                     reward = spg.state.custom_reward

                spg.total_reward += reward
                
                _, is_terminal = self.game.get_value_and_terminated(spg.state, 0)
                
                if is_terminal and reward == 0.0:
                     val, _ = self.game.get_value_and_terminated(spg.state, 0)
                     if not hasattr(spg.state, 'custom_reward'):
                         spg.total_reward += val
                
                if is_terminal:
                    results.append(spg.total_reward)
                else:
                    next_active_games.append(spg)
                    
            active_games = next_active_games
        
        return results

def run_evaluation(args):
    if not ray.is_initialized():
        project_root = os.getcwd() 
        module_paths = [
            project_root,
            os.path.join(project_root, 'games'),
            os.path.join(project_root, 'alphazero'),
            os.path.join(project_root, 'mcts'),
            os.path.join(project_root, 'stochastic_muzero'),
        ]
        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": module_paths})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Device: {device}")

    GameClass = get_game_class(args.game)
    game = GameClass()
    
    model_config = get_model_config(args.game, "StochasticMuZero")
    
    if hasattr(game, 'tensor_shape'):
        obs_shape = game.tensor_shape
    else:
        dummy_state = game.get_initial_state()
        dummy_obs = game.get_encoded_state(dummy_state)
        obs_shape = dummy_obs.shape

    model_args = {
        "observation_shape": obs_shape,
        "action_size": game.action_size,
        "hidden_channels": model_config["num_hidden"],
        "num_resblocks": model_config["num_resBlocks"],
        "chance_space_size": model_config.get("chance_space_size", 32),
        "use_afterstate": model_config.get("use_afterstate", False),
        "support_size": model_config.get("support_size", 601),
        "support_range": model_config.get("support_range", (-300, 301, 1)),
        "device": device 
    }
    
    ModelClass = StochasticMuZeroNetwork

    # Define the 8 algorithms to run
    ALGO_CONFIGS = [
        {"name": "Classic",         "mcts_type": "classic",   "p": None},
        {"name": "PowerMean_p1.0", "mcts_type": "powermean", "p": 1.0},
        {"name": "PowerMean_p1.2", "mcts_type": "powermean", "p": 1.2},
        {"name": "PowerMean_p1.5", "mcts_type": "powermean", "p": 1.5},
        {"name": "PowerMean_p2.0", "mcts_type": "powermean", "p": 2.0},
        {"name": "PowerMean_p2.5", "mcts_type": "powermean", "p": 2.5},
        {"name": "PowerMean_p3.0", "mcts_type": "powermean", "p": 3.0},
        {"name": "PowerMean_p4.0", "mcts_type": "powermean", "p": 4.0},
    ]

    num_gpus = torch.cuda.device_count()
    gpu_per_worker = (num_gpus / args.num_games_parallel) if num_gpus > 0 else 0
    WorkerRemote = EvaluationWorker.options(num_gpus=gpu_per_worker)
    
    print(f"Initializing {args.num_games_parallel} workers...")
    workers = [
        WorkerRemote.remote(
            game_cls=GameClass, 
            model_cls=ModelClass,
            model_args=model_args,
            games_per_worker=args.games_per_worker
        ) for _ in range(args.num_games_parallel)
    ]
    
    results_summary = {}
    checkpoints = args.checkpoint_paths if args.checkpoint_paths else ["random"]
    
    total_steps = len(checkpoints) * len(ALGO_CONFIGS)
    current_step = 0

    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        results_summary[checkpoint_name] = {}
        
        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*80}")
        
        if checkpoint_path == "random":
            print(f"Initializing random model...")
            local_args = model_args.copy()
            local_args["device"] = "cpu"
            local_model = ModelClass(**local_args)
            
            # Lazy init 
            obs_shape = local_model.observation_shape
            dummy_obs = torch.zeros((1, *obs_shape), device="cpu")
            hidden = local_model.representation(dummy_obs)
            dummy_action = torch.zeros((1,), dtype=torch.long, device="cpu")
            next_hidden, _ = local_model.dynamics(hidden, dummy_action)
            local_model.prediction(hidden)
            if local_model.use_afterstate:
                local_model.afterstate_prediction(next_hidden)
                chance_onehot = torch.zeros((1, local_model.chance_space_size), device="cpu")
                local_model.afterstate_dynamics(hidden, chance_onehot)
            
            weights = local_model.state_dict()
        else:
            print(f"Loading weights from disk...")
            weights = torch.load(checkpoint_path, map_location='cpu')
            
        weights_ref = ray.put(weights)
        
        for algo_cfg in ALGO_CONFIGS:
            algo_name = algo_cfg["name"]
            current_step += 1
            print(f"\n[{current_step}/{total_steps}] Running {algo_name} ...")
            
            current_mcts_config = {
                "mcts_type": algo_cfg["mcts_type"],
                "num_searches": args.num_searches,
                "c_puct": args.C,
                "p": algo_cfg["p"], 
                "dirichlet_epsilon": args.dirichlet_epsilon,
                "dirichlet_alpha": args.dirichlet_alpha,
                "discount": args.gamma, 
                "use_chance_nodes": True,
                "support_range": model_config.get("support_range", (-300, 301, 1))
            }

            all_scores = []
            total_games = args.num_runs * args.num_games_per_pair   
            
            games_launched = 0
            games_completed = 0
            futures = []
            
            pbar = tqdm.tqdm(total=total_games, desc=f"{algo_name}")
            
            while games_completed < total_games:
                # Launch tasks
                while games_launched < total_games and len(futures) < args.num_games_parallel * 2:
                    batch_index = games_launched // args.games_per_worker
                    worker = workers[batch_index % args.num_games_parallel]
                    
                    fut = worker.evaluate_batch.remote(weights_ref, args.temperature, current_mcts_config)
                    futures.append(fut)
                    games_launched += args.games_per_worker

                # Collect results
                done_ids, futures = ray.wait(futures, num_returns=1)
                
                for res in ray.get(done_ids):
                    all_scores.extend(res)
                    batch_len = len(res)
                    pbar.update(min(batch_len, total_games - games_completed))
                    games_completed += batch_len
                    
            pbar.close()
            
            scores = np.array(all_scores[:total_games]) 
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            print(f" -> {algo_name} Result: Mean={round(mean_score/10, 2)}, Std={round(std_score/10, 2)}")
            
            results_summary[checkpoint_name][algo_name] = {
                "mean": float(mean_score),
                "std": float(std_score),
                "min": float(min_score),
                "max": float(max_score),
                "raw": scores.tolist()
            }
        
    os.makedirs("evaluate_result", exist_ok=True)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    fn = f"evaluate_result/eval_8algos_{suffix}.json"
    
    with open(fn, "w") as f:
        json.dump({
            "config_args": vars(args), 
            "algo_configs_tested": ALGO_CONFIGS,
            "results": results_summary
        }, f, indent=4)
        
    print(f"\nSaved FULL evaluation results to {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Stochastic_MiniGrid_8x8_Empty",
                        choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                 "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                 "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y",
                                 "Stochastic_MiniGrid_8x8_Empty", "Stochastic_MiniGrid_6x6_Empty_Random",
                                 "Stochastic_FrozenLake_4x4_Random_Map", "Stochastic_FrozenLake_8x8_Random_Map",
                                 "Taxi_Is_Raining_Fickle_Passenger"],
                        help="Game to play.")  
    
    parser.add_argument("--checkpoint_paths", nargs='+', default=[], help="List of model checkpoints to evaluate.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated runs (batches).")
    parser.add_argument("--num_searches", type=int, default=50, help="Number of MCTS simulations per move.")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma factor for PowerMean.")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.01, help="Noise (usually 0 for eval).") 
    parser.add_argument("--dirichlet_alpha", type=float, default=0.01)
    parser.add_argument("--num_games_per_pair", type=int, default=500, help="Total games to play per algorithm.")
    
    parser.add_argument("--games_per_worker", type=int, default=15, 
                        help="Batch size per worker.")
    
    parser.add_argument("--num_games_parallel", type=int, default=10, help="Number of Ray workers.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for move selection.") 
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    run_evaluation(args)