import copy 
import ray
import tqdm
import torch
import numpy as np
import argparse, sys, itertools, time, json, os, random, string
from collections import defaultdict
from functools import partial

sys.path.append('powermean-mcts-alphazero/')
sys.path.append('powermean-mcts-alphazero/games')
sys.path.append('powermean-mcts-alphazero/alphazero')
sys.path.append('powermean-mcts-alphazero/mcts')

from games import ConnectFour, Breakthrough, TicTacToe, Havannah, Y, Stochastic_ConnectFour, Stochastic_Breakthrough, Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y
from alphazero import ResNet
from mcts import Stochastic_Powermean_UCT, PUCT


class SPG:
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.memory = []

def random_suffix(k=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def get_game_class(game_name):
    mapping = {
        "ConnectFour": ConnectFour, "Breakthrough": Breakthrough, "TicTacToe": TicTacToe,
        "Havannah": Havannah, "Y": Y, "Stochastic_ConnectFour": Stochastic_ConnectFour, 
        "Stochastic_Breakthrough": Stochastic_Breakthrough, "Stochastic_TicTacToe": Stochastic_TicTacToe, 
        "Stochastic_Havannah": Stochastic_Havannah, "Stochastic_Y": Stochastic_Y
    }
    return mapping[game_name]

def get_model_config_and_class(game_name):
    if "ConnectFour" in game_name:
        config = {"num_resBlocks": 9, "num_hidden": 128}
    elif "Breakthrough" in game_name:
        config = {"num_resBlocks": 12, "num_hidden": 128}
    else:
        config = {"num_resBlocks": 5, "num_hidden": 64} 
    return ResNet, config


@ray.remote
class TournamentWorker:
    def __init__(self, game_cls, model_cls, model_args):
        self.game = game_cls()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_cls(game=self.game, device=self.device, **model_args)
        self.model.eval()
        
        print(f"Tournament Worker initialized on {self.device}")

    def update_weights(self, weights):
        self.model.load_state_dict(weights)

    def run_match(self, temperature, starting_player, 
                  mcts_cls_A, mcts_args_A, mcts_cls_B, mcts_args_B):
        
        mcts_A = mcts_cls_A(game=self.game, model=self.model, **mcts_args_A)
        mcts_B = mcts_cls_B(game=self.game, model=self.model, **mcts_args_B)

        state = self.game.get_initial_state()
        player = starting_player
        
        bots = {0: mcts_A, 1: mcts_B}
        
        spg_wrapper = SPG(self.game)

        while True:
            current_bot = bots[player]
            
            spg_wrapper.state = state 
            
            current_bot.search([state], [spg_wrapper]) 
            
            action_probs = np.zeros(self.game.action_size)
            if spg_wrapper.root is not None:
                for child in spg_wrapper.root.children:
                    action_probs[child.action_taken] = child.visit_count
            
            if np.sum(action_probs) == 0:
                 action_probs = np.ones_like(action_probs) / len(action_probs)
            else:
                action_probs /= np.sum(action_probs)

            temp_action_probs = action_probs ** (1 / temperature)
            temp_action_probs /= np.sum(temp_action_probs)
            
            action = np.random.choice(self.game.action_size, p=temp_action_probs)
            state = self.game.get_next_state(state, action)
            
            value, is_terminal = self.game.get_value_and_terminated(state, player)
            
            if is_terminal:
                if value == 1.0:
                    return {'winner': player, 'outcome': 1.0} 
                elif value == 0.5:
                    return {'winner': None, 'outcome': 0.5} 
                elif value == 0.0:
                    return {'winner': self.game.get_opponent(player), 'outcome': 1.0} 

            player = self.game.get_opponent(player)

def run_single_tournament_distributed(game_cls, mcts_list, args, run_id, workers, device):
    checkpoint_path = mcts_list[0]["checkpoint_path"]
    
    print(f"  Loading weights from: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location='cpu')
    weights_ref = ray.put(weights) 
    
    print(f"  Broadcasting weights to {len(workers)} workers...")
    ray.get([w.update_weights.remote(weights_ref) for w in workers])
    print("  Weights synchronized.")

    futures = []
    run_results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    matchups = list(itertools.combinations(mcts_list, 2))
    
    total_games = len(matchups) * args.num_games_per_pair * 2
    print(f"  Scheduling {total_games} games across {len(matchups)} matchups...")

    worker_index = 0

    for m1, m2 in matchups:
        cls_A, args_A = m1["mcts_cls"], m1["mcts_args"]
        cls_B, args_B = m2["mcts_cls"], m2["mcts_args"]
        
        for start_p in [0, 1]: 
            for _ in range(args.num_games_per_pair):
                worker = workers[worker_index % len(workers)]
                worker_index += 1
                
                if start_p == 0:
                    fut = worker.run_match.remote(
                        args.temperature, 0, 
                        cls_A, args_A, cls_B, args_B
                    )
                else:
                    fut = worker.run_match.remote(
                        args.temperature, 0, 
                        cls_B, args_B, cls_A, args_A
                    )

                p1_name = m1["name"] if start_p == 0 else m2["name"]
                p2_name = m2["name"] if start_p == 0 else m1["name"]
                
                futures.append((fut, p1_name, p2_name))

    with tqdm.tqdm(total=total_games, desc=f"Tournament Run #{run_id + 1}") as pbar:
        while futures:
            done_ids, _ = ray.wait([f[0] for f in futures], num_returns=1)
            done_id = done_ids[0]
            
            idx = -1
            for i, f in enumerate(futures):
                if f[0] == done_id:
                    idx = i
                    break
            
            fut, p1_name, p2_name = futures.pop(idx)
            
            try:
                result = ray.get(done_id)
                pbar.update(1)

                if result['outcome'] == 1.0:
                    if result['winner'] == 0: 
                        run_results[p1_name]["win"] += 1
                        run_results[p2_name]["loss"] += 1
                    else: 
                        run_results[p2_name]["win"] += 1
                        run_results[p1_name]["loss"] += 1
                elif result['outcome'] == 0.5:
                    run_results[p1_name]["draw"] += 1
                    run_results[p2_name]["draw"] += 1
            except Exception as e:
                print(f"Error in game: {e}")

    return run_results

def aggregate_results(all_runs_results):
    """Aggregate results from multiple runs"""
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
    if not ray.is_initialized():
        project_root = os.path.abspath('powermean-mcts-alphazero') 
        
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
    
    for checkpoint_idx, checkpoint_path in enumerate(args.checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"PROCESSING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")
        
        mcts_list = []
        
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
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)

        with open(fn, "w") as f:
            json.dump(save_data, f, indent=4, default=convert)
        print(f"  Saved: {fn}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="ConnectFour",
                            choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                    "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                    "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y"],
                            help="Game to play (default: ConnectFour).")  
    parser.add_argument("--checkpoint_paths", nargs='+', required=True)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--num_searches", type=int, default=50)
    parser.add_argument("--C", type=float, default=1.41)
    parser.add_argument("--p", type=float, nargs='+', default=[1.5])
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.0) 
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--num_games_per_pair", type=int, default=10)
    parser.add_argument("--num_games_parallel", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.01) 
    
    args = parser.parse_args()
    run_tournament(args)