import torch
import numpy as np
import argparse, sys, itertools, time, json, os, random, string
from collections import defaultdict

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

def create_game(game):
    if game == "ConnectFour":
        return ConnectFour()
    elif game == "Breakthrough":
        return Breakthrough()
    elif game == "TicTacToe":
        return TicTacToe()
    elif game == "Havannah":
        return Havannah()
    elif game == "Y":
        return Y()
    elif game == "Stochastic_ConnectFour":
        return Stochastic_ConnectFour()
    elif game == "Stochastic_Breakthrough":
        return Stochastic_Breakthrough()
    elif game == "Stochastic_TicTacToe":
        return Stochastic_TicTacToe()
    elif game == "Stochastic_Havannah":
        return Stochastic_Havannah()
    elif game == "Stochastic_Y":
        return Stochastic_Y()

def create_model(game, device, checkpoint_path):
    if (game.name == "ConnectFour") or (game.name == "Stochastic_ConnectFour"):
        model = ResNet(
            game=game, 
            num_resBlocks=9, 
            num_hidden=128, 
            device=device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model   
    elif (game.name == "Breakthrough") or (game.name == "Stochastic_Breakthrough"):
        model = ResNet(
            game=game, 
            num_resBlocks=12, 
            num_hidden=128, 
            device=device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model 
    elif (game.name == "TicTacToe") or (game.name == "Stochastic_TicTacToe"):
        model = ResNet(
            game=game, 
            num_resBlocks=5, 
            num_hidden=64, 
            device=device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model   
    elif (game.name == "Havannah") or (game.name == "Stochastic_Havannah"):
        model = ResNet(
            game=game, 
            num_resBlocks=20, 
            num_hidden=256, 
            device=device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model  
    elif (game.name == "Y") or (game.name == "Stochastic_Y"):
        model = ResNet(
            game=game, 
            num_resBlocks=20, 
            num_hidden=256, 
            device=device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model                    

def play_interactive(game, player1, player2, num_games_parallel, temperature): 
    result = {
        "first_win": 0,
        "second_win": 0,
        "draw": 0
    }    
    
    player = game.get_current_player(
        state=game.get_initial_state()
    )
    spGames = [SPG(game) for _ in range(num_games_parallel)]

    last_print_time = time.time()

    while len(spGames) > 0:
        if time.time() - last_print_time >= 30:
            print(f"    [Progress] Remaining games: {len(spGames)}")
            last_print_time = time.time()

        if player == 0:
            states = [spg.state for spg in spGames]
            player1["alphazero"].search(
                states=states, 
                spGames=spGames
            )
        else:
            states = [spg.state for spg in spGames]
            player2["alphazero"].search(
                states=states, 
                spGames=spGames
            )

        for i in range(len(spGames))[::-1]:
            spg = spGames[i]
            action_probs = np.zeros(game.action_size)
            
            for child in spg.root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            temperature_action_probs = action_probs ** (1 / temperature)
            if np.sum(temperature_action_probs) == 0:
                temperature_action_probs = np.ones_like(temperature_action_probs) / len(temperature_action_probs)
            else:
                temperature_action_probs /= np.sum(temperature_action_probs)
            
            action = np.random.choice(game.action_size, p=temperature_action_probs)      
            spg.state = game.get_next_state(spg.state, action)

            value, is_terminal = game.get_value_and_terminated(spg.state, player)
            if is_terminal:
                if player == 0:
                    if value == 1.0:
                        result["first_win"] += 1
                    elif value == 0.0:
                        result["second_win"] += 1
                    elif value == 0.5:
                        result["draw"] += 1
                elif player == 1:
                    if value == 1.0:
                        result["second_win"] += 1
                    elif value == 0.0:
                        result["first_win"] += 1
                    elif value == 0.5:
                        result["draw"] += 1
                del spGames[i]

        player = game.get_opponent(player)  

    return result

def run_single_tournament(game, mcts_list, args, run_id):
    """Run a single tournament iteration"""
    print(f"\n{'='*70}")
    print(f"TOURNAMENT RUN #{run_id + 1}")
    print(f"{'='*70}")
    
    results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    num_games_per_pair = args.num_games_per_pair

    for m1, m2 in itertools.combinations(mcts_list, 2):
        print(f"\n  Matchup: {m1['name']} vs {m2['name']}")

        for batch_idx in range(num_games_per_pair // args.num_games_parallel):
            # Game 1: m1 as Player 1
            result = play_interactive(
                game=game, 
                player1=m1, 
                player2=m2, 
                num_games_parallel=args.num_games_parallel, 
                temperature=args.temperature
            )
            
            results[m1["name"]]["win"] += result["first_win"]
            results[m1["name"]]["loss"] += result["second_win"]
            results[m1["name"]]["draw"] += result["draw"]

            results[m2["name"]]["win"] += result["second_win"]
            results[m2["name"]]["loss"] += result["first_win"]
            results[m2["name"]]["draw"] += result["draw"]

            # Game 2: m2 as Player 1
            result = play_interactive(
                game=game, 
                player1=m2, 
                player2=m1, 
                num_games_parallel=args.num_games_parallel, 
                temperature=args.temperature
            )

            results[m2["name"]]["win"] += result["first_win"]
            results[m2["name"]]["loss"] += result["second_win"]
            results[m2["name"]]["draw"] += result["draw"]

            results[m1["name"]]["win"] += result["second_win"]
            results[m1["name"]]["loss"] += result["first_win"]
            results[m1["name"]]["draw"] += result["draw"]

    return results

def aggregate_results(all_runs_results):
    """Aggregate results from multiple runs"""
    aggregated = defaultdict(lambda: {"win": [], "loss": [], "draw": []})
    
    for run_results in all_runs_results:
        for name, record in run_results.items():
            aggregated[name]["win"].append(record["win"])
            aggregated[name]["loss"].append(record["loss"])
            aggregated[name]["draw"].append(record["draw"])
    
    # Calculate statistics
    final_results = {}
    for name, records in aggregated.items():
        wins = np.array(records["win"])
        losses = np.array(records["loss"])
        draws = np.array(records["draw"])
        total_games = wins + losses + draws
        win_rates = (wins / total_games * 100)
        
        final_results[name] = {
            "win_mean": float(np.mean(wins)),
            "win_std": float(np.std(wins)),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "draw_mean": float(np.mean(draws)),
            "draw_std": float(np.std(draws)),
            "win_rate_mean": float(np.mean(win_rates)),
            "win_rate_std": float(np.std(win_rates)),
            "all_runs": {
                "wins": wins.tolist(),
                "losses": losses.tolist(),
                "draws": draws.tolist(),
                "win_rates": win_rates.tolist()
            }
        }
    
    return final_results

def run_tournament(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = create_game(game=args.game)

    print("=" * 70)
    print("MULTI-TOURNAMENT CONFIGURATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Game: {game.name}")
    print(f"Number of checkpoints: {len(args.checkpoint_paths)}")
    print(f"Number of runs per checkpoint: {args.num_runs}")
    print(f"Checkpoints:")
    for i, cp in enumerate(args.checkpoint_paths, 1):
        print(f"  {i}. {cp}")
    print("=" * 70)

    # Process each checkpoint
    all_checkpoint_results = {}
    
    for checkpoint_idx, checkpoint_path in enumerate(args.checkpoint_paths):
        print(f"\n{'#'*70}")
        print(f"PROCESSING CHECKPOINT {checkpoint_idx + 1}/{len(args.checkpoint_paths)}")
        print(f"Path: {checkpoint_path}")
        print(f"{'#'*70}")
        
        # Load model for this checkpoint
        model = create_model(game=game, device=device, checkpoint_path=checkpoint_path)
        
        # Create MCTS list for this checkpoint
        mcts_list = []
        
        # Add powermean variants
        for p in args.p:
            mcts_list.append({   
                "name": f"Stochastic_Powermean_UCT_p={p}", 
                "alphazero": Stochastic_Powermean_UCT(
                    game=game, 
                    model=model, 
                    C=args.C, 
                    p=p, 
                    gamma=args.gamma, 
                    dirichlet_epsilon=args.dirichlet_epsilon, 
                    dirichlet_alpha=args.dirichlet_alpha, 
                    num_searches=args.num_searches                
                )
            })

        # Add PUCT
        mcts_list.append({
            "name": "PUCT",
            "alphazero": PUCT(
                game=game, 
                model=model, 
                C=args.C, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            )
        })
        
        # Run multiple tournaments for this checkpoint
        all_runs_results = []
        for run_id in range(args.num_runs):
            run_results = run_single_tournament(game, mcts_list, args, run_id)
            all_runs_results.append(run_results)
            
            # Print current run summary
            print(f"\n  Run #{run_id + 1} Summary:")
            for name, record in run_results.items():
                total = record["win"] + record["loss"] + record["draw"]
                wr = (record["win"] / total * 100) if total > 0 else 0
                print(f"    {name}: W={record['win']}, L={record['loss']}, D={record['draw']}, WR={wr:.2f}%")
        
        # Aggregate results for this checkpoint
        checkpoint_results = aggregate_results(all_runs_results)
        checkpoint_name = os.path.basename(checkpoint_path)
        all_checkpoint_results[checkpoint_name] = checkpoint_results
        
        # Print checkpoint summary
        print(f"\n{'='*70}")
        print(f"CHECKPOINT SUMMARY: {checkpoint_name}")
        print(f"{'='*70}")
        for name, stats in checkpoint_results.items():
            print(f"{name}:")
            print(f"  Win Rate: {stats['win_rate_mean']:.2f}% ± {stats['win_rate_std']:.2f}%")
            print(f"  Wins: {stats['win_mean']:.1f} ± {stats['win_std']:.1f}")
            print(f"  Losses: {stats['loss_mean']:.1f} ± {stats['loss_std']:.1f}")
            print(f"  Draws: {stats['draw_mean']:.1f} ± {stats['draw_std']:.1f}")
            print("-" * 50)
        
        # Save individual checkpoint results
        os.makedirs("individual_result", exist_ok=True)
        checkpoint_basename = os.path.splitext(checkpoint_name)[0]  # Remove .pt extension

        suffix = random_suffix()
        individual_json_path = f"individual_result/{checkpoint_basename}_search{args.num_searches}_runs{args.num_runs}_{suffix}.json"        

        individual_save_data = {
            "config": {
                "game": args.game,
                "checkpoint": checkpoint_path,
                "num_runs": args.num_runs,
                "num_searches": args.num_searches,
                "C": args.C,
                "p_values": args.p,
                "gamma": args.gamma,
                "num_games_per_pair": args.num_games_per_pair,
                "num_games_parallel": args.num_games_parallel,
                "temperature": args.temperature
            },
            "results": checkpoint_results
        }
        
        with open(individual_json_path, "w") as f:
            json.dump(individual_save_data, f, indent=4)
        
        print(f"Checkpoint results saved to: {individual_json_path}")

    # Final summary across all checkpoints
    print(f"\n{'#'*70}")
    print("FINAL SUMMARY - ALL CHECKPOINTS")
    print(f"{'#'*70}")
    
    for checkpoint_name, results in all_checkpoint_results.items():
        print(f"\n{checkpoint_name}:")
        for name, stats in results.items():
            print(f"  {name}: WR={stats['win_rate_mean']:.2f}%±{stats['win_rate_std']:.2f}%")

    # Save results
    os.makedirs("evaluate_result", exist_ok=True)

    suffix = random_suffix()
    json_file_path = f"evaluate_result/summary_search{args.num_searches}_runs{args.num_runs}_{suffix}.json"    

    save_data = {
        "config": {
            "game": args.game,
            "num_runs": args.num_runs,
            "num_searches": args.num_searches,
            "C": args.C,
            "p_values": args.p,
            "gamma": args.gamma,
            "num_games_per_pair": args.num_games_per_pair,
            "checkpoints": args.checkpoint_paths
        },
        "results": all_checkpoint_results
    }
    
    with open(json_file_path, "w") as f:
        json.dump(save_data, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {json_file_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-checkpoint Tournament with Multiple Runs.")
    parser.add_argument("--game", type=str, default="ConnectFour",
                        choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                 "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                 "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y"],
                        help="Game to play (default: ConnectFour).")    
    parser.add_argument("--checkpoint_paths", type=str, nargs='+', required=True, 
                        help="List of paths to model checkpoints.")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of times to run tournament for each checkpoint (default: 3).")
    parser.add_argument("--num_searches", type=int, default=600, 
                        help="Number of MCTS searches per bot move (default: 600).")
    parser.add_argument("--C", type=float, default=1.41, 
                        help="Exploration constant C for MCTS (default: 1.41).")
    parser.add_argument("--p", type=float, nargs='+', default=[1.5], 
                        help="List of power parameter p for power mean algorithms (default: [1.5]).")
    parser.add_argument("--gamma", type=float, default=0.95, 
                        help="Discount factor gamma for MCTS (default: 0.95).")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.0, 
                        help="Dirichlet noise epsilon for MCTS (default: 0.0).")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.0, 
                        help="Dirichlet noise alpha for MCTS (default: 0.0).")
    
    parser.add_argument("--num_games_per_pair", type=int, default=500, 
                        help="Number of games per MCTS pair in the tournament.")
    parser.add_argument("--num_games_parallel", type=int, default=10, 
                        help="Number of parallel games to run.")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature parameter for MCTS.")
    args = parser.parse_args()
    run_tournament(args)