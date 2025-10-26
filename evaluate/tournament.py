import torch
import numpy as np
import argparse, sys, itertools, time, json, os

sys.path.append('/content/powermean-mcts-alphazero/')
sys.path.append('/content/powermean-mcts-alphazero/games')
sys.path.append('/content/powermean-mcts-alphazero/alphazero')
sys.path.append('/content/powermean-mcts-alphazero/mcts')

from games import ConnectFour, Breakthrough
from alphazero import ResNet
from mcts import Stochastic_Powermean_UCT, PUCT

class SPG:
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.memory = []

def create_game(game):
    if game == "ConnectFour":
        return ConnectFour()
    elif game == "Breakthrough":
        return Breakthrough()

def create_model(game, device, args):
    if (game.name == "ConnectFour"):
        model = ResNet(
            game=game, 
            num_resBlocks=9, 
            num_hidden=128, 
            device=device
        )
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        return model   
    elif (game.name == "Breakthrough"):
        model = ResNet(
            game=game, 
            num_resBlocks=12, 
            num_hidden=128, 
            device=device
        )
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
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
            print(f"Remaining games: {len(spGames)}")
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

    print("GAME ENDED")
    print(f"Result: Player 1 wins: {result['first_win']}, Player 2 wins: {result['second_win']}, Draws: {result['draw']}")
    print("=" * 50)
    return result

def run_tournament(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = create_game(
        game=args.game
    )

    MODEL_CHECKPOINT = args.checkpoint_path

    print("=" * 70)
    print("TOURNAMENT CONFIGURATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Game: {game.name}")
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")

    model = create_model(
        game=game, 
        device=device, 
        args=args
    )

    mcts_list = []
    
    # add powermean
    for p in args.p:
        mcts_list.append(
        {   
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
        }
    )

    # add puct
    mcts_list.append(
        {
            "name": "PUCT",
            "alphazero": PUCT(
                game=game, 
                model=model, 
                C=args.C, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            )
        }        
    )    

    results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    num_games_per_pair = args.num_games_per_pair

    print("=" * 70)
    print("TOURNAMENT START")
    print("=" * 70)

    for m1, m2 in itertools.combinations(mcts_list, 2):
        print(f"\n{'=' * 70}")
        print(f"MATCHUP: {m1['name']} vs {m2['name']}")
        print(f"{'=' * 70}")

        for batch_idx in range(num_games_per_pair // args.num_games_parallel):
            print(f"\n--- Round {batch_idx + 1}/{num_games_per_pair // args.num_games_parallel} ---")
            print(f"Game 1: {m1['name']} (Player 1) vs {m2['name']} (Player 2)")
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

            print(f"\nGame 2: {m2['name']} (Player 1) vs {m1['name']} (Player 2)")
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

        print(f"\nFinished matchup: {m1['name']} vs {m2['name']}")
        print(f"Current standings:")
        print(f"  {m1['name']}: {results[m1['name']]}")
        print(f"  {m2['name']}: {results[m2['name']]}")

    print("\n" + "=" * 70)
    print("FINAL TOURNAMENT RESULTS")
    print("=" * 70)
    result = []
    for name, record in results.items():
        total_games = record["win"] + record["loss"] + record["draw"]
        win_rate = (record["win"] / total_games * 100) if total_games > 0 else 0
        print(f"{name}:")
        print(f"  Wins: {record['win']}")
        print(f"  Losses: {record['loss']}")
        print(f"  Draws: {record['draw']}")
        print(f"  Win Rate: {win_rate:.2f}%")
        result.append(
            {
                "name": name,
                "Wins": record['win'],
                "Losses": record['loss'],
                "Draws": record['draw'],
            }
        )
        print("-" * 50)
    

    print("=" * 70)
    print("TOURNAMENT COMPLETED")
    print("=" * 70)

    os.makedirs("evaluate_result", exist_ok=True)
    json_file_path = f"evaluate_result/{game.name}_model_{args.checkpoint_path.split('/')[-1]}.json"
    with open(json_file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament.")
    parser.add_argument("--game", type=str, default="ConnectFour",
                        choices=["ConnectFour", "Breakthrough"],
                        help="game to player (default: ConnectFour).")    
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the model checkpoint.")
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