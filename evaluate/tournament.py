import torch, itertools, argparse, logging, os, sys
import numpy as np

# sys.path.append('/content/powermean_mcts_alphazero/')
# sys.path.append('/content/powermean_mcts_alphazero/games')
# sys.path.append('/content/powermean_mcts_alphazero/alphazero')
# sys.path.append('/content/powermean_mcts_alphazero/mcts')

from games import ConnectFour
from alphazero import ResNet, SPG
from mcts import MCTS_Global_Parallel, MCTS_Local_Parallel, \
    Stochastic_Powermean_UCT_New, Stochastic_Powermean_UCT, PUCT 

def setup_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    logger = logging.getLogger("tournament")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(path, encoding='utf-8')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

@torch.no_grad()
def play_game(game, first, second, num_games_parallel, temperature, logger=None):
    if logger:
        logger.info("=" * 50)
        logger.info("NEW GAME STARTED")
        logger.info(f"Player 1: {first['name']}")
        logger.info(f"Player 2: {second['name']}")
    
    result = {
        "first_win": 0,
        "second_win": 0,
        "draw": 0
    }

    player = 1
    spGames = [SPG(game) for _ in range(num_games_parallel)]
    while len(spGames) > 0:
        states = np.stack([spg.state for spg in spGames])
        neutral_states = game.change_perspective(states, player)

        if player == 1:
            first["mcts"].search(neutral_states, spGames)
        else:
            second["mcts"].search(neutral_states, spGames)
        
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
                if (player == 1):
                    if value == 1:
                        result["first_win"] += 1
                    elif value == -1:
                        result["second_win"] += 1
                    else:
                        result["draw"] += 1
                else:
                    if value == 1:
                        result["second_win"] += 1
                    elif value == -1:
                        result["first_win"] += 1
                    else:
                        result["draw"] += 1
                del spGames[i]
        player = game.get_opponent(player)  

    if logger:
        logger.info("GAME ENDED")
        logger.info(f"Result: Player 1 wins: {result['first_win']}, Player 2 wins: {result['second_win']}, Draws: {result['draw']}")
        logger.info("=" * 50)
    return result              

def run_tournament(args):
    logger = setup_logger(path=args.log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = ConnectFour()

    MODEL_CHECKPOINT = args.model_checkpoint

    logger.info("=" * 70)
    logger.info("TOURNAMENT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Game: {game.name}")
    logger.info(f"Model checkpoint: {MODEL_CHECKPOINT}")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    mcts_list = []
    if args.mcts_global:
        mcts_global = {
            "name": "MCTS_Global",
            "mcts": MCTS_Global_Parallel(
                game=game, 
                model=model, 
                C=args.C, 
                p=args.p, 
                gamma=args.gamma, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            ),
            "model": model,
        }
        mcts_list.append(mcts_global)
    
    if args.mcts_local:
        mcts_local = {
            "name": "MCTS_Local",
            "mcts": MCTS_Local_Parallel(
                game=game, 
                model=model, 
                C=args.C, 
                p=args.p, 
                gamma=args.gamma, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            ),
            "model": model,
        }
        mcts_list.append(mcts_local)
    
    if args.stochastic_powermean_uct_new:
        spm_uct_new = {
            "name": "Stochastic_Powermean_UCT_New",
            "mcts": Stochastic_Powermean_UCT_New(
                game=game, 
                model=model, 
                C=args.C, 
                p=args.p, 
                gamma=args.gamma, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            ),
            "model": model,
        }
        mcts_list.append(spm_uct_new)

    if args.stochastic_powermean_uct:
        spm_uct = {
            "name": "Stochastic_Powermean_UCT",
            "mcts": Stochastic_Powermean_UCT(
                game=game, 
                model=model, 
                C=args.C, 
                p=args.p, 
                gamma=args.gamma, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            ),
            "model": model,
        }
        mcts_list.append(spm_uct)

    if args.puct:
        puct = {
            "name": "PUCT",
            "mcts": PUCT(
                game=game, 
                model=model, 
                C=args.C, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            ),
            "model": model,
        }
        mcts_list.append(puct)

    results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    num_games_per_pair = args.num_games_per_pair

    logger.info("=" * 70)
    logger.info("TOURNAMENT START")
    logger.info("=" * 70)

    for m1, m2 in itertools.combinations(mcts_list, 2):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"MATCHUP: {m1['name']} vs {m2['name']}")
        logger.info(f"{'=' * 70}")

        for batch_idx in range(num_games_per_pair // args.num_games_parallel):
            logger.info(f"\n--- Round {batch_idx + 1}/{num_games_per_pair // args.num_games_parallel} ---")
            logger.info(f"Game 1: {m1['name']} (Player 1) vs {m2['name']} (Player 2)")
            result = play_game(game, m1, m2, args.num_games_parallel, args.temperature, logger)
            
            results[m1["name"]]["win"] += result["first_win"]
            results[m1["name"]]["loss"] += result["second_win"]
            results[m1["name"]]["draw"] += result["draw"]

            results[m2["name"]]["win"] += result["second_win"]
            results[m2["name"]]["loss"] += result["first_win"]
            results[m2["name"]]["draw"] += result["draw"]

            logger.info(f"\nGame 2: {m2['name']} (Player 1) vs {m1['name']} (Player 2)")
            result = play_game(game, m2, m1, args.num_games_parallel, args.temperature, logger)

            results[m2["name"]]["win"] += result["first_win"]
            results[m2["name"]]["loss"] += result["second_win"]
            results[m2["name"]]["draw"] += result["draw"]

            results[m1["name"]]["win"] += result["second_win"]
            results[m1["name"]]["loss"] += result["first_win"]
            results[m1["name"]]["draw"] += result["draw"]

        logger.info(f"\nFinished matchup: {m1['name']} vs {m2['name']}")
        logger.info(f"Current standings:")
        logger.info(f"  {m1['name']}: {results[m1['name']]}")
        logger.info(f"  {m2['name']}: {results[m2['name']]}")

    logger.info("\n" + "=" * 70)
    logger.info("FINAL TOURNAMENT RESULTS")
    logger.info("=" * 70)
    for name, record in results.items():
        total_games = record["win"] + record["loss"] + record["draw"]
        win_rate = (record["win"] / total_games * 100) if total_games > 0 else 0
        logger.info(f"{name}:")
        logger.info(f"  Wins: {record['win']}")
        logger.info(f"  Losses: {record['loss']}")
        logger.info(f"  Draws: {record['draw']}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info("-" * 50)
    
    logger.info("=" * 70)
    logger.info("TOURNAMENT COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCTS tournament.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    
    parser.add_argument("--num_searches", type=int, default=600, help="Number of MCTS searches per move.")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant C for MCTS.")
    parser.add_argument("--p", type=float, default=1.5, help="Exploration constant p for MCTS.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma for MCTS.")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet noise epsilon for MCTS.")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet noise alpha for MCTS.")

    parser.add_argument("--num_games_per_pair", type=int, default=500, help="Number of games per MCTS pair in the tournament.")
    parser.add_argument("--num_games_parallel", type=int, default=10, help="Number of parallel games to run.")
    parser.add_argument("--temperature", type=float, default=1.25, help="Temperature parameter for MCTS.")

    parser.add_argument("--mcts_global", action='store_true', help="Include `MCTS_Global_Parallel` class to tournament.")
    parser.add_argument("--mcts_local", action='store_true', help="Include `MCTS_Local_Parallel` class to tournament.")
    parser.add_argument("--stochastic_powermean_uct_new", action='store_true', help="Include `Stochastic_Powermean_UCT_New` class to tournament.")
    parser.add_argument("--stochastic_powermean_uct", action='store_true', help="Include `Stochastic_Powermean_UCT` class to tournament.")
    parser.add_argument("--puct", action='store_true', help="Include `PUCT` class to tournament.")

    parser.add_argument("--log_file", type=str, default="tournament.log", help="Path to the log file.")

    args = parser.parse_args()
    run_tournament(args)

# !python /content/powermean-mcts-alphazero/evaluate/tournament.py \
#   --model_checkpoint /content/PUCT_ConnectFour_iteration_8.pt \
#   --num_searches 50 \
#   --C 1.41 \
#   --p 1.5 \
#   --gamma 0.95 \
#   --dirichlet_epsilon 0.25 \
#   --dirichlet_alpha 0.3 \
#   --num_games_per_pair 10 \
#   --num_games_parallel 10 \
#   --mcts_global \
#   --mcts_local \
#   --stochastic_powermean_uct_new \
#   --stochastic_powermean_uct \
#   --puct \
#   --log_file /content/tournament_results.txt