import torch
import numpy as np
import itertools
from games import ConnectFour
from alphazero import ResNet
from mcts import PUCT, Stochastic_Powermean_UCT
import argparse
import logging
from datetime import datetime
import os


def setup_logger(log_dir="logs"):
    """
    Thiết lập logger để ghi kết quả vào file
    """
    # Tạo thư mục logs nếu chưa có
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Tạo tên file log với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tournament_{timestamp}.log")
    
    # Cấu hình logger
    logger = logging.getLogger("tournament")
    logger.setLevel(logging.INFO)
    
    # Xóa các handler cũ nếu có
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler - ghi vào file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler - hiển thị trên màn hình
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"=== Logger initialized. Log file: {log_file} ===")
    
    return logger

@torch.no_grad()
def play_game(game, first, second, logger=None):
    if logger:
        logger.info("=" * 50)
        logger.info("NEW GAME_dafix")
        logger.info(f"Player 1: {first['name']}")
        logger.info(f"Player 2: {second['name']}")
    
    state = game.get_initial_state()
    current_player = 1
    move_count = 0
    debug_state = state.copy()
    while True:
        if current_player == 1:
            policy, _ = first["mtcs"].model(
                torch.tensor(game.get_encoded_state(np.stack([state])), device=first["model"].device)
            )
        else:
            policy, _ = second["mtcs"].model(
                torch.tensor(game.get_encoded_state(np.stack([state])), device=second["model"].device)
            )

        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        valid_moves = game.get_valid_moves(state)
        valid_moves = np.array([1 if j in valid_moves else 0 for j in range(game.action_size)])
        policy *= valid_moves
        policy = policy / policy.sum() if policy.sum() > 0 else valid_moves / valid_moves.sum()

        action = int(torch.multinomial(torch.tensor(policy), 1).item())
        state = game.get_next_state(state, action)
        if(current_player == 1):
            debug_state = state.copy()
        else:
            debug_state = game.change_perspective(state, player=-1)
            
        move_count += 1
        current_player_name = first['name'] if current_player == 1 else second['name']
        
        # if logger:
        #     logger.info(f"Move {move_count} - {current_player_name} plays action: {action}")
        #     # logger.info(f"state: \n {debug_state}")
        
        win, done = game.get_value_and_terminated(state, current_player)
        # logger.info(f"win: {win} - done: {done}")
        if done:
            if logger:
                logger.info(f"debug_state: \n {debug_state}")
                if win == 1:
                    logger.info(f"Game Over! Winner: {current_player_name}")
                elif win == 0.5:
                    logger.info("Game Over! Draw")
                elif win == 0:
                    opponent_name = second['name'] if current_player == 1 else first['name']
                    logger.info(f"Game Over! Winner: {opponent_name}")
            
            if(win == 1):
                return current_player
            elif win == 0.5:
                return 0
            elif win == 0:
                return -current_player
        
        state = game.change_perspective(state, player=-1)
        current_player *= -1
    

def run_tournament(args):
    # Khởi tạo logger
    logger = setup_logger()
    
    torch.manual_seed(np.random.randint(0, 1000000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = ConnectFour()

    PUCT_CHECKPOINT = args.puct_checkpoint
    STOCHASTIC_POWERMEAN_UCT_CHECKPOINT = args.stochastic_powermean_uct_checkpoint

    logger.info("=" * 70)
    logger.info("TOURNAMENT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Game: ConnectFour")
    logger.info(f"PUCT checkpoint: {PUCT_CHECKPOINT}")
    logger.info(f"Stochastic Powermean UCT checkpoint: {STOCHASTIC_POWERMEAN_UCT_CHECKPOINT}")

    puct_model = ResNet(game, 9, 128, device)
    puct_model.load_state_dict(torch.load(PUCT_CHECKPOINT, map_location=device))
    puct_model.eval()
    puct = {
        "name": "PUCT",
        "mtcs": PUCT(
            game=game, 
            model=puct_model, 
            C=1.41, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=600
        ),
        "model": puct_model,
    }

    stochastic_powermean_uct_model= ResNet(game, 9, 128, device)
    stochastic_powermean_uct_model.load_state_dict(torch.load(STOCHASTIC_POWERMEAN_UCT_CHECKPOINT, map_location=device))
    stochastic_powermean_uct_model.eval()
    stochastic_powermean_uct = {
        "name": "Stochastic_Powermean_UCT",
        "mtcs": Stochastic_Powermean_UCT(
            game=game, 
            model=stochastic_powermean_uct_model, 
            C=1.41, 
            p=1.2, 
            gamma=0.95, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=600
        ),
        "model": stochastic_powermean_uct_model,
    }

    mcts_list = [puct, stochastic_powermean_uct]
    results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    num_games_per_pair = 500

    logger.info("=" * 70)
    logger.info("TOURNAMENT START")
    logger.info("=" * 70)

    for m1, m2 in itertools.combinations(mcts_list, 2):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"MATCHUP: {m1['name']} vs {m2['name']}")
        logger.info(f"{'=' * 70}")

        for game_idx in range(num_games_per_pair):
            logger.info(f"\n--- Round {game_idx + 1}/{num_games_per_pair} ---")
            logger.info(f"Game 1: {m1['name']} (Player 1) vs {m2['name']} (Player 2)")
            result = play_game(game, m1, m2, logger)
            
            if result == 1:
                results[m1["name"]]["win"] += 1
                results[m2["name"]]["loss"] += 1
                logger.info(f"Result: {m1['name']} wins!")
            elif result == -1:
                results[m1["name"]]["loss"] += 1
                results[m2["name"]]["win"] += 1
                logger.info(f"Result: {m2['name']} wins!")
            else:
                results[m1["name"]]["draw"] += 1
                results[m2["name"]]["draw"] += 1
                logger.info("Result: Draw!")

            logger.info(f"\nGame 2: {m2['name']} (Player 1) vs {m1['name']} (Player 2)")
            result = play_game(game, m2, m1, logger)
            
            if result == 1:
                results[m2["name"]]["win"] += 1
                results[m1["name"]]["loss"] += 1
                logger.info(f"Result: {m2['name']} wins!")
            elif result == -1:
                results[m2["name"]]["loss"] += 1
                results[m1["name"]]["win"] += 1
                logger.info(f"Result: {m1['name']} wins!")
            else:
                results[m2["name"]]["draw"] += 1
                results[m1["name"]]["draw"] += 1
                logger.info("Result: Draw!")

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
    parser.add_argument("--puct_checkpoint", type=str, required=True, help="Path to the PUCT model checkpoint.")
    parser.add_argument("--stochastic_powermean_uct_checkpoint", type=str, required=True, help="Path to the Stochastic Powermean UCT model checkpoint.")
    args = parser.parse_args()
    run_tournament(args)
# python -m tournament --puct_checkpoint checkpoint\PUCT_ConnectFour_iteration_10.pt --stochastic_powermean_uct_checkpoint checkpoint\Stochastic_Powermean_UCT_ConnectFour_iteration_10.pt