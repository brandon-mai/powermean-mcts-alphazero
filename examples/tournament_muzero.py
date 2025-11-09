"""
Tournament for MuZero Models
Test MuZero vs MuZero, MuZero vs AlphaZero, hoặc MuZero vs Random
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import time
from tqdm import tqdm

from games import ConnectFour
from muzero.model import MuZeroNetwork
from mcts import MuZeroMCTS, PUCT
from alphazero import ResNet


class SPG:
    """Self-Play Game wrapper"""
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.root = None
        self.node = None


def create_muzero_player(checkpoint_path, game, device, num_searches=100):
    """Tạo MuZero player từ checkpoint"""
    print(f"📥 Loading MuZero: {checkpoint_path}")
    
    observation_shape = (game.num_planes, game.row_count, game.column_count)
    model = MuZeroNetwork(
        observation_shape=observation_shape,
        action_space_size=game.action_size,
        num_res_blocks=2,
        num_channels=32,
        use_chance_encoder=False,
        use_afterstate=False,
        use_categorical=True,
        reward_support_range=(-10., 11., 1.),
        value_support_range=(-10., 11., 1.),
        device=device
    )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    mcts = MuZeroMCTS(
        game=game,
        model=model,
        num_searches=num_searches,
        c_puct=1.41,
        use_chance_nodes=False
    )
    
    print(f"✅ MuZero loaded ({num_searches} searches)")
    return {"type": "muzero", "mcts": mcts, "model": model}


def create_alphazero_player(checkpoint_path, game, device, num_searches=100):
    """Tạo AlphaZero player từ checkpoint"""
    print(f"📥 Loading AlphaZero: {checkpoint_path}")
    
    # Auto-detect architecture from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect num_hidden from first layer
    if 'startBlock.0.weight' in checkpoint:
        num_hidden = checkpoint['startBlock.0.weight'].shape[0]
    else:
        num_hidden = 128  # default
    
    # Detect num_resBlocks from backbone layers
    num_resBlocks = len([k for k in checkpoint.keys() if k.startswith('backBone.') and k.endswith('.conv1.weight')])
    
    print(f"   Detected: {num_resBlocks} ResBlocks, {num_hidden} channels")
    
    model = ResNet(
        game=game,
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden,
        device=device
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    mcts = PUCT(
        game=game,
        model=model,
        C=2.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        num_searches=num_searches
    )
    
    print(f"✅ AlphaZero loaded ({num_searches} searches)")
    return {"type": "alphazero", "mcts": mcts, "model": model}


def create_random_player(game):
    """Tạo random player"""
    print("🎲 Creating Random player")
    return {"type": "random", "game": game}


def play_tournament(game, player1, player2, num_games=10):
    """
    Chơi tournament giữa 2 players
    
    Args:
        game: Game object
        player1: Dict với keys: type, mcts/model
        player2: Dict với keys: type, mcts/model
        num_games: Số games chơi
    
    Returns:
        Dict với win/loss/draw statistics
    """
    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "games": []
    }
    
    print(f"\n{'='*80}")
    print(f"🎮 TOURNAMENT: {player1['type'].upper()} vs {player2['type'].upper()}")
    print(f"   Total games: {num_games}")
    print(f"{'='*80}\n")
    
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        # Alternate starting player
        if game_idx % 2 == 0:
            p1, p2 = player1, player2
            p1_name, p2_name = "Player1", "Player2"
        else:
            p1, p2 = player2, player1
            p1_name, p2_name = "Player2", "Player1"
        
        # Play one game
        winner, moves = play_single_game(game, p1, p2)
        
        # Record result
        game_result = {
            "game_idx": game_idx + 1,
            "starting_player": p1_name,
            "winner": winner,
            "moves": moves
        }
        results["games"].append(game_result)
        
        # Update statistics
        if winner == 0:
            if game_idx % 2 == 0:
                results["player1_wins"] += 1
            else:
                results["player2_wins"] += 1
        elif winner == 1:
            if game_idx % 2 == 0:
                results["player2_wins"] += 1
            else:
                results["player1_wins"] += 1
        else:
            results["draws"] += 1
    
    return results


def play_single_game(game, player1, player2):
    """
    Chơi 1 game giữa 2 players
    
    Returns:
        winner: 0 (player1 wins), 1 (player2 wins), -1 (draw)
        moves: số moves trong game
    """
    state = game.get_initial_state()
    player = 0
    moves = 0
    max_moves = 100  # Prevent infinite games
    
    while moves < max_moves:
        current_player = player1 if player == 0 else player2
        
        # Get action từ player
        if current_player["type"] == "random":
            # Random player
            valid_moves = game.get_valid_moves(state)
            action = np.random.choice(valid_moves)
        else:
            # MCTS-based player (MuZero or AlphaZero)
            spg = SPG(game)
            spg.state = state
            
            with torch.no_grad():
                current_player["mcts"].search([state], [spg])
            
            # Get action from root node
            if hasattr(current_player["mcts"], 'get_action_probs'):
                # MuZero MCTS
                action_probs = current_player["mcts"].get_action_probs(
                    spg.root, temperature=0
                )
                action = np.argmax(action_probs)
            else:
                # AlphaZero PUCT - extract from visit counts
                action_probs = np.zeros(game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                if np.sum(action_probs) > 0:
                    action = np.argmax(action_probs)
                else:
                    # Fallback to valid moves
                    valid_moves = game.get_valid_moves(state)
                    action = np.random.choice(valid_moves)
        
        # Make move
        state = game.get_next_state(state, action)
        moves += 1
        
        # Check terminal
        value, is_terminal = game.get_value_and_terminated(state, player)
        
        if is_terminal:
            if value == 1:
                return player, moves  # Current player wins
            elif value == 0:
                return -1, moves  # Draw
            else:
                return 1 - player, moves  # Opponent wins
        
        # Switch player
        player = game.get_opponent(player)
    
    # Max moves reached -> draw
    return -1, moves


def print_results(results, player1_name, player2_name):
    """In kết quả tournament"""
    total_games = len(results["games"])
    p1_wins = results["player1_wins"]
    p2_wins = results["player2_wins"]
    draws = results["draws"]
    
    print(f"\n{'='*80}")
    print(f"📊 TOURNAMENT RESULTS")
    print(f"{'='*80}")
    print(f"\n{player1_name}:")
    print(f"   Wins: {p1_wins}/{total_games} ({p1_wins/total_games*100:.1f}%)")
    print(f"\n{player2_name}:")
    print(f"   Wins: {p2_wins}/{total_games} ({p2_wins/total_games*100:.1f}%)")
    print(f"\nDraws: {draws}/{total_games} ({draws/total_games*100:.1f}%)")
    
    # Average moves
    avg_moves = np.mean([g["moves"] for g in results["games"]])
    print(f"\nAverage game length: {avg_moves:.1f} moves")
    
    # Win rate
    if p1_wins + p2_wins > 0:
        p1_win_rate = p1_wins / (p1_wins + p2_wins) * 100
        print(f"\n{player1_name} win rate (excluding draws): {p1_win_rate:.1f}%")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="MuZero Tournament")
    
    # Game
    parser.add_argument("--game", type=str, default="ConnectFour",
                       help="Game to play")
    
    # Players
    parser.add_argument("--player1", type=str, default="muzero",
                       choices=["muzero", "alphazero", "random"],
                       help="Player 1 type")
    parser.add_argument("--player1_checkpoint", type=str,
                       default="checkpoint/Stochastic_MuZero_MCTS_ConnectFour_iteration_1.pt",
                       help="Player 1 checkpoint path")
    parser.add_argument("--player1_searches", type=int, default=50,
                       help="MCTS searches for player 1")
    
    parser.add_argument("--player2", type=str, default="random",
                       choices=["muzero", "alphazero", "random"],
                       help="Player 2 type")
    parser.add_argument("--player2_checkpoint", type=str,
                       default="checkpoint/PUCT_ConnectFour_iteration_10.pt",
                       help="Player 2 checkpoint path")
    parser.add_argument("--player2_searches", type=int, default=50,
                       help="MCTS searches for player 2")
    
    # Tournament
    parser.add_argument("--num_games", type=int, default=10,
                       help="Number of games to play")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Create game
    if args.game == "ConnectFour":
        game = ConnectFour()
    else:
        raise ValueError(f"Unsupported game: {args.game}")
    
    print(f"🎮 Game: {game.name}")
    
    # Create players
    print(f"\n{'='*80}")
    print("SETTING UP PLAYERS")
    print(f"{'='*80}\n")
    
    if args.player1 == "muzero":
        player1 = create_muzero_player(
            args.player1_checkpoint, game, device, args.player1_searches
        )
    elif args.player1 == "alphazero":
        player1 = create_alphazero_player(
            args.player1_checkpoint, game, device, args.player1_searches
        )
    else:
        player1 = create_random_player(game)
    
    print()
    
    if args.player2 == "muzero":
        player2 = create_muzero_player(
            args.player2_checkpoint, game, device, args.player2_searches
        )
    elif args.player2 == "alphazero":
        player2 = create_alphazero_player(
            args.player2_checkpoint, game, device, args.player2_searches
        )
    else:
        player2 = create_random_player(game)
    
    # Play tournament
    start_time = time.time()
    results = play_tournament(game, player1, player2, args.num_games)
    elapsed_time = time.time() - start_time
    
    # Print results
    player1_name = f"{args.player1.upper()}"
    player2_name = f"{args.player2.upper()}"
    
    if args.player1 != "random":
        player1_name += f" ({args.player1_searches} searches)"
    if args.player2 != "random":
        player2_name += f" ({args.player2_searches} searches)"
    
    print_results(results, player1_name, player2_name)
    
    print(f"\n⏱️  Total time: {elapsed_time:.1f}s")
    print(f"   Average per game: {elapsed_time/args.num_games:.1f}s")
    
    # Save results
    os.makedirs("tournament_results", exist_ok=True)
    result_file = f"tournament_results/{args.player1}_vs_{args.player2}_{args.num_games}games.txt"
    
    with open(result_file, "w") as f:
        f.write(f"Tournament Results\n")
        f.write(f"==================\n\n")
        f.write(f"Game: {game.name}\n")
        f.write(f"Player 1: {player1_name}\n")
        f.write(f"Player 2: {player2_name}\n")
        f.write(f"Total games: {args.num_games}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Player 1 wins: {results['player1_wins']}\n")
        f.write(f"  Player 2 wins: {results['player2_wins']}\n")
        f.write(f"  Draws: {results['draws']}\n")
    
    print(f"\n💾 Results saved to: {result_file}")


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("🏆 MUZERO TOURNAMENT")
    print(f"{'='*80}\n")
    
    main()
    
    print(f"\n{'='*80}")
    print("🎉 TOURNAMENT COMPLETED!")
    print(f"{'='*80}\n")

