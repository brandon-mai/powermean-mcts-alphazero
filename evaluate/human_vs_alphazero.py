import torch
import numpy as np
import argparse, sys

sys.path.append('/content/powermean-mcts-alphazero/')
sys.path.append('/content/powermean-mcts-alphazero/games')
sys.path.append('/content/powermean-mcts-alphazero/alphazero')
sys.path.append('/content/powermean-mcts-alphazero/mcts')

from games import ConnectFour, Breakthrough, TicTacToe, Havannah, Y, Stochastic_ConnectFour, Stochastic_Breakthrough, Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y
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
    
def create_mcts(algorithm, game, model, args):
    if algorithm == "PUCT":
        return PUCT(
            game=game, 
            model=model, 
            C=args.C, 
            dirichlet_epsilon=args.dirichlet_epsilon, 
            dirichlet_alpha=args.dirichlet_alpha, 
            num_searches=args.num_searches
        )
    elif algorithm == "Stochastic_Powermean_UCT":
        return Stochastic_Powermean_UCT(
            game=game, 
            model=model, 
            C=args.C, 
            p=args.p, 
            gamma=args.gamma, 
            dirichlet_epsilon=args.dirichlet_epsilon, 
            dirichlet_alpha=args.dirichlet_alpha, 
            num_searches=args.num_searches
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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

def play_interactive(args): 
    game = create_game(
        game=args.game
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint_path}")
    model = create_model(
        game=game, 
        device=device,
        args=args
    )
    
    mcts = create_mcts(
        algorithm=args.algorithm, 
        game=game, 
        model=model, 
        args=args
    )

    player = game.get_current_player(
        state=game.get_initial_state()
    )

    print("=" * 50)
    print(f"Welcome to {game.name}!")
    print(f"You are (player {player}). Bot is O (player {game.get_opponent(player)}).")
    print(f"Algorithm: {args.algorithm}")
    print(f"Bot searches: {args.num_searches}")
    print("=" * 50)

    spGame = [SPG(game)]

    while True:
        print("\n" + "=" * 50)
        game.render(spGame[0].state)
        valid_moves = game.get_valid_moves(spGame[0].state)

        if player == 0:
            move = input("Your move: ").strip()
            try:
                move = int(move)
                if move not in valid_moves:
                    print("Invalid move, try again.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue
            action = move
        else:
            states = [spg.state for spg in spGame]
            mcts.search(states, spGame)

            action_probs = np.zeros(game.action_size)
            for child in spGame[0].root.children:
                action_probs[child.action_taken] = child.visit_count

            action = np.argmax(action_probs)
            print(f"{action_probs}")
            print(f"Bot plays at column {action}")

        spGame[0].state = game.get_next_state(
            state=spGame[0].state, 
            action=action)
        
        value, is_terminal = game.get_value_and_terminated(
            state=spGame[0].state, 
            player=player)
        
        if is_terminal:
            game.render(spGame[0].state)
            print("=" * 50)
            
            if (player == 0):
                if (value == 1.0):
                    print("You wins!")
                elif (value == 0.0):
                    print("Bot win!")
                elif (value == 0.5):
                    print("It's a draw")
            elif (player == 1):
                if (value == 1.0):
                    print("Bot wins!")
                elif (value == 0.0):
                    print("You win!")
                elif (value == 0.5):
                    print("It's a draw")                
            break
            
        player = game.get_opponent(player)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against AlphaZero bot.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the model checkpoint.")
    parser.add_argument("--game", type=str, default="ConnectFour",
                        choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                 "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                 "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y"],
                        help="Game to play (default: ConnectFour).")  
    parser.add_argument("--algorithm", type=str, default="PUCT",
                        choices=["PUCT", "Stochastic_Powermean_UCT"],
                        help="MCTS algorithm to use (default: PUCT).")
    parser.add_argument("--num_searches", type=int, default=600, 
                        help="Number of MCTS searches per bot move (default: 600).")
    parser.add_argument("--C", type=float, default=1.41, 
                        help="Exploration constant C for MCTS (default: 1.41).")
    parser.add_argument("--p", type=float, default=1.5, 
                        help="Power parameter p for power mean algorithms (default: 1.5).")
    parser.add_argument("--gamma", type=float, default=0.95, 
                        help="Discount factor gamma for MCTS (default: 0.95).")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.0, 
                        help="Dirichlet noise epsilon for MCTS (default: 0.0).")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.0, 
                        help="Dirichlet noise alpha for MCTS (default: 0.0).")

    args = parser.parse_args()
    play_interactive(args)