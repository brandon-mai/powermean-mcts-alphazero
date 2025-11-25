import math
from alphazero.model import ResNet, weights_init_normal
import torch
from games import ConnectFour, Breakthrough, TicTacToe, Havannah, Y, Stochastic_ConnectFour, Stochastic_Breakthrough, Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y
from alphazero import AlphaZero
from mcts import PUCT, Stochastic_Powermean_UCT
import argparse
import numpy as np

torch.manual_seed(0)

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

def create_model(game, device, args):
    if (game.name == "TicTacToe" or game.name == "Stochastic_TicTacToe"):
        model = ResNet(
            game=game, 
            num_resBlocks=5, 
            num_hidden=64, 
            device=device
        )
        return model
    elif (game.name == "ConnectFour" or game.name == "Stochastic_ConnectFour"):
        model = ResNet(
            game=game, 
            num_resBlocks=9, 
            num_hidden=128, 
            device=device
        )
        return model    
    elif (game.name == "Breakthrough" or game.name == "Stochastic_Breakthrough"):
        model = ResNet(
            game=game, 
            num_resBlocks=12, 
            num_hidden=128, 
            device=device
        )
        return model
    elif (game.name == "Havannah" or game.name == "Stochastic_Havannah"):
        model = ResNet(
            game=game, 
            num_resBlocks=20, 
            num_hidden=256, 
            device=device
        )
        return model
    elif (game.name == "Y" or game.name == "Stochastic_Y"):
        model = ResNet(
            game=game, 
            num_resBlocks=20, 
            num_hidden=256, 
            device=device
        )
        return model


def main(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        game = create_game(
            game=args.game
        )

        model = create_model(
            game=game,
            device=device,
            args=args
        ) 

        if args.checkpoint_path:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        else:
            model.apply(weights_init_normal)

        if args.algorithm == "PUCT":
            mcts = PUCT(
                game=game, 
                model=model, 
                C=args.C, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            )
        elif args.algorithm == "Stochastic_Powermean_UCT":
            mcts = Stochastic_Powermean_UCT(
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
            raise ValueError(f"Unknown algorithm: {args.algorithm}")
        
        print(f"Algorithm: {args.algorithm}")
        print(f"Game: {args.game}")
        print(f"Device: {device}")
        
        alphaZero = AlphaZero(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
            game=game,
            mcts=mcts,
            num_parallel_games=args.num_parallel_games,
            temperature=args.temperature,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_selfPlay_iterations=args.num_selfPlay_iterations,
            num_epochs=args.num_epochs
        )
        alphaZero.learn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full AlphaZero pipeline.")
    
    # Game selection
    parser.add_argument("--game", type=str, default="ConnectFour",
                        choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                 "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                 "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y"],
                        help="Game to train (default: ConnectFour).")    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, 
                        choices=["PUCT", "Stochastic_Powermean_UCT"], 
                        default="PUCT", 
                        help="Choose MCTS algorithm")
    
    # Training parameters
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint file used for loading pretrained weights or resuming training.")
    parser.add_argument("--num_parallel_games", type=int, default=100, help="Number of parallel games for MCTS and AlphaZero")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of AlphaZero iterations")
    parser.add_argument("--num_selfPlay_iterations", type=int, default=500, help="Number of self-play games per AlphaZero iteration")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs per iteration")

    # MCTS parameters
    parser.add_argument("--num_searches", type=int, default=600, help="Number of MCTS searches")
    parser.add_argument("--temperature", type=float, default=1.25, help="Temperature for action selection")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant for MCTS")
    parser.add_argument("--p", type=float, default=1.2, help="Power mean parameter for Stochastic_Powermean_UCT")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma parameter for Stochastic_Powermean_UCT")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet epsilon for MCTS")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha for MCTS")

    args = parser.parse_args()

    main(args)