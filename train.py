import math
from alphazero.model import PongAtariResNet, ResNet, weights_init_normal
import torch
from games import PongAtari, ConnectFour, TicTacToe
from alphazero import AlphaZero
from mcts import PUCT, Stochastic_Powermean_UCT
import argparse

torch.manual_seed(0)


def main(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.game == "pong":
            game = PongAtari()
            model = PongAtariResNet(game, 9, 128, device)
        elif args.game == "connect4":
            game = ConnectFour()
            model = ResNet(game, 9, 128, device)
        elif args.game == "tictactoe":
            game = TicTacToe()
            model = ResNet(game, 9, 64, device)
        else:
            raise ValueError(f"Unknown game: {args.game}")
        
        model.apply(weights_init_normal)

        mtcs = Stochastic_Powermean_UCT(
            game=game,
            model=model,
            C=args.C,
            p=args.p,
            gamma=args.gamma,
            dirichlet_epsilon=args.dirichlet_epsilon,
            dirichlet_alpha=args.dirichlet_alpha,
            num_searches=args.num_searches
        )

        alphaZero = AlphaZero(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
            game=game,
            mcts=mtcs,
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
    parser.add_argument("--game", type=str, choices=["pong", "connect4", "tictactoe"], default="connect4", help="Select the game to train on")
    
    # Training parameters
    parser.add_argument("--num_parallel_games", type=int, default=300, help="Number of parallel games for MCTS and AlphaZero")
    parser.add_argument("--num_iterations", type=int, default=8, help="Number of AlphaZero iterations")
    parser.add_argument("--num_selfPlay_iterations", type=int, default=300, help="Number of self-play games per AlphaZero iteration")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs per iteration")
    
    # MCTS parameters
    parser.add_argument("--num_searches", type=int, default=100, help="Number of MCTS searches")
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
