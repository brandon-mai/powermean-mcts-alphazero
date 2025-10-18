import math
from alphazero.model import PongAtariResNet, ResNet, weights_init_normal
import torch
from games import PongAtari, ConnectFour, TicTacToe
from alphazero import AlphaZero
from mcts import PUCT, Stochastic_Powermean_UCT, MCTS_Global_Parallel, MCTS_Local_Parallel
import argparse
import numpy as np
from mcts.powermean_one import Stochastic_Powermean_UCT_New

torch.manual_seed(np.random.randint(0, 1000000))


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

        # Lựa chọn thuật toán MCTS
        if args.algorithm == "PUCT":
            mcts = PUCT(
                game=game, 
                model=model, 
                C=args.C, 
                dirichlet_epsilon=args.dirichlet_epsilon, 
                dirichlet_alpha=args.dirichlet_alpha, 
                num_searches=args.num_searches
            )
        elif args.algorithm == "SPU":
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
        elif args.algorithm == "SPUN":
            mcts = Stochastic_Powermean_UCT_New(
                game=game,
                model=model,
                C=args.C,
                p=args.p,
                gamma=args.gamma,
                dirichlet_epsilon=args.dirichlet_epsilon,
                dirichlet_alpha=args.dirichlet_alpha,
                num_searches=args.num_searches
            )
        elif args.algorithm == "MCTS_Global":
            mcts = MCTS_Global_Parallel(
                game=game,
                model=model,
                C=args.C,
                p=args.p,
                gamma=args.gamma,
                dirichlet_epsilon=args.dirichlet_epsilon,
                dirichlet_alpha=args.dirichlet_alpha,
                num_searches=args.num_searches
            )
        elif args.algorithm == "MCTS_Local":
            mcts = MCTS_Local_Parallel(
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
        
        print(f"Đang sử dụng thuật toán: {args.algorithm}")
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
    parser.add_argument("--game", type=str, choices=["pong", "connect4", "tictactoe"], default="connect4", help="Chọn game để train")
    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, 
                        choices=["PUCT", "SPU", "SPUN", "MCTS_Global", "MCTS_Local"], 
                        default="PUCT", 
                        help="Chọn thuật toán MCTS để sử dụng")
    
    # Training parameters
    parser.add_argument("--num_parallel_games", type=int, default=100, help="Số lượng game song song cho MCTS và AlphaZero")
    parser.add_argument("--num_iterations", type=int, default=10, help="Số lượng vòng lặp AlphaZero")
    parser.add_argument("--num_selfPlay_iterations", type=int, default=500, help="Số lượng game tự chơi mỗi vòng lặp AlphaZero")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size cho training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Số lượng epoch mỗi vòng lặp")
    
    # MCTS parameters
    parser.add_argument("--num_searches", type=int, default=600, help="Số lượng tìm kiếm MCTS")
    parser.add_argument("--temperature", type=float, default=1.25, help="Temperature cho việc chọn action")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate cho optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay cho optimizer")
    parser.add_argument("--C", type=float, default=1.41, help="Hằng số exploration cho MCTS")
    parser.add_argument("--p", type=float, default=1.2, help="Tham số power mean cho Stochastic_Powermean_UCT")
    parser.add_argument("--gamma", type=float, default=0.95, help="Tham số gamma cho Stochastic_Powermean_UCT")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet epsilon cho MCTS")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha cho MCTS")
    args = parser.parse_args()

    main(args)
