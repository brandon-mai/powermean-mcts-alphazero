import argparse
import torch
import numpy as np
import random
import os
from functools import partial 

from alphazero.model import ResNet, weights_init_normal
from alphazero import AlphaZero 
from mcts import PUCT, Stochastic_Powermean_UCT

from games import (
    ConnectFour, Breakthrough, TicTacToe, Havannah, Y,
    Stochastic_ConnectFour, Stochastic_Breakthrough, 
    Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y
)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_game_class(game_name):
    mapping = {
        "ConnectFour": ConnectFour,
        "Breakthrough": Breakthrough,
        "TicTacToe": TicTacToe,
        "Havannah": Havannah,
        "Y": Y,
        "Stochastic_ConnectFour": Stochastic_ConnectFour,
        "Stochastic_Breakthrough": Stochastic_Breakthrough,
        "Stochastic_TicTacToe": Stochastic_TicTacToe,
        "Stochastic_Havannah": Stochastic_Havannah,
        "Stochastic_Y": Stochastic_Y
    }
    return mapping[game_name]

def get_model_config(game_name):
    if "TicTacToe" in game_name:
        return {"num_resBlocks": 5, "num_hidden": 64}
    elif "ConnectFour" in game_name:
        return {"num_resBlocks": 9, "num_hidden": 128}
    elif "Breakthrough" in game_name:
        return {"num_resBlocks": 12, "num_hidden": 128}
    elif "Havannah" in game_name or "Y" in game_name:
        return {"num_resBlocks": 20, "num_hidden": 256}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Process Device: {device}")
    print(f"Game: {args.game}")
    print(f"Algorithm: {args.algorithm}")

    GameClass = get_game_class(args.game)
    game = GameClass() 

    model_config = get_model_config(args.game)

    model_args = {
        "game": game, 
        "device": "cuda",  
        **model_config
    }

    main_model_args = model_args.copy()
    main_model_args["device"] = device
    
    model = ResNet(**main_model_args)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        model.apply(weights_init_normal)

    if args.algorithm == "PUCT":

        mcts_cls = partial(
            PUCT, 
            C=args.C,
            dirichlet_epsilon=args.dirichlet_epsilon,
            dirichlet_alpha=args.dirichlet_alpha,
            num_searches=args.num_searches
        )
    elif args.algorithm == "Stochastic_Powermean_UCT":
        mcts_cls = partial(
            Stochastic_Powermean_UCT,
            C=args.C,
            p=args.p,
            gamma=args.gamma,
            dirichlet_epsilon=args.dirichlet_epsilon,
            dirichlet_alpha=args.dirichlet_alpha,
            num_searches=args.num_searches
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    alphaZero = AlphaZero(
        model=model,
        optimizer=optimizer,
        game_cls=GameClass,       
        mcts_cls=mcts_cls,        
        model_cls=ResNet,         
        model_args=model_args,    
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
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_parallel_games", type=int, default=4, help="Number of parallel workers (CPUs)")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of AlphaZero iterations")
    parser.add_argument("--num_selfPlay_iterations", type=int, default=100, help="Total games per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per iteration")

    # MCTS parameters
    parser.add_argument("--num_searches", type=int, default=100, help="MCTS searches")
    parser.add_argument("--temperature", type=float, default=1.25, help="Temperature")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant")
    parser.add_argument("--p", type=float, default=1.2, help="Power mean parameter")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma parameter")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet epsilon")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha")

    args = parser.parse_args()

    main(args)