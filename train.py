import argparse
import torch
import numpy as np
import random
import os
from functools import partial 

from alphazero.model import ResNet, weights_init_normal
from alphazero import AlphaZero 
from mcts import PUCT, Stochastic_Powermean_UCT

from stochastic_muzero.model import StochasticMuZeroNetwork, weights_init_stochastic_muzero
from stochastic_muzero.policy import StochasticMuZero

from games import (
    ConnectFour, Breakthrough, TicTacToe, Havannah, Y,
    Stochastic_ConnectFour, Stochastic_Breakthrough, 
    Stochastic_TicTacToe, Stochastic_Havannah, Stochastic_Y,
    Stochastic_MiniGrid_8x8_Empty, Stochastic_MiniGrid_6x6_Empty_Random,
    Stochastic_FrozenLake_4x4_Random_Map, Stochastic_FrozenLake_8x8_Random_Map,
    Taxi_Is_Raining_Fickle_Passenger
)

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
        "Stochastic_Y": Stochastic_Y,
        "Stochastic_MiniGrid_8x8_Empty": Stochastic_MiniGrid_8x8_Empty,
        "Stochastic_MiniGrid_6x6_Empty_Random": Stochastic_MiniGrid_6x6_Empty_Random,
        "Stochastic_FrozenLake_4x4_Random_Map": Stochastic_FrozenLake_4x4_Random_Map,
        "Stochastic_FrozenLake_8x8_Random_Map": Stochastic_FrozenLake_8x8_Random_Map,
        "Taxi_Is_Raining_Fickle_Passenger": Taxi_Is_Raining_Fickle_Passenger
    }
    return mapping[game_name]

def get_model_config(game_name, algorithm):
    config = {}
    
    if "TicTacToe" in game_name:
        config.update({"num_resBlocks": 5, "num_hidden": 64})
    elif "ConnectFour" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "Breakthrough" in game_name:
        config.update({"num_resBlocks": 12, "num_hidden": 128})
    elif "Havannah" in game_name or "Y" in game_name:
        config.update({"num_resBlocks": 20, "num_hidden": 256})
    elif "Stochastic_MiniGrid" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "FrozenLake_4x4" in game_name:
        config.update({"num_resBlocks": 5, "num_hidden": 64})
    elif "FrozenLake_8x8" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "Taxi" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    
    if algorithm == "StochasticMuZero":
        if "Taxi" in game_name:
             min_val, max_val = -220, 30
        else:
             min_val, max_val = -2, 13 
             
        step = 1
        support_size = (max_val - min_val) // step 
        
        config.update({
            "chance_space_size": 12,
            "support_size": support_size, 
            "support_range": (min_val, max_val, step),
            "use_afterstate": True
        })
            
    return config

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Process Device: {device}")
    print(f"Game: {args.game}")
    print(f"Algorithm: {args.algorithm}")

    GameClass = get_game_class(args.game)
    game = GameClass() 

    model_config = get_model_config(args.game, args.algorithm)

    if args.algorithm == "StochasticMuZero":
        if hasattr(game, 'tensor_shape'):
            obs_shape = game.tensor_shape
        else:
            dummy_state = game.get_initial_state()
            dummy_obs = game.get_encoded_state(dummy_state)
            obs_shape = dummy_obs.shape
            
        model_args = {
            "observation_shape": obs_shape,
            "action_size": game.action_size,
            "hidden_channels": model_config["num_hidden"],
            "num_resblocks": model_config["num_resBlocks"],
            "chance_space_size": model_config.get("chance_space_size", 32),
            "use_afterstate": model_config.get("use_afterstate", False),
            "support_size": model_config.get("support_size", 601),
            "support_range": model_config.get("support_range", (-300, 301, 1)),
            "device": device
        }
        
        ModelClass = StochasticMuZeroNetwork
        model = ModelClass(**model_args)
        
        if args.checkpoint_path:
            print(f"Loading checkpoint: {args.checkpoint_path}")
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        else:
            model.apply(weights_init_stochastic_muzero)
            
    else:
        model_args = {
            "game": game, 
            "device": device,  
            "num_resBlocks": model_config["num_resBlocks"],
            "num_hidden": model_config["num_hidden"]
        }
        
        ModelClass = ResNet
        model = ModelClass(**model_args)

        if args.checkpoint_path:
            print(f"Loading checkpoint: {args.checkpoint_path}")
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        else:
            model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.algorithm == "StochasticMuZero":
        mcts_config = {
            "num_searches": args.num_searches,
            "c_puct": args.C,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "dirichlet_alpha": args.dirichlet_alpha,
            "discount": args.gamma, 
            "use_chance_nodes": True,
            "support_range": model_config.get("support_range", (-300, 301, 1))
        }
        
        worker_model_args = model_args.copy()
        # device handled inside StochasticMuZero via Ray
        
        learner = StochasticMuZero(
            model=model,
            optimizer=optimizer,
            game_cls=GameClass,
            model_cls=ModelClass,
            model_args=worker_model_args,
            mcts_config=mcts_config,
            num_parallel_games=args.num_parallel_games,
            temperature=args.temperature,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            num_selfPlay_iterations=args.num_selfPlay_iterations,
            num_epochs=args.num_epochs,
            games_per_worker=args.num_selfPlay_iterations // args.num_parallel_games,
            discount=args.gamma
        )
        
        learner.learn()

    elif args.algorithm in ["PUCT", "Stochastic_Powermean_UCT"]:
        if args.algorithm == "PUCT":
            mcts_cls = partial(
                PUCT, 
                C=args.C,
                dirichlet_epsilon=args.dirichlet_epsilon,
                dirichlet_alpha=args.dirichlet_alpha,
                num_searches=args.num_searches
            )
        else:
            mcts_cls = partial(
                Stochastic_Powermean_UCT,
                C=args.C,
                p=args.p,
                gamma=args.gamma,
                dirichlet_epsilon=args.dirichlet_epsilon,
                dirichlet_alpha=args.dirichlet_alpha,
                num_searches=args.num_searches
            )
        
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
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full AlphaZero / Stochastic MuZero pipeline.")
    
    # Game selection
    parser.add_argument("--game", type=str, default="ConnectFour",
                        choices=["ConnectFour", "Breakthrough", "TicTacToe", "Havannah", "Y",
                                 "Stochastic_ConnectFour", "Stochastic_Breakthrough", 
                                 "Stochastic_TicTacToe", "Stochastic_Havannah", "Stochastic_Y", 
                                 "Stochastic_MiniGrid",
                                 "Stochastic_MiniGrid_8x8_Empty", "Stochastic_MiniGrid_6x6_Empty_Random",
                                 "Stochastic_FrozenLake_4x4_Random_Map", "Stochastic_FrozenLake_8x8_Random_Map",
                                 "Taxi_Is_Raining_Fickle_Passenger"],
                        help="Game to train.")    
    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, 
                        choices=["PUCT", "Stochastic_Powermean_UCT", "StochasticMuZero"], 
                        default="PUCT", 
                        help="Choose algorithm")
    
    # Training parameters
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_parallel_games", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--num_selfPlay_iterations", type=int, default=100, help="Total games per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per iteration")

    # MCTS parameters
    parser.add_argument("--num_searches", type=int, default=100, help="MCTS searches")
    parser.add_argument("--temperature", type=float, default=1.25, help="Temperature")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--C", type=float, default=1.41, help="Exploration constant (PUCT)")
    parser.add_argument("--p", type=float, default=1.2, help="Power mean parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (MuZero) / Gamma (Powermean)")
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet epsilon")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha")

    args = parser.parse_args()

    main(args)