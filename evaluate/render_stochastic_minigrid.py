import argparse
import sys
import os
import torch
import numpy as np
import random
from PIL import Image
import gymnasium as gym

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'games'))
sys.path.append(os.path.join(project_root, 'alphazero'))
sys.path.append(os.path.join(project_root, 'mcts'))
sys.path.append(os.path.join(project_root, 'stochastic_muzero'))

from games import (
    Stochastic_MiniGrid_8x8_Empty, Stochastic_MiniGrid_6x6_Empty_Random,
    Stochastic_FrozenLake_4x4_Random_Map, Stochastic_FrozenLake_8x8_Random_Map,
    Taxi_Is_Raining_Fickle_Passenger
)

from stochastic_muzero.model import StochasticMuZeroNetwork
from stochastic_muzero.mcts import StochasticMuZeroMCTS as ClassicMCTS
from stochastic_muzero.powermean_mcts import StochasticMuZeroMCTS as PowerMeanMCTS

def get_game_class(game_name):
    print(f"Loading game: {game_name}")
    mapping = {
        "Stochastic_MiniGrid_8x8_Empty": Stochastic_MiniGrid_8x8_Empty,
        "Stochastic_MiniGrid_6x6_Empty_Random": Stochastic_MiniGrid_6x6_Empty_Random,
        "Stochastic_FrozenLake_4x4_Random_Map": Stochastic_FrozenLake_4x4_Random_Map,
        "Stochastic_FrozenLake_8x8_Random_Map": Stochastic_FrozenLake_8x8_Random_Map,
        "Taxi_Is_Raining_Fickle_Passenger": Taxi_Is_Raining_Fickle_Passenger
    }
    return mapping[game_name]

def get_model_config(game_name):
    config = {}
    if "Stochastic_MiniGrid" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "FrozenLake_4x4" in game_name:
        config.update({"num_resBlocks": 5, "num_hidden": 64})
    elif "FrozenLake_8x8" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    elif "Taxi" in game_name:
        config.update({"num_resBlocks": 9, "num_hidden": 128})
    
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

class StochasticMuZeroSPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.root = None
        self.total_reward = 0.0

def run_render(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    GameClass = get_game_class(args.game)
    game = GameClass()
         
    print(f"Game: {game.name}")

    model_config = get_model_config(game.name)
    
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

    model = StochasticMuZeroNetwork(**model_args)
    
    print("Triggering lazy initialization...")
    obs_shape = model.observation_shape
    dummy_obs = torch.zeros((1, *obs_shape), device=device)
    hidden = model.representation(dummy_obs)
    dummy_action = torch.zeros((1,), dtype=torch.long, device=device)
    next_hidden, _ = model.dynamics(hidden, dummy_action)
    model.prediction(hidden)
    
    if model.use_afterstate:
        model.afterstate_prediction(next_hidden)
        chance_onehot = torch.zeros((1, model.chance_space_size), device=device)
        model.afterstate_dynamics(hidden, chance_onehot)
    
    if args.checkpoint_path and args.checkpoint_path != "random":
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Using random weights")
    
    model.to(device)
    model.eval()

    mcts_config = {
        "num_searches": args.num_searches,
        "c_puct": args.C,
        "p": args.p, 
        "dirichlet_epsilon": 0.0, 
        "dirichlet_alpha": 0.01,
        "discount": args.gamma, 
        "use_chance_nodes": True,
        "support_range": model_config.get("support_range", (-300, 301, 1))
    }

    mcts_type = args.mcts_type
    MCTSClass = PowerMeanMCTS if mcts_type == 'powermean' else ClassicMCTS
    
    mcts_kwargs = {
        "game": game,
        "model": model,
        "num_searches": mcts_config['num_searches'],
        "dirichlet_epsilon": mcts_config['dirichlet_epsilon'], 
        "dirichlet_alpha": mcts_config['dirichlet_alpha'],
        "discount": mcts_config['discount'],
        "use_chance_nodes": mcts_config['use_chance_nodes'],
        "support_range": mcts_config['support_range']
    }
    
    if mcts_type == 'powermean':
        mcts_kwargs["C"] = mcts_config['c_puct']
        mcts_kwargs["p"] = mcts_config['p']
    else:
        mcts_kwargs["c_puct"] = mcts_config['c_puct']

    mcts = MCTSClass(**mcts_kwargs)

    spg = StochasticMuZeroSPG(game)
    frames = []

    print("Starting simulation...")
    
    try:
        frame = game.render(spg.state)
        if frame is not None:
            frames.append(Image.fromarray(frame))
    except Exception as e:
        print(f"Render failed: {e}")

    step = 0
    while step < args.max_steps:
        mcts.search([spg.state], [spg])
        
        root = spg.root
        action_probs = mcts.get_action_probs(root, args.temperature)


        # Pick action
        if args.temperature < 1e-3:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
            
        root_val = 0.0
        if hasattr(root, 'value') and callable(root.value):
            root_val = root.value()
        elif hasattr(root, 'v_value'):
            root_val = root.v_value
             
        print(f"Step {step}: Action {action}, Value {root_val:.4f}, Action prob {action_probs}")

        # Apply action
        spg.state = game.get_next_state(spg.state, action)
        
        # Collect reward
        reward = 0.0
        if hasattr(spg.state, 'reward'): 
            reward = spg.state.reward
        elif hasattr(spg.state, 'custom_reward'): 
            reward = spg.state.custom_reward
        
        spg.total_reward += reward
        
        # Render
        try:
            frame = game.render(spg.state)
            if frame is not None:
                frames.append(Image.fromarray(frame))
        except Exception as e:
            raise e

        # Check termination
        _, is_terminal = game.get_value_and_terminated(spg.state, 0)
        
        if is_terminal:
            print(f"Game terminated at step {step}. Total Reward: {spg.total_reward}")
            break
        
        step += 1

    if frames:
        output_filename = args.output
        print(f"Saving GIF to {output_filename} ({len(frames)} frames)...")
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration,  # Duration per frame in milliseconds
            loop=0
        )
        print("Done.")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Stochastic_MiniGrid_8x8_Empty", 
                        choices=["Stochastic_MiniGrid_8x8_Empty", "Stochastic_MiniGrid_6x6_Empty_Random",
                                 "Stochastic_FrozenLake_4x4_Random_Map", "Stochastic_FrozenLake_8x8_Random_Map",
                                 "Taxi_Is_Raining_Fickle_Passenger"], help="Game to render")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--mcts_type", type=str, default="classic", choices=["classic", "powermean"])
    parser.add_argument("--num_searches", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--C", type=float, default=1.41)
    parser.add_argument("--p", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--output", type=str, default="render_output.gif")
    parser.add_argument("--duration", type=int, default=500, help="Frame duration in ms")



    args = parser.parse_args()
    
    run_render(args)
