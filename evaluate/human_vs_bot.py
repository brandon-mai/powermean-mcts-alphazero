import torch
import numpy as np
import argparse, sys, os

sys.path.append('/content/powermean-mcts-alphazero/')
sys.path.append('/content/powermean-mcts-alphazero/games')
sys.path.append('/content/powermean-mcts-alphazero/alphazero')
sys.path.append('/content/powermean-mcts-alphazero/mcts')

from games import ConnectFour
from alphazero import ResNet
from mcts import MCTS_Global_Parallel, MCTS_Local_Parallel, \
    Stochastic_Powermean_UCT_New, Stochastic_Powermean_UCT, PUCT


class SPG:
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.memory = []

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
    elif algorithm == "MCTS_Global":
        return MCTS_Global_Parallel(
            game=game, 
            model=model, 
            C=args.C, 
            p=args.p, 
            gamma=args.gamma, 
            dirichlet_epsilon=args.dirichlet_epsilon, 
            dirichlet_alpha=args.dirichlet_alpha, 
            num_searches=args.num_searches
        )
    elif algorithm == "MCTS_Local":
        return MCTS_Local_Parallel(
            game=game, 
            model=model, 
            C=args.C, 
            p=args.p, 
            gamma=args.gamma, 
            dirichlet_epsilon=args.dirichlet_epsilon, 
            dirichlet_alpha=args.dirichlet_alpha, 
            num_searches=args.num_searches
        )
    elif algorithm == "Stochastic_Powermean_UCT_New":
        return Stochastic_Powermean_UCT_New(
            game=game, 
            model=model, 
            C=args.C, 
            p=args.p, 
            gamma=args.gamma, 
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

def play_interactive(args):
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint_path}")
    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    
    mcts = create_mcts(args.algorithm, game, model, args)

    print("=" * 50)
    print("Welcome to Connect Four!")
    print("You are X (player 1). Bot is O (player -1).")
    print(f"Algorithm: {args.algorithm}")
    print(f"Temperature: {args.temperature}")
    print(f"Bot searches: {args.num_searches}")
    print("=" * 50)

    player = 1
    spGame = [SPG(game)]
    
    while True:
        game.render(spGame[0].state)
        valid_moves = game.get_valid_moves(spGame[0].state)

        if player == 1:
            move = input("Your move (0-6): ").strip()
            try:
                move = int(move)
                if move < 0 or move >= game.column_count or move not in valid_moves:
                    print("Invalid move, try again.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue
            action = move
        else:
            states = np.stack([spg.state for spg in spGame])
            neutral_states = game.change_perspective(states, player)
            mcts.search(neutral_states, spGame)

            action_probs = np.zeros(game.action_size)
            for child in spGame[0].root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            temperature_action_probs = action_probs ** (1 / args.temperature)
            if np.sum(temperature_action_probs) == 0:
                temperature_action_probs = np.ones_like(temperature_action_probs) / len(temperature_action_probs)
            else:
                temperature_action_probs /= np.sum(temperature_action_probs)
            
            action = np.random.choice(game.action_size, p=temperature_action_probs)
            print(f"Bot plays at column {action}")

        spGame[0].state = game.get_next_state(spGame[0].state, action)
        value, is_terminal = game.get_value_and_terminated(spGame[0].state, player)
        if is_terminal:
            game.render(spGame[0].state)
            print("=" * 50)
            
            if player == 1:
                if value == 1:
                    print("You win!")
                elif value == 0.5:
                    print("It's a draw!")
                elif value == 0.0:
                    print("Bot wins!")
            else:
                if value == 1:
                    print("Bot wins!")
                elif value == 0.5:
                    print("It's a draw!")
                elif value == 0.0:
                    print("You win!")

            break
            
        player = game.get_opponent(player)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Connect Four against MCTS bot.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the model checkpoint.")
    parser.add_argument("--algorithm", type=str, default="PUCT",
                        choices=["PUCT", "MCTS_Global", "MCTS_Local", 
                                "Stochastic_Powermean_UCT_New", "Stochastic_Powermean_UCT"],
                        help="MCTS algorithm to use (default: PUCT).")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature parameter for bot move selection (default: 1.0).")
    
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