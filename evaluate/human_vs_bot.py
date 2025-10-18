import torch
import numpy as np


from games import ConnectFour
from alphazero import ResNet
from mcts import PUCT

class SPG:
    def __init__(self, game):
        self.game = game
        self.state = game.get_initial_state()
        self.memory = []

if __name__ == "__main__":
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = "/content/PUCT_ConnectFour.pt"
    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    mtcs = PUCT(
        game=game, 
        model=model, 
        C=1.41, 
        dirichlet_epsilon=0.25, 
        dirichlet_alpha=0.3, 
        num_searches=600
    )

    print("You are X (player 1). Bot is O (player -1).")
    player = 1

    spGame = [SPG(game)]
    while True:
        game.render(spGame[0].state)
        valid_moves = game.get_valid_moves(spGame[0].state)

        neutral_state = game.change_perspective(spGame[0].state, player)
        if player == 1:
            move = int(input("Your move (0-6): "))
            if move < 0 or move >= game.column_count or valid_moves[move] == 0:
                print("Invalid move, try again.")
                continue
        else:
            mtcs.search(neutral_state, spGame)
            neutral_state = game.change_perspective(spGame[0].state, -1)

            action_probs = np.zeros(game.action_size)
            for child in spGame[0].root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            temperature_action_probs = action_probs ** (1 / 1)
            if np.sum(temperature_action_probs) == 0:
                temperature_action_probs = np.ones_like(temperature_action_probs) / len(temperature_action_probs)
            else:
                temperature_action_probs /= np.sum(temperature_action_probs)
            
            action = np.random.choice(game.action_size, p=temperature_action_probs)
            print(f"Bot plays at column {action}")

        spGame[0].state = mtcs.game.get_next_state(spGame[0].state, action, player)
        value, is_terminal = game.get_value_and_terminated(spGame[0].state, action)
        if is_terminal:
            game.render(spGame[0].state)
            if value == 1:
                print("You win!" if player == 1 else "Bot wins!")
            else:
                print("It's a draw!")
            break
        player = game.get_opponent(player)
