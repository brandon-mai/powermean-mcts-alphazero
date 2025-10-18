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

def render_connect4(state, connect4):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nBoard:")
    for r in range(connect4.row_count):
        print(" | ".join(symbols[int(x)] for x in state[r]))
        if r < connect4.row_count - 1:
            print("-" * (connect4.column_count * 4 - 1))
    print()

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

    spGame = SPG(game)
    while True:
        render_connect4(spGame.state, game)
        valid_moves = game.get_valid_moves(spGame.state)
        if player == 1:
            move = int(input("Your move (0-6): "))
            if move < 0 or move >= game.column_count or valid_moves[move] == 0:
                print("Invalid move, try again.")
                continue
        else:
            mcts_probs = mtcs.search(neutral_state, spGame)
            neutral_state = game.change_perspective(spGame.state, -1)
            
            # track history (bot's perspective)
            action_probs = np.zeros(spGame.game.action_size)
            for child in spGame.root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            spGame.memory.append((spGame.state, action_probs, player))

            move = int(np.argmax(action_probs))
            print(f"Bot plays at column {move}")

        spGame.state = mtcs.game.get_next_state(spGame.state, move, player)
        value, is_terminal = game.get_value_and_terminated(spGame.state, move)
        if is_terminal:
            render_connect4(spGame.state, game)
            if value == 1:
                print("You win!" if player == 1 else "Bot wins!")
            else:
                print("It's a draw!")
            break
        player = game.get_opponent(player)
