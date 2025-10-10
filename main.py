import math
import torch
from games import ConnectFour
from models import ResNet
from alphazero import AlphaZero


def main():
    game = ConnectFour()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 1.41,
        'num_searches': 25,
        'num_iterations': 1,             # Number of AlphaZero sp/train/eval loops
        'num_selfPlay_iterations': 500,  # Number of self-play games per AlphaZero iteration
        'num_parallel_games': 100,       # for parallel mcts and alphazero
        'num_epochs': 4,                 # Number of training epochs per AlphaZero iteration
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'gamma': 0.95,
        'p': 1.5,
        'exploration_fn': lambda parent_visits, child_visits, C: C * math.pow(parent_visits, 0.25) / math.sqrt(child_visits),
        'num_rollout': 20,
        'num_workers': 3
    }

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


if __name__ == "__main__":
    main()
