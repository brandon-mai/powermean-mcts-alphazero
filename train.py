import math
from alphazero.model import PongAtariResNet
import torch

torch.manual_seed(0)

from games import ConnectFour, PongAtari
from alphazero import AlphaZero, ResNet
from mcts import PUCT, Stochastic_Powermean_MCTS

def main():
    game = PongAtari()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PongAtariResNet(game, 9, 128, device)

    mtcs = PUCT(
            game=game, 
            model=model, 
            C=1.41, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=600
    )

    # mtcs = Stochastic_Powermean_MCTS(
    #         game=game, 
    #         model=model, 
    #         C=1.41, 
    #         p=1.2, 
    #         gamma=0.95, 
    #         dirichlet_epsilon=0.25, 
    #         dirichlet_alpha=0.3, 
    #         num_searches=600
    # )

    alphaZero = AlphaZero(
        model=model, 
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
        game=game,
        mcts=mtcs,
        num_parallel_games=100, # for parallel mcts and alphazero
        temperature=1.25,
        batch_size=128,
        num_iterations=8,
        num_selfPlay_iterations=500, # Number of self-play games per AlphaZero iteration
        num_epochs=4
    )
    alphaZero.learn()

if __name__ == "__main__":
    main()
