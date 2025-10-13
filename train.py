import math
import torch

torch.manual_seed(0)

from games import ConnectFour
from alphazero import AlphaZero, ResNet
from mcts import MCTS_Global_Parallel, MCTS_Local_Parallel, PUCT_Parallel

def main():
    game = ConnectFour()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)

    mtcs = PUCT_Parallel(
            game=game, 
            model=model, 
            C=1.41, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=25
    )
    
    # mtcs = MCTS_Local_Parallel(
    #         game=game, 
    #         model=model, 
    #         C=1.41, 
    #         p=1.5, 
    #         gamma=0.95, 
    #         dirichlet_epsilon=0.25, 
    #         dirichlet_alpha=0.3, 
    #         num_searches=25
    #     )    

    # mtcs = MCTS_Global_Parallel(
    #         game=game, 
    #         model=model, 
    #         C=1.41, 
    #         p=1.5,
    #         dirichlet_epsilon=0.25, 
    #         dirichlet_alpha=0.3, 
    #         num_searches=25
    #     )

    alphaZero = AlphaZero(
        model=model, 
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
        game=game,
        mcts=mtcs,
        num_parallel_games=100, # for parallel mcts and alphazero
        temperature=1.25,
        batch_size=128,
        num_iterations=1,
        num_selfPlay_iterations=500, # Number of self-play games per AlphaZero iteration
        num_epochs=4
    )
    alphaZero.learn()

if __name__ == "__main__":
    main()
