import torch
import numpy as np
import itertools
from games import ConnectFour
from alphazero import ResNet
from mcts import MCTS_Global_Parallel, MCTS_Local_Parallel, PUCT_Parallel

@torch.no_grad()
def play_game(game, first, second):
    state = game.get_initial_state()
    current_player = 1
    while True:
        if current_player == 1:
            policy, _ = first["mtcs"].model(
                torch.tensor(game.get_encoded_state(np.stack([state])), device=first["model"].device)
            )
        else:
            policy, _ = second["mtcs"].model(
                torch.tensor(game.get_encoded_state(np.stack([state])), device=second["model"].device)
            )

        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        valid_moves = game.get_valid_moves(state)
        valid_moves = np.array([1 if j in valid_moves else 0 for j in range(game.action_size)])
        policy *= valid_moves
        policy = policy / policy.sum() if policy.sum() > 0 else valid_moves / valid_moves.sum()

        action = int(torch.multinomial(torch.tensor(policy), 1).item())
        state = game.get_next_state(state, action)
        win, done = game.get_value_and_terminated(state, action)
        if done:
            return current_player if win == 1 else -current_player
        state = game.change_perspective(state, player=-1)
        current_player *= -1


def run_tournament():
    torch.manual_seed(0)
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PUCT_CHECKPOINT = "/content/PUCT_Parallel_<games.connect4.ConnectFour object at 0x792db7c63b60>.pt"
    MCTS_LOCAL_PARALLEL_CHECKPOINT = "/content/PUCT_Parallel_<games.connect4.ConnectFour object at 0x792db7c63b60>.pt"
    MCTS_GLOBAL_PARALLEL_CHECKPOINT = "/content/PUCT_Parallel_<games.connect4.ConnectFour object at 0x792db7c63b60>.pt"

    puct_model = ResNet(game, 9, 128, device)
    puct_model.load_state_dict(torch.load(PUCT_CHECKPOINT, map_location=device))
    puct_model.eval()
    puct = {
        "name": "PUCT",
        "mtcs": PUCT_Parallel(
            game=game, 
            model=puct_model, 
            C=1.41, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=25
        ),
        "model": puct_model,
    }

    mcts_local_model = ResNet(game, 9, 128, device)
    mcts_local_model.load_state_dict(torch.load(MCTS_LOCAL_PARALLEL_CHECKPOINT, map_location=device))
    mcts_local_model.eval()
    mcts_local = {
        "name": "MCTS_Local",
        "mtcs": MCTS_Local_Parallel(
            game=game, 
            model=mcts_local_model, 
            C=1.41, 
            p=1.5, 
            gamma=0.95, 
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=25
        ),
        "model": mcts_local_model,
    }

    mcts_global_model = ResNet(game, 9, 128, device)
    mcts_global_model.load_state_dict(torch.load(MCTS_GLOBAL_PARALLEL_CHECKPOINT, map_location=device))
    mcts_global_model.eval()
    mcts_global = {
        "name": "MCTS_Global",
        "mtcs": MCTS_Global_Parallel(
            game=game, 
            model=mcts_global_model, 
            C=1.41, 
            p=1.5,
            dirichlet_epsilon=0.25, 
            dirichlet_alpha=0.3, 
            num_searches=25
        ),
        "model": mcts_global_model,
    }

    mcts_list = [puct, mcts_local, mcts_global]
    results = {m["name"]: {"win": 0, "loss": 0, "draw": 0} for m in mcts_list}
    num_games_per_pair = 5

    for m1, m2 in itertools.combinations(mcts_list, 2):
        print(f"\n=== {m1['name']} vs {m2['name']} ===")

        for game_idx in range(num_games_per_pair):
            result = play_game(game, m1, m2)
            if result == 1:
                results[m1["name"]]["win"] += 1
                results[m2["name"]]["loss"] += 1
            elif result == -1:
                results[m1["name"]]["loss"] += 1
                results[m2["name"]]["win"] += 1
            else:
                results[m1["name"]]["draw"] += 1
                results[m2["name"]]["draw"] += 1

            result = play_game(game, m2, m1)
            if result == 1:
                results[m2["name"]]["win"] += 1
                results[m1["name"]]["loss"] += 1
            elif result == -1:
                results[m2["name"]]["loss"] += 1
                results[m1["name"]]["win"] += 1
            else:
                results[m2["name"]]["draw"] += 1
                results[m1["name"]]["draw"] += 1

        print(f"Finished {m1['name']} vs {m2['name']}")

    print("\n=== Tournament Results ===")
    for name, record in results.items():
        print(f"{name}: {record}")

if __name__ == "__main__":
    
    run_tournament()
