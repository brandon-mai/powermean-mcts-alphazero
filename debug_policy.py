import torch
import numpy as np
import math
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from games.stochastic_minigrid_8x8_empty import Stochastic_MiniGrid_8x8_Empty
from stochastic_muzero.model import StochasticMuZeroNetwork
from stochastic_muzero.powermean_mcts import StochasticMuZeroMCTS

def test_overconfidence():
    print("Initializing Game...")
    game = Stochastic_MiniGrid_8x8_Empty()
    obs_shape = game.tensor_shape
    action_size = game.action_size
    
    device = "cpu"
    print("Initializing Model...")
    model = StochasticMuZeroNetwork(obs_shape, action_size, 
                                    hidden_channels=16, 
                                    num_resblocks=2,
                                    support_size=601,
                                    device=device)
    model.eval()
    
    print("Initializing MCTS...")
    # Using 100 searches to see distribution clearly
    mcts = StochasticMuZeroMCTS(game, model, num_searches=100, C=1.14, p=1.0)
    
    state = game.get_initial_state()
    spg = type('SPG', (object,), {'root': None})() 
    
    print("Running MCTS Search...")
    mcts.search([state], [spg])
    
    root = spg.root
    print("\nRoot Visit Counts:")
    total_visits = 0
    
    visits = []
    
    # Sort children by visits
    children_items = list(root.children.items())
    children_items.sort(key=lambda x: x[1][1].visit_count, reverse=True)
    
    for action, (q_node, child) in children_items:
        visits.append(child.visit_count)
        print(f"Action {action}:")
        print(f"  Visits: {child.visit_count}")
        print(f"  Q-value: {q_node['q_value']:.4f}")
        print(f"  Prior: {child.prior:.4f}")
        
        # Calculate pb_c as per code
        pb_c = mcts.C * math.sqrt(math.sqrt(root.visit_count) / (child.visit_count + 1e-6))
        print(f"  Code pb_c term (factor): {pb_c:.4f}")
        print(f"  Code Prior Score (Prior * pb_c): {child.prior * pb_c:.4f}")
        
        # Standard PUCT 
        # using sqrt(N)/(1+n)
        std_pb_c = mcts.C * math.sqrt(root.visit_count) / (child.visit_count + 1)
        print(f"  Standard PUCT term (factor): {std_pb_c:.4f}")
        print(f"  Standard Prior Score: {child.prior * std_pb_c:.4f}")
        
        total_visits += child.visit_count

    print(f"\nTotal parent visits (N): {root.visit_count}")
    print(f"Sum of child visits: {total_visits}")
    
    visits = np.array(visits)
    if np.sum(visits) > 0:
        dist = visits / np.sum(visits)
        print(f"\nPolicy Distribution: {dist}")
        entropy = -np.sum(dist * np.log(dist + 1e-9))
        print(f"Policy Entropy: {entropy:.4f}")
    else:
        print("No visits recorded.")

if __name__ == "__main__":
    test_overconfidence()
