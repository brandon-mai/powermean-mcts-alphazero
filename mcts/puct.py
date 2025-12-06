import numpy as np
import torch
import math

class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS).
    Optimized with __slots__ and Numpy for memory efficiency and vectorized calculations.
    """
    __slots__ = ['game', 'C', 'state', 'player', 'parent', 'action_taken', 
                 'children', 'visit_count', 'value_sum', 
                 'child_probs', 'child_visits', 'child_values', 'valid_actions']

    def __init__(self, game, C, state, player, parent=None, action_taken=None):
        self.game = game
        self.C = C
        self.state = state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        
        # Map: actual_action -> Node object
        self.children = {} 
        
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Numpy arrays for vectorized child statistics
        self.valid_actions = None
        self.child_probs = None    # Prior probabilities from Policy Network
        self.child_visits = None   # Visit counts (N) for each action
        self.child_values = None   # Total Value (W) for each action

    def is_fully_expanded(self):
        return self.child_probs is not None

    def select(self):
        """
        Selects the next node using the PUCT formula.
        Handles stochastic environments by distinguishing between 
        'Intended Action' (based on UCB) and 'Actual Action' (result of slip).
        """
        # 1. Vectorized UCB Calculation
        # Formula: Q + U = (W/N) + C * P * sqrt(N_parent) / (1 + N_child)
        
        # Calculate Q-values (Mean Value) safely avoiding division by zero
        q_values = np.divide(self.child_values, self.child_visits, 
                             out=np.zeros_like(self.child_values), 
                             where=self.child_visits != 0)
        
        sqrt_n = math.sqrt(self.visit_count)
        u_values = self.C * self.child_probs * sqrt_n / (1 + self.child_visits)
        
        ucb_scores = q_values + u_values

        # Select the best action based on UCB score (Intended Action)
        best_idx = np.argmax(ucb_scores)
        intended_action = self.valid_actions[best_idx]
        
        actual_action = intended_action
        
        # 2. Handle Stochasticity (Slip probability)
        # If slip occurs, we pick a random valid action instead of the intended one.
        if self.game.is_stochastic and np.random.rand() < self.game.randomness:
            actual_action = np.random.choice(self.valid_actions)
            # Note: 'best_idx' remains pointing to the 'intended_action' 
            # to properly update Q-values for the chosen strategy.
            
        # 3. Lazy Expansion
        # Only create the child node object if we actually traverse into it.
        if actual_action not in self.children:
            # Create next state
            if self.game.is_stochastic:
                if hasattr(self.game, 'get_next_absolute_state'):
                    next_state = self.game.get_next_absolute_state(self.state, actual_action)
                else:
                    next_state = self.game.get_next_state(self.state, actual_action)
            else:
                next_state = self.game.get_next_state(self.state, actual_action)

            opponent = self.game.get_opponent(self.player)
            
            self.children[actual_action] = Node(
                game=self.game, C=self.C, state=next_state, 
                player=opponent, parent=self, action_taken=actual_action
            )
            
        # Return the Node we landed on AND the Index of the action we Intended to take.
        return self.children[actual_action], best_idx

    def expand(self, policy, valid_moves):
        """
        Initializes child statistics based on the policy network output.
        """
        self.valid_actions = np.array(valid_moves)
        num_actions = len(valid_moves)
        
        # Normalize policy over valid moves
        probs = policy[valid_moves]
        if np.sum(probs) == 0:
            probs = np.ones(num_actions) / num_actions
        else:
            probs = probs / np.sum(probs)
            
        self.child_probs = probs
        self.child_visits = np.zeros(num_actions, dtype=np.float32)
        self.child_values = np.zeros(num_actions, dtype=np.float32)


class PUCT:
    """
    Standard AlphaZero PUCT algorithm implementation with Batch Processing support.
    """
    def __init__(self, game, model, C, dirichlet_epsilon, dirichlet_alpha, num_searches):
        self.game = game
        self.model = model
        self.C = C
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.num_searches = num_searches

    @torch.no_grad()
    def search(self, states, spGames):
        # --- 1. Root Expansion & Batch Prediction ---
        # Handle batch encoding for efficiency
        if isinstance(states, list):
             if len(states) > 0 and isinstance(states[0], np.ndarray):
                 encoded_states = np.stack(states)
                 tensor_states = torch.tensor(encoded_states, device=self.model.device)
             else:
                 encoded_states = self.game.get_encoded_state(states)
                 tensor_states = torch.tensor(encoded_states, device=self.model.device)
        else:
             encoded_states = self.game.get_encoded_state(states)
             tensor_states = torch.tensor(encoded_states, device=self.model.device)

        policies, _ = self.model(tensor_states)
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        
        # Apply Dirichlet noise to the root node for exploration
        noise = np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size, size=len(spGames))
        policies = (1 - self.dirichlet_epsilon) * policies + self.dirichlet_epsilon * noise
        
        for i, spg in enumerate(spGames):
            spg.root = Node(self.game, self.C, states[i], self.game.get_current_player(states[i]))
            valid_moves = self.game.get_valid_moves(states[i])
            spg.root.expand(policies[i], valid_moves)
        
        # --- 2. MCTS Simulation Loop ---
        for _ in range(self.num_searches):
            expandable_spGames = []
            expandable_nodes = []
            
            # Phase A: Selection
            for i, spg in enumerate(spGames):
                node = spg.root
                spg_path = [] # Stores list of (parent_node, intended_action_index)
                
                while node.is_fully_expanded():
                    # Select next node
                    next_node, intended_idx = node.select()
                    
                    # Store path for backpropagation
                    spg_path.append((node, intended_idx))
                    node = next_node

                # Check for Terminal State
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
                
                if is_terminal:
                    # If terminal, backpropagate immediately
                    self._backpropagate_path(spg_path, node, value)
                else:
                    spg.node = node
                    spg.path = spg_path 
                    expandable_spGames.append(i)
                    expandable_nodes.append(node)

            if not expandable_spGames:
                continue

            # Phase B: Evaluation (Batch)
            states_to_eval = [n.state for n in expandable_nodes]
            
            if isinstance(states_to_eval[0], np.ndarray):
                 input_tensor = torch.tensor(np.stack(states_to_eval), device=self.model.device)
            else:
                 input_tensor = torch.tensor(self.game.get_encoded_state(states_to_eval), device=self.model.device)

            p_preds, v_preds = self.model(input_tensor)
            p_preds = torch.softmax(p_preds, dim=1).cpu().numpy()
            v_preds = v_preds.cpu().numpy().flatten()
            
            # Phase C: Expansion & Backpropagation
            for i, idx in enumerate(expandable_spGames):
                spg = spGames[idx]
                node = spg.node
                policy = p_preds[i]
                value = (v_preds[i] + 1) / 2 # Normalize value to [0, 1]
                
                valid_moves = self.game.get_valid_moves(node.state)
                node.expand(policy, valid_moves)
                
                # Backpropagate the evaluated value up to the root
                self._backpropagate_path(spg.path, node, value)

    def _backpropagate_path(self, path, leaf_node, value):
        """
        Iterative Backpropagation helper.
        Updates visit counts and values from the leaf node up to the root.
        
        Args:
            path: List of (parent_node, action_index_used)
            leaf_node: The newly evaluated node
            value: Value of the leaf node (from the perspective of the leaf's player)
        """
        # 1. Update Leaf Node
        leaf_node.visit_count += 1
        leaf_node.value_sum += value
        
        current_value = value 
        
        # 2. Iterate up to the Root
        for i in range(len(path) - 1, -1, -1):
            parent_node, action_idx = path[i]
            
            # Flip value because the parent is the opponent 
            current_value = 1.0 - current_value
            
            # Update Parent Stats (Vectorized array update)
            parent_node.visit_count += 1
            parent_node.value_sum += current_value
            
            parent_node.child_visits[action_idx] += 1
            parent_node.child_values[action_idx] += current_value