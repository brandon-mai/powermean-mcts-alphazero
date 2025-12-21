import numpy as np
import math
import torch
import copy

class Node:
    """
    Represents a node in the Monte Carlo Tree Search.
    optimized with __slots__ and Numpy for memory and speed efficiency.
    """
    __slots__ = ['game', 'state', 'player', 'C', 'p', 'gamma', 
                 'parent', 'action_taken', 'children', 
                 'visit_count', 'v_value', 
                 'valid_actions', 'child_probs', 
                 'child_visits', 'child_q_values']

    def __init__(self, game, state, player, C, p, gamma, parent=None, action_taken=None):
        self.game = game
        self.state = state
        self.player = player
        self.C = C
        self.p = p
        self.gamma = gamma
        self.parent = parent
        self.action_taken = action_taken
        
        # Map: actual_action -> Node object
        self.children = {} 
        
        self.visit_count = 0
        self.v_value = 0.0  # Stores the PowerMean aggregated value
        
        # Numpy arrays for vectorized child statistics
        self.valid_actions = None
        self.child_probs = None    # Prior probabilities from Policy Network
        self.child_visits = None   # Visit counts (N) for each action
        self.child_q_values = None # Q-values for each action

    def is_fully_expanded(self):
        return self.child_probs is not None
    
    def select_child(self):
        epsilon = 1e-8
        
        n_pow = math.pow(self.visit_count, 0.25) if self.visit_count > 0 else 0
        
        u_term = self.C * (n_pow / (np.sqrt(self.child_visits) + epsilon)) * self.child_probs
        
        ucb_scores = self.child_q_values - u_term
        
        mask_unvisited = (self.child_visits == 0)
        ucb_scores[mask_unvisited] = -np.inf
        
        worst_idx = np.argmin(ucb_scores)
        
        intended_action = self.valid_actions[worst_idx]
        actual_action = intended_action

        # 2. Handle Stochasticity (Slip probability)
        if self.game.is_stochastic and np.random.rand() < self.game.randomness:
            actual_action = np.random.choice(self.valid_actions)
        
        # 3. Lazy Expansion
        if actual_action not in self.children:
            # Optimize state copying
            if hasattr(self.state, 'clone'): 
                next_state = self.state.clone()
            else: 
                next_state = copy.deepcopy(self.state)

            # Apply action
            if self.game.is_stochastic:
                next_state = self.game.get_next_absolute_state(next_state, actual_action)
            else:
                next_state = self.game.get_next_state(next_state, actual_action)

            opponent = self.game.get_opponent(self.player)
            
            self.children[actual_action] = Node(
                game=self.game, state=next_state, player=opponent,
                C=self.C, p=self.p, gamma=self.gamma,
                parent=self, action_taken=actual_action
            )
            
        return self.children[actual_action], worst_idx

    def expand(self, policy, valid_moves, top_k=None):
        """
        Initializes child statistics based on the policy network output.
        Applies Top-K Pruning if top_k is specified.
        """
        valid_moves = np.array(valid_moves)
        probs = policy[valid_moves]
        
        if top_k is not None and len(valid_moves) > top_k:
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            valid_moves = valid_moves[top_indices]
            probs = probs[top_indices]
            
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(len(valid_moves)) / len(valid_moves)
            
        self.valid_actions = valid_moves
        self.child_probs = probs
        num_actions = len(valid_moves)
        
        self.child_visits = np.zeros(num_actions, dtype=np.float32)
        self.child_q_values = np.zeros(num_actions, dtype=np.float32)


    def compute_powermean_value(self):
        """
        Computes the PowerMean value using the 'Leapfrog' method.
        
        Logic:
        - If a child is a Leaf: Value = (1 + gamma) * child.v_value
        - If a child is Expanded: Value = Aggregate of Grandchildren's Q-values
        
        Formula: V = ( Sum( weight * val^p ) / Sum(weight) ) ^ (1/p)
        """
        
        values_to_mean = []
        weights = []
        
        for child in self.children.values():
            if not child.is_fully_expanded(): 
                # Case 1: Leaf Child
                val = (1 + self.gamma) * child.v_value
                
                values_to_mean.append(val)
                weights.append(child.visit_count)
            else: 
                # Case 2: Expanded Child (Leapfrog to Grandchildren)
                mask = child.child_visits > 0
                if np.any(mask):
                    q_vals = child.child_q_values[mask]
                    vis_vals = child.child_visits[mask]
                    
                    values_to_mean.extend(q_vals)
                    weights.extend(vis_vals)

        # Vectorized calculation
        np_vals = np.array(values_to_mean, dtype=np.float32)
        np_weights = np.array(weights, dtype=np.float32)
        
        total_weight = np.sum(np_weights)
        if total_weight == 0:
            return self.v_value  # No update possible

        # PowerMean Calculation
        powered = np.power(np_vals, self.p)
        weighted_sum = np.sum(np_weights * powered)
        
        res = weighted_sum / total_weight
        return np.power(res, 1.0 / self.p)

class Stochastic_Powermean_UCT:
    def __init__(self, game, model, C=1.41, p=1.5, gamma=0.95,
                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3, num_searches=25, top_k=None):
        self.name = "Stochastic_Powermean_UCT"
        self.game = game
        self.model = model
        self.C = C
        self.p = p
        self.gamma = gamma
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.num_searches = num_searches
        self.top_k = top_k

    @torch.no_grad()
    def search(self, states, spGames):    
        # 1. Batch Encoding & Model Prediction
        if isinstance(states, list) and len(states) > 0 and isinstance(states[0], np.ndarray):
            tensor_states = torch.tensor(np.stack(states), device=self.model.device)
        else:
            encoded = self.game.get_encoded_state(states)
            tensor_states = torch.tensor(encoded, device=self.model.device)

        policies, _ = self.model(tensor_states)
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        
        # Add Dirichlet Noise for exploration
        noise = np.random.dirichlet([self.dirichlet_alpha] * self.game.action_size, size=len(spGames))
        policies = (1 - self.dirichlet_epsilon) * policies + self.dirichlet_epsilon * noise
        
        # 2. Root Initialization
        for i, spg in enumerate(spGames):
            spg_policy = policies[i]
            valid_moves = self.game.get_valid_moves(states[i])
            
            # Mask invalid moves
            mask = np.zeros(self.game.action_size)
            mask[valid_moves] = 1
            spg_policy *= mask
            s = np.sum(spg_policy)
            
            if s > 0: 
                spg_policy /= s
            else: 
                spg_policy[valid_moves] = 1.0 / len(valid_moves)

            spg.root = Node(
                game=self.game, state=states[i], player=self.game.get_current_player(states[i]),
                C=self.C, p=self.p, gamma=self.gamma, parent=None
            )
            spg.root.expand(spg_policy, valid_moves, top_k=self.top_k)
            spg.root.visit_count = 1 
        
        # 3. MCTS Simulation Loop
        for _ in range(self.num_searches):
            expandable_spGames = []
            expandable_nodes = []
            
            # --- Selection Phase ---
            for i, spg in enumerate(spGames):
                node = spg.root
                path = [] # Stores (parent, action_index) tuples
                
                while node.is_fully_expanded():
                    # Select next node using Worst-Action logic
                    node, idx_in_parent = node.select_child()
                    path.append((node.parent, idx_in_parent))

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
                
                if is_terminal:
                    # backpropagate immediately
                    self._backpropagate(path, node, node.player, value)
                    
                else:
                    spg.node = node
                    spg.path = path
                    expandable_spGames.append(i)
                    expandable_nodes.append(node)
            
            if not expandable_spGames:
                continue
                
            # --- Evaluation Phase ---
            states_eval = [n.state for n in expandable_nodes]
            
            enc_states = self.game.get_encoded_state(states_eval)
            input_tensor = torch.tensor(enc_states, device=self.model.device)
            
            p_preds, v_preds = self.model(input_tensor)
            p_preds = torch.softmax(p_preds, dim=1).cpu().numpy()
            v_preds = v_preds.cpu().numpy().flatten()
            
            # --- Expansion & Backpropagation Phase ---
            for i, idx in enumerate(expandable_spGames):
                spg = spGames[idx]
                node = spg.node
                
                policy = p_preds[i]
                value = (v_preds[i] + 1) / 2 # Normalize to [0, 1]
                
                valid_moves = self.game.get_valid_moves(node.state)
                node.expand(policy, valid_moves, top_k=self.top_k)
                
                # backpropagate 
                self._backpropagate(spg.path, node, node.player, value)

    def _backpropagate(self, path, start_node, update_player, final_reward):
        # start from the leaf node
        current_node = start_node
        current_node.visit_count += 1
        current_node.v_value = final_reward

        # second leaf node (parent of start_node)
        is_travesed_second_leaf = False
        
        # Traverse up the path
        for i in range(len(path) - 1, -1, -1):
            parent, action_idx = path[i]

            immediate_reward, _ = self.game.get_value_and_terminated(current_node.state, current_node.player)
            
            n_visit = parent.child_visits[action_idx]
            old_q = parent.child_q_values[action_idx]
            
            # Q Update Formula: Q_new = (Q_old * N + Reward + Gamma * V_child) / (N + 1)
            update_target = (old_q * n_visit + immediate_reward + self.gamma * current_node.v_value) / (n_visit + 1)
            
            parent.child_q_values[action_idx] = update_target
            parent.child_visits[action_idx] += 1

            # Update V-value (PowerMean)
            if not is_travesed_second_leaf:
                parent.visit_count += 1
                parent.v_value = self.game.get_opponent_value(final_reward)
                
                is_travesed_second_leaf = True            
            else:            
                parent.visit_count += 1
                parent.v_value = parent.compute_powermean_value()

            # Move up the tree to maintain topology
            current_node = parent