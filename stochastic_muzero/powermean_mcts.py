import numpy as np
import torch
import math
import collections

class Node:
    def __init__(self, prior, C, p, v_value):
        self.visit_count = 0
        
        self.v_value = v_value
        self.C = C
        self.p = p

        self.prior = prior
        """
        action -> child, also track Q_value, construct as a tuple ({
            "q_value": q_value,
            "visit_count": visit_count
        }, child)
        """
        self.children = {} 
        self.hidden_state = None
        self.reward = 0.0
        
    def expanded(self):
        return len(self.children) > 0
        
    def v_value_func(self):
        sum_q = 0.0
        for q_node, child in self.children.values():
            sum_q += pow(q_node["q_value"], self.p) * (q_node["visit_count"] / self.visit_count)             
        return sum_q ** (1 / self.p)

class DecisionNode(Node):
    def __init__(self, prior, C, p, v_value):
        super().__init__(prior, C, p, v_value)
        self.to_play = 0

class ChanceNode(Node):
    def __init__(self, prior, C, p, v_value):
        super().__init__(prior, C, p, v_value)

class MinMaxStats:
    def __init__(self, known_bounds=None):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        self.known_bounds = known_bounds

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.known_bounds:
            min_v, max_v = self.known_bounds
            return (value - min_v) / (max_v - min_v)
            
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class StochasticMuZeroMCTS:
    def __init__(self, game, model, num_searches, C=1.14, p=1.0, 
                dirichlet_epsilon=0.25, dirichlet_alpha=0.3,
                discount=0.997, use_chance_nodes=False, support_range=(-300, 301, 1)):
        self.game = game
        self.model = model
        self.num_searches = num_searches
        self.C = C
        self.p = p
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.discount = discount
        self.use_chance_nodes = use_chance_nodes
        self.support_range = support_range
        
    def search(self, current_states, spgs):
        min_max_stats = MinMaxStats()
        
        # Initial Inference (Root)
        obs_batch = []
        for state in current_states:
            obs = self.game.get_encoded_state(state)
            obs_batch.append(obs)
            
        if hasattr(self.model, 'device'):
            obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.model.device)
        else:
            obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32)
        
        with torch.no_grad():
            root_hidden, root_policy, root_value, root_reward, root_chance = \
                self.model.initial_inference(obs_tensor)
                
        root_hidden = root_hidden.cpu().numpy()
        root_policy = self._softmax_batch(root_policy.cpu().numpy())
        root_value = self._value_batch(root_value.cpu().numpy())
        
        if np.isnan(root_policy).any():
            print(f"Error: NaN in root_policy (PowerMean).")
            raise RuntimeError("NaN in root_policy (PowerMean)")

        # Initialize Roots
        for i, spg in enumerate(spgs):
            root = DecisionNode(1.0, self.C, self.p, 0.0) 
            root.hidden_state = root_hidden[i]
            root.to_play = self.game.get_current_player(current_states[i])
            
            policy = root_policy[i]
            
            # Mask illegal moves
            legal_moves = self.game.get_valid_moves(current_states[i])
            mask = np.zeros_like(policy)
            mask[legal_moves] = 1.0
            policy = policy * mask
            
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy = policy / sum_policy
            else:
                # uniform over legal moves
                policy[legal_moves] = 1.0 / len(legal_moves)

            if self.dirichlet_epsilon > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
                policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
                
            for action_idx, prob in enumerate(policy):
                if self.use_chance_nodes:
                    child = ChanceNode(prob, self.C, self.p, 0.0)
                else:
                    child = DecisionNode(prob, self.C, self.p, 0.0)
                root.children[action_idx] = ({"q_value": 0.0, "visit_count": 0}, child)
                
            spg.root = root
            self._backpropagate([root], root_value[i], 0.0, min_max_stats)

        # Simulations 
        for _ in range(self.num_searches):
            decision_parents = [] # (spg_index, parent_node, action, child_node)
            chance_parents = []   # (spg_index, parent_node, outcome, child_node)
            search_paths = []     # List of paths for each game

            # --- Selection (Parallel) ---
            for i, spg in enumerate(spgs):
                node = spg.root
                path = [node]
                
                while node.expanded():
                    if isinstance(node, DecisionNode):
                        action, node = self._select_action(node, min_max_stats)
                    elif isinstance(node, ChanceNode):
                        outcome, node = self._select_chance(node, min_max_stats)
                    else:
                        break
                    path.append(node)
                
                search_paths.append(path)
                
                # transition to the leaf
                parent = path[-2]
                child = path[-1]
                
                action_or_outcome = None
                for act, (q_node, c) in parent.children.items():
                    if c is child:
                        action_or_outcome = act
                        break
                
                if isinstance(parent, DecisionNode):
                    decision_parents.append((i, parent, action_or_outcome, child))
                elif isinstance(parent, ChanceNode):
                    chance_parents.append((i, parent, action_or_outcome, child))

            with torch.no_grad():
                # helper for rescaling
                min_v, max_v, step = self.support_range
                real_max = max_v - step
                real_min = min_v
                val_range = real_max - real_min
                if val_range == 0: 
                    val_range = 1.0

                def rescale(x):
                    return (x - real_min) / val_range

                # decision -> [afterstate] -> decision
                if decision_parents:
                    hidden_list = []
                    action_list = []
                    
                    for _, p, a, _ in decision_parents:
                        hidden_list.append(p.hidden_state)
                        action_list.append(a)
                        
                    hidden_tensor = torch.tensor(np.array(hidden_list), device=self.model.device)
                    action_tensor = torch.tensor(np.array(action_list), device=self.model.device)
                    
                    # dynamics
                    next_hidden, reward_logits = self.model.dynamics(hidden_tensor, action_tensor)
                    rewards = self._scalar_value(reward_logits).cpu().numpy()
                    rewards = rescale(rewards)
                    
                    if self.use_chance_nodes:
                        # afterstate predict
                        chance_logits, value_logits = self.model.afterstate_prediction(next_hidden)
                        chance_probs_batch = self._softmax_batch(chance_logits.cpu().numpy())
                        values = self._value_batch(value_logits.cpu().numpy())
                        values = rescale(values)
                        
                        next_hidden_np = next_hidden.cpu().numpy()
                        
                        for idx, (original_idx, _, _, child) in enumerate(decision_parents):
                            child.hidden_state = next_hidden_np[idx]
                            child.reward = rewards[idx] # this is also the intermedate reward used for Q_node that move to it
                            
                            # expand 
                            for outcome_idx, prob in enumerate(chance_probs_batch[idx]):
                                child.children[outcome_idx] = ({"q_value": 0.0, "visit_count": 0}, DecisionNode(prob, self.C, self.p, 0.0))
                                
                            self._backpropagate(search_paths[original_idx], values[idx], 0.0, min_max_stats)

                # chance -> [next state] -> chance
                if chance_parents:
                    hidden_list = []
                    outcome_onehots = []
                    
                    chance_space = self.model.chance_space_size
                    
                    for _, p, o, _ in chance_parents:
                        hidden_list.append(p.hidden_state)
                        onehot = np.zeros(chance_space)
                        onehot[o] = 1.0
                        outcome_onehots.append(onehot)
                        
                    hidden_tensor = torch.tensor(np.array(hidden_list), device=self.model.device)
                    outcome_tensor = torch.tensor(np.array(outcome_onehots), dtype=torch.float32, device=self.model.device)
                    
                    # afterstate dynamics
                    next_hidden, _ = self.model.afterstate_dynamics(hidden_tensor, outcome_tensor)
                    
                    # prediction
                    policy_logits, value_logits = self.model.prediction(next_hidden)
                    policy_probs_batch = self._softmax_batch(policy_logits.cpu().numpy())
                    values = self._value_batch(value_logits.cpu().numpy())
                    values = rescale(values)
                    
                    next_hidden_np = next_hidden.cpu().numpy()
                    
                    for idx, (original_idx, _, _, child) in enumerate(chance_parents):
                        child.hidden_state = next_hidden_np[idx]
                        child.reward = 0.0 # no reward on chance transition
                        
                        for action_idx, prob in enumerate(policy_probs_batch[idx]):
                            if self.use_chance_nodes:
                                child.children[action_idx] = ({"q_value": 0.0, "visit_count": 0}, ChanceNode(prob, self.C, self.p, 0.0))
                                
                        self._backpropagate(search_paths[original_idx], values[idx], 0.0, min_max_stats)
        
    def _select_action(self, node, min_max_stats):
        max_score = -float('inf')
        best_child = None
        best_action = -1
        
        for action, (q_node, child) in node.children.items():
            if min_max_stats.maximum == min_max_stats.minimum:
                q_value = q_node["q_value"]
            else:
                q_value = min_max_stats.normalize(q_node["q_value"])
                
            pb_c = self.C * math.sqrt(math.sqrt(node.visit_count) / (child.visit_count + 1e-6))
            
            prior_score = child.prior * pb_c
            score = q_value + prior_score
            
            if score > max_score:
                max_score = score
                best_child = child
                best_action = action
                
        return best_action, best_child
        
    def _select_chance(self, node, min_max_stats):
        # randomly pick child according to node prior
        outcomes = []
        children = []
        priors = []
        
        for action, (q_node, child) in node.children.items():
            outcomes.append(action)
            children.append(child)
            priors.append(child.prior)
            
        priors = np.array(priors, dtype=np.float32)
        sum_priors = np.sum(priors)
        if sum_priors > 0:
            priors /= sum_priors
        else:
            priors = np.ones_like(priors) / len(priors)
            
        selected_idx = np.random.choice(len(outcomes), p=priors)
        return outcomes[selected_idx], children[selected_idx]

    def _backpropagate(self, search_path, value, mix_reward, min_max_stats):
        current_value = value
        
        current_node = search_path[-1]
        current_node.v_value = value
        current_node.visit_count += 1

        for parent_node in reversed(search_path[:-1]):
            for action, (q_node, child) in parent_node.children.items():
                # update Q_node
                if child == current_node:
                    q_node["q_value"] = (q_node["q_value"] * q_node["visit_count"] + current_node.reward + self.discount * current_node.v_value) / (q_node["visit_count"] + 1)
                    q_node["visit_count"] += 1
                    min_max_stats.update(q_node["q_value"])
                    break    
            
            # update V_node
            parent_node.visit_count += 1
            parent_node.v_value = parent_node.v_value_func()

    def _tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.unsqueeze(0)
        return torch.tensor(arr, device=self.model.device).unsqueeze(0)
    
    def _softmax_batch(self, logits):
        if logits.ndim == 1:
            logits = logits[None, :]
        
        if np.isnan(logits).any():
            print(f"Error: NaN in logits input to _softmax_batch (PowerMean). shape={logits.shape}")
            raise RuntimeError("NaN in logits input to _softmax_batch (PowerMean)")

        max_logits = np.max(logits, axis=1, keepdims=True)
        e_x = np.exp(logits - max_logits)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        
    def _value_batch(self, logits):
        if logits.ndim > 1 and logits.shape[1] > 1:
            min_v, max_v, step = self.support_range
            support = np.arange(min_v, max_v, step) 
            probs = self._softmax_batch(logits)
            return np.sum(probs * support, axis=1)
        return logits.flatten()
        
    def _scalar_value(self, logits):
        if logits.ndim > 1 and logits.shape[1] > 1:
            min_v, max_v, step = self.support_range
            support = torch.arange(min_v, max_v, step, device=logits.device, dtype=torch.float32)
            probs = torch.softmax(logits, dim=1)
            return torch.sum(probs * support, dim=1)
        return logits.squeeze()

    def get_action_probs(self, root, temperature=1.0):
        visit_counts = {}
        for action, (q_node, child) in root.children.items():
            visit_counts[action] = child.visit_count
        
        actions = list(visit_counts.keys())
        if not actions:
            return np.zeros(0) # Should verify action space size
            
        max_action = max(actions)
        dense_probs = np.zeros(max_action + 1)
        
        sum_visits = sum(visit_counts.values())
        if sum_visits == 0:
            return np.ones(len(dense_probs)) / len(dense_probs)
            
        for a, count in visit_counts.items():
            dense_probs[a] = count
            
        if temperature == 0:
            best_a = np.argmax(dense_probs)
            probs = np.zeros_like(dense_probs)
            probs[best_a] = 1.0
            return probs
            
        dense_probs = dense_probs ** (1/temperature)
        return dense_probs / np.sum(dense_probs)
        