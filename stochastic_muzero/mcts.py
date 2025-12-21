import numpy as np
import torch
import math
import collections

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0
        
    def expanded(self):
        return len(self.children) > 0
        
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class DecisionNode(Node):
    def __init__(self, prior):
        super().__init__(prior)
        self.to_play = 0

class ChanceNode(Node):
    def __init__(self, prior):
        super().__init__(prior)


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
    def __init__(self, game, model, num_searches, c_puct=1.25, 
                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3,
                 discount=0.997, use_chance_nodes=False):
        self.game = game
        self.model = model
        self.num_searches = num_searches
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.discount = discount
        self.use_chance_nodes = use_chance_nodes
        
    def search(self, current_states, spgs):
        min_max_stats = MinMaxStats()
        
        # 1. Initial Inference (Root)
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
        
        # 2. Initialize Roots
        for i, spg in enumerate(spgs):
            root = DecisionNode(1.0)
            root.hidden_state = root_hidden[i]
            root.to_play = self.game.get_current_player(current_states[i])
            
            policy = root_policy[i]
            if self.dirichlet_epsilon > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
                policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
                
            for action_idx, prob in enumerate(policy):
                if self.use_chance_nodes:
                    child = ChanceNode(prob)
                else:
                    child = DecisionNode(prob)
                root.children[action_idx] = child
                
            spg.root = root
            self._backpropagate([root], root_value[i], 0.0, min_max_stats)

        # 3. Simulations
        for _ in range(self.num_searches):
            for i, spg in enumerate(spgs):
                self._run_simulation(spg, min_max_stats)
                
    def _run_simulation(self, spg, min_max_stats):
        node = spg.root
        search_path = [node]
        
        # --- Selection ---
        while node.expanded():
            if isinstance(node, DecisionNode):
                action, node = self._select_action(node, min_max_stats)
            elif isinstance(node, ChanceNode):
                outcome, node = self._select_chance(node, min_max_stats)
            else:
                break
            search_path.append(node)
            
        # --- Expansion & Evaluation ---
        parent = search_path[-2]
        action_or_outcome = None
        for act, child in parent.children.items():
            if child is node:
                action_or_outcome = act
                break
                
        with torch.no_grad():
            if isinstance(parent, DecisionNode):
                # Transition: Decision -> Chance (Afterstate)
                hidden_tensor = torch.tensor(parent.hidden_state, device=self.model.device).unsqueeze(0)
                action_tensor = torch.tensor([action_or_outcome], device=self.model.device)
                
                next_hidden, reward_logits = self.model.dynamics(hidden_tensor, action_tensor)
                
                reward = self._scalar_value(reward_logits).item()
                next_hidden = next_hidden.cpu().numpy()[0]
                
                node.hidden_state = next_hidden
                node.reward = reward
                
                if self.use_chance_nodes:
                    # Expand Chance Outcomes
                    chance_logits, value_logits = self.model.afterstate_prediction(self._tensor(next_hidden))
                    
                    chance_probs = self._softmax_batch(chance_logits.cpu().numpy())[0]
                    value = self._value_batch(value_logits.cpu().numpy())[0]
                    
                    for outcome_idx, prob in enumerate(chance_probs):
                        child = DecisionNode(prob) # Child of Chance is Decision
                        node.children[outcome_idx] = child
                else:
                    # Deterministic case logic...
                    pass 

            elif isinstance(parent, ChanceNode):
                # Transition: Chance -> Decision (Next State)
                hidden_tensor = torch.tensor(parent.hidden_state, device=self.model.device).unsqueeze(0)
                
                chance_space = self.model.chance_space_size
                chance_onehot = torch.zeros((1, chance_space), device=self.model.device)
                chance_onehot[0, action_or_outcome] = 1.0
                
                # afterstate dynamics
                next_hidden, _ = self.model.afterstate_dynamics(hidden_tensor, chance_onehot)
                
                reward = 0.0 
                next_hidden = next_hidden.cpu().numpy()[0]
                
                node.hidden_state = next_hidden
                node.reward = reward
                
                policy_logits, value_logits = self.model.prediction(self._tensor(next_hidden))
                
                policy_probs = self._softmax_batch(policy_logits.cpu().numpy())[0]
                value = self._value_batch(value_logits.cpu().numpy())[0]
                
                for action_idx, prob in enumerate(policy_probs):
                    if self.use_chance_nodes:
                        child = ChanceNode(prob) 
                    else:
                        child = DecisionNode(prob)
                    
        # --- Backup ---
        self._backpropagate(search_path, value, 0.0, min_max_stats)
        
    def _select_action(self, node, min_max_stats):
        max_score = -float('inf')
        best_child = None
        best_action = -1
        
        for action, child in node.children.items():
            if min_max_stats.maximum == min_max_stats.minimum:
                val_score = child.value()
            else:
                val_score = min_max_stats.normalize(child.value())
                
            pb_c = math.log((node.visit_count + 19652 + 1) / 19652) + self.c_puct
            pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
            
            prior_score = child.prior * pb_c
            score = val_score + prior_score
            
            if score > max_score:
                max_score = score
                best_child = child
                best_action = action
                
        return best_action, best_child
        
    def _select_chance(self, node, min_max_stats):
        return self._select_action(node, min_max_stats)

    def _backpropagate(self, search_path, value, mix_reward, min_max_stats):
        current_value = value
        for node in reversed(search_path):
            node.value_sum += current_value
            node.visit_count += 1
            min_max_stats.update(node.value())
            current_value = node.reward + self.discount * current_value
            
    def _tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.unsqueeze(0)
        return torch.tensor(arr, device=self.model.device).unsqueeze(0)
    
    def _softmax_batch(self, logits):
        if logits.ndim == 1:
            logits = logits[None, :]
        max_logits = np.max(logits, axis=1, keepdims=True)
        e_x = np.exp(logits - max_logits)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        
    def _value_batch(self, logits):
        if logits.ndim > 1 and logits.shape[1] > 1:
            min_v, max_v = -300, 301
            support = np.arange(min_v, max_v, 1)
            probs = self._softmax_batch(logits)
            return np.sum(probs * support, axis=1)
        return logits.flatten()
        
    def _scalar_value(self, logits):
        if logits.ndim > 1 and logits.shape[1] > 1:
            min_v, max_v = -300, 301
            support = torch.arange(min_v, max_v, 1, device=logits.device, dtype=torch.float32)
            probs = torch.softmax(logits, dim=1)
            return torch.sum(probs * support, dim=1)
        return logits.squeeze()

    def get_action_probs(self, root, temperature=1.0):
        visit_counts = {}
        for action, child in root.children.items():
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
