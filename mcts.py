import numpy as np
import math
import torch


class Stochastic_PowerMean_UCT_Qnode:
    """
    Q-node (action-value node) in the MCTS tree.
    Represents a state-action pair and stores the Q-value for that action.
    """
    def __init__(self, game, args, model, action, curr_v_node, next_v_node, gamma):
        self.game = game
        self.args = args
        self.model = model
        self.q_value = 0
        self.visit_count = 0
        self.action = action
        self.curr_v_node = curr_v_node
        self.next_v_node = next_v_node
        self.gamma = gamma

    def update(self, reward):
        """
        Update Q-value using incremental average with discounted next state value.
        Formula: Q = (Q*N + r + gamma*V_next) / (N+1)
        """
        reward_scaled = (reward + 1) / 2
        numerator = self.q_value * self.visit_count + reward_scaled + self.gamma * self.next_v_node.v_value
        denominator = self.visit_count + 1
        self.q_value = numerator / denominator
        self.visit_count += 1


class Stochastic_PowerMean_UCT_Vnode:
    """
    V-node (state-value node) in the MCTS tree.
    Represents a game state and stores the V-value computed using power mean of children Q-values.
    """
    def __init__(self, game, args, model, gamma, p, num_rollout, num_workers, exploration_fn, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.model = model
        self.parent = parent
        self.gamma = gamma
        self.p = p
        self.num_rollout = num_rollout
        self.num_workers = num_workers
        self.exploration_fn = exploration_fn
        self.children = []
        self.visit_count = 0
        self.v_value = 0
        self.state_cache = {}

    def _get_policy_and_value(self, state):
        """
        Get policy and value from neural network model with caching.
        Applies valid move masking and normalization to policy.
        """
        state_key = state.tobytes()
        if state_key not in self.state_cache:
            value, policy = self.model.forward(state)
            valid_moves = self.game.get_valid_moves(state)
            policy = policy * valid_moves
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = np.zeros(self.game.action_size)
            self.state_cache[state_key] = {
                "value": value,
                "policy": policy
            }
        return self.state_cache[state_key]["value"], self.state_cache[state_key]["policy"]
    
    def is_fully_expanded(self, state):
        """
        Check if all valid moves from this state have been expanded.
        Returns True if there are no unexpanded moves remaining.
        """
        valid_moves = self.game.get_valid_moves(state)
        expandable_moves = set(np.where(valid_moves == 1)[0])
        expanded_moves = set([child.action for child in self.children])
        self.unexpanded_moves = [move for move in expandable_moves if move not in expanded_moves]
        return len(self.unexpanded_moves) == 0 and len(self.children) > 0
    
    def select(self, state):
        """
        Select best child using UCB (Upper Confidence Bound) formula.
        Returns the action, next V-node, and resulting state.
        """
        state_key = state.tobytes()
        if state_key not in self.state_cache:
            self._get_policy_and_value(state)
        valid_moves = set(np.where(self.game.get_valid_moves(state) == 1)[0])
        valid_children = [child for child in self.children if child.action in valid_moves]
        best_child = None
        best_ucb = -np.inf
        policy = self.state_cache[state_key]["policy"]
        for child in valid_children:
            prior_p = policy[child.action]
            ucb = self.get_ucb(child, prior_p, state, child.action)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        next_state = self.game.get_next_state(state, best_child.action, 1)
        return best_child.action, best_child.next_v_node, next_state
    
    def get_ucb(self, child, prior_p, state, action):
        """
        Compute UCB (Upper Confidence Bound) value for a child node.
        Formula: UCB = Q + prior_p * exploration_fn(parent_visits, child_visits, C)
        """
        q_value = child.q_value
        parent_visits = self.visit_count
        child_visits = child.visit_count
        C = self.args['C']
        exploration_base = self.exploration_fn(parent_visits, child_visits, C)
        exploration = prior_p * exploration_base
        ucb = q_value + exploration
        return ucb
    
    def expand(self, state, opponent_node):
        """
        Expand the tree by adding a new child Q-node for an unexpanded action.
        Estimates the value by averaging neural network evaluations of all possible
        next moves from the child state.
        Returns the estimated value, new V-node, and resulting state.
        """
        state_key = state.tobytes()
        if state_key not in self.state_cache:
            self._get_policy_and_value(state)
        action = np.random.choice(self.unexpanded_moves)
        next_v_node = Stochastic_PowerMean_UCT_Vnode(
            self.game, self.args, self.model, gamma=self.gamma, p=self.p,
            num_rollout=self.num_rollout, num_workers=self.num_workers,
            exploration_fn=self.exploration_fn, parent=None
        )
        child_state = self.game.get_next_state(state, action, 1)
        child_valid_moves = self.game.get_valid_moves(child_state)
        child_expandable_moves = set(np.where(child_valid_moves == 1)[0])
        child_expanded_moves = set([child.action for child in next_v_node.children])
        values = []
        unexpanded_child_moves = child_expandable_moves - child_expanded_moves
        for move in unexpanded_child_moves:
            move_state = child_state.copy()
            move_state = self.game.get_next_state(move_state, move, -1)
            value, policy = next_v_node._get_policy_and_value(move_state)
            values.append(value)
        if len(values) == 0:
            value, is_terminal = self.game.get_value_and_terminated(child_state, action)
            avg_value = value
        else:
            avg_value = np.mean(values)
        next_v_node.v_value = (avg_value + 1) / 2
        child = Stochastic_PowerMean_UCT_Qnode(
            self.game, self.args, self.model, action=action,
            curr_v_node=self, next_v_node=next_v_node, gamma=self.gamma
        )
        next_v_node.parent = child
        self.children.append(child)
        return avg_value, next_v_node, child_state
    
    def backpropagate(self, value, opponent_node=None):
        """
        Backpropagate value up the tree, updating V-values using power mean aggregation.
        For two-player games, alternates between players via opponent_node.
        """
        self.visit_count += 1
        if len(self.children) > 0:
            power_sum = 0
            for child in self.children:
                weight = child.visit_count / self.visit_count
                powered = child.q_value ** self.p
                contribution = weight * powered
                power_sum += contribution
            self.v_value = power_sum ** (1.0 / self.p)
        if self.parent is not None:
            self.parent.update(value)
            if opponent_node is not None:
                flipped_value = self.game.get_opponent_value(value)
                next_opponent = self.parent.curr_v_node
                opponent_node.backpropagate(flipped_value, next_opponent)
            else:
                self.parent.curr_v_node.backpropagate(value)
        else:
            if opponent_node is not None:
                flipped_value = self.game.get_opponent_value(value)
                opponent_node.backpropagate(flipped_value, None)


class Stochastic_PowerMean_UCT:
    """
    Stochastic Power Mean UCT implementation for MCTS.
    Only support two-player games.
    """
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.gamma = args['gamma']
        self.p = args['p']
        self.num_rollout = args['num_rollout']
        self.num_workers = args['num_workers']
        self.exploration_fn = args['exploration_fn']
    
    @torch.no_grad()
    def search(self, state):
        """
        Perform MCTS search from the given state.
        Returns action probability distribution based on visit counts.
        """
        root_player1 = Stochastic_PowerMean_UCT_Vnode(
            self.game, self.args, self.model, gamma=self.gamma, p=self.p,
            num_rollout=self.num_rollout, num_workers=self.num_workers,
            exploration_fn=self.exploration_fn
        )
        root_player2 = Stochastic_PowerMean_UCT_Vnode(
            self.game, self.args, self.model, gamma=self.gamma, p=self.p,
            num_rollout=self.num_rollout, num_workers=self.num_workers,
            exploration_fn=self.exploration_fn
        )
        for search in range(self.args['num_searches']):
            action_selected = None
            curr_player = 1
            node = root_player1
            node_state = state.copy()
            opponent_node = None
            while node.is_fully_expanded(node_state):
                action_selected, selected_next_node, node_state = node.select(node_state)
                value, is_terminal = self.game.get_value_and_terminated(node_state, action_selected)
                if is_terminal:
                    node = selected_next_node
                    break
                if opponent_node is None:
                    opponent_node = selected_next_node
                    node = root_player2
                else:
                    node = opponent_node
                    opponent_node = selected_next_node
                node_state = self.game.change_perspective(node_state, -1)
                curr_player = -curr_player
            value, is_terminal = self.game.get_value_and_terminated(node_state, action_selected)
            if not is_terminal:
                value, node, node_state = node.expand(node_state, opponent_node)
            node.backpropagate(value, opponent_node)
        action_probs = np.zeros(self.game.action_size)
        for child in root_player1.children:
            action_probs[child.action] = child.visit_count
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs /= total_visits
        return action_probs
