import numpy as np
import math
import torch


class Node_Local:
    """Local: Separately tracks Q and V values for both players, no playout value is negated.
    Args:
        game: The game object.
        args: The arguments for MCTS.
        state: The state of the game at this node.
        player: The player making the move: 1 for player, -1 for opponent.
        parent: The parent node.
        action_taken: The action taken to reach this node.
        prior: The prior probability of selecting this node.
        visit_count: The number of times this node has been visited.
    """
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.player_idx = 0 if player == 1 else 1
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.q_node_values = np.array([0, 0], dtype=np.float32)
        self.v_node_values = np.array([0, 0], dtype=np.float32)

    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        q_value = child.q_node_values[self.player_idx]
        return q_value + self.args['C'] * (math.pow(self.visit_count, 0.25) / math.sqrt(child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node_Local(game=self.game, args=self.args, state=child_state, player=-self.player,
                             parent=self, action_taken=action, prior=prob)
                self.children.append(child)
                
        return child
    
    def backpropagate(self, update_player, value=None, immediate_reward=0):
        """Backpropagate the value up to the root node.
        Args:
            update_player: The player for whom the value is being updated/recalculated.
            value: Value of the V node just evaluated. If None, compute from children.
            immediate_reward: Immediate reward (endgame or game-specific) to be added to Q node value.
        """
        if update_player != self.player and self.parent:
            self.parent.backpropagate(update_player)
            return
        
        if value:
            value = (value + 1) / 2
            self.v_node_values[self.player_idx] = value
        else:
            power_sum = 0
            for child in self.children:
                weight = child.visit_count / (self.visit_count + 1)
                powered = child.q_node_values[self.player_idx] ** self.args['p']
                contribution = weight * powered
                power_sum += contribution
            self.v_node_values[self.player_idx] = power_sum ** (1.0 / self.args['p'])

        if self.parent:
            self.q_node_values[self.player_idx] = (self.q_node_values[self.player_idx] * self.visit_count + immediate_reward + self.args['gamma'] * self.v_node_values[self.player_idx]) / (self.visit_count + 1)

        self.visit_count += 1
        
        if self.parent:
            self.parent.q_node_values[self.player_idx] = self.q_node_values[self.player_idx]
            self.parent.v_node_values[self.player_idx] = self.v_node_values[self.player_idx]
            self.parent.backpropagate(update_player)  


class MCTS_Local_Parallel:
    """Local: Each player maximizes their own value and does not care of the opponents', no playout value is negated."""
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames):    
        policies, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policies = torch.softmax(policies, axis=1).cpu().numpy()
        policies = (1 - self.args['dirichlet_epsilon']) * policies + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policies.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policies[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            if np.sum(spg_policy) == 0:
                spg_policy = valid_moves / np.sum(valid_moves)
            else:
                spg_policy /= np.sum(spg_policy)

            spg.root = Node_Local(self.game, self.args, states[i], self.game.get_current_player(states[i]), visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                # Get child node (opponent's turn)
                while node.is_fully_expanded():
                    node = node.select()

                # Check terminal and if win FOR OPPONENT
                value, is_terminal = self.game.get_value_and_terminated(node.state, self.game.get_current_player(node.state))
                
                # Just backprop, no negation
                if is_terminal:
                    node.backpropagate(immediate_reward=value)
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policies, values = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policies = torch.softmax(policies, axis=1).cpu().numpy()
                values = values.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policies[i], values[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                if np.sum(spg_policy) == 0:
                    spg_policy = valid_moves / np.sum(valid_moves)
                else:
                    spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(value=spg_value)
