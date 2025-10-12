import numpy as np
import math
import torch


class Node_Global:
    """Global: Both players share the same Q and V values, playout value is negated for the opponent."""
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.v_node_value = 0
        self.q_node_value = 0
        
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
        q_value = child.q_node_value
        return q_value + self.args['C'] * (math.pow(self.visit_count, 0.25) / math.sqrt(child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node_Global(game=self.game, args=self.args, state=child_state,
                                    parent=self, action_taken=action, prior=prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value=None, immediate_reward=0):
        if value:
            value = (value + 1) / 2
            self.v_node_value = value
        else:
            power_sum = 0
            for child in self.children:
                weight = child.visit_count / (self.visit_count + 1)
                powered = min(child.q_node_value, 1) ** self.args['p'] # Q node value can exceed 1 due to immediate reward, so we clip
                contribution = weight * powered
                power_sum += contribution
            self.v_node_value = power_sum ** (1.0 / self.args['p'])
        
        if self.parent:
            flipped_value = 1 - self.v_node_value
            self.q_node_value = (self.q_node_value * self.visit_count + immediate_reward + self.args['gamma'] * flipped_value) / (self.visit_count + 1)
        
        self.visit_count += 1
        
        if self.parent:
            self.parent.backpropagate()  
 

class MCTS_Global_Parallel:
    """Global: Both players share the same Q and V values, playout value is negated for the opponent."""
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node_Global(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                # Child node (B's turn)
                while node.is_fully_expanded():
                    node = node.select()

                # If win (B's POV), MUST NEGATE because reward is used for A's Q node update.
                value, is_terminal = self.game.get_value_and_terminated(node.state, self.game.get_current_player(node.state))
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(immediate_reward=value)

                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)
