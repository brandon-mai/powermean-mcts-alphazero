import numpy as np
import torch
import math

class Node:
    def __init__(self, game, C, dirichlet_epsilon, dirichlet_alpha, num_searches, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.C = C
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.num_searches = num_searches
        
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child_state = self.game.change_perspective(
                    state=child_state, 
                    player=-1
                )

                child = Node(
                    game=self.game, 
                    C=self.C, 
                    dirichlet_epsilon=self.dirichlet_epsilon, 
                    dirichlet_alpha=self.dirichlet_alpha, 
                    num_searches=self.num_searches,
                    state=child_state, 
                    parent=self, 
                    action_taken=action, 
                    prior=prob
                )
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class PUCT:
    def __init__(self, game, model, C, dirichlet_epsilon, dirichlet_alpha, num_searches):
        self.name = "PUCT"
        self.game = game
        self.model = model
        
        self.C = C
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.num_searches = num_searches

    @torch.no_grad()
    def search(self, states, spGames):
        policies, _ = self.model(
            states=states
        )
        policies = torch.softmax(policies, axis=1).cpu().numpy()
        policies = (1 - self.dirichlet_epsilon) * policies + self.dirichlet_epsilon * np.random.dirichlet(
            [self.dirichlet_alpha] * self.game.action_size, size=policies.shape[0]
        )
        
        for i, spg in enumerate(spGames):
            spg_policy = policies[i]

            valid_moves = self.game.get_valid_moves(states[i])
            # create mask for valid moves
            valid_moves = np.array([1 if j in valid_moves else 0 for j in range(self.game.action_size)])

            spg_policy *= valid_moves
            if np.sum(spg_policy) == 0:
                spg_policy = valid_moves / np.sum(valid_moves)
            else:
                spg_policy /= np.sum(spg_policy)

            spg.root = Node(
                game=self.game, 
                C=self.C, 
                dirichlet_epsilon=self.dirichlet_epsilon, 
                dirichlet_alpha=self.dirichlet_alpha, 
                num_searches=self.num_searches,
                state=states[i], 
                visit_count=1
            )
            spg.root.expand(spg_policy)
        
        for search in range(self.num_searches):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.game.get_current_player(node.state))
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node
                    
            expandable_spGames = [idx for idx in range(len(spGames)) if spGames[idx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[i].node.state for i in expandable_spGames])
                
                policies, values = self.model(
                    states=states
                )
                
                policies = torch.softmax(policies, axis=1).cpu().numpy()
                values = values.cpu().numpy()
                
            for i, idx in enumerate(expandable_spGames):
                node = spGames[idx].node
                spg_policy, spg_value = policies[i], values[i]

                spg_value = (spg_value + 1) / 2  # Normalize value to [0, 1]
                
                valid_moves = self.game.get_valid_moves(states[i])
                # create mask for valid moves
                valid_moves = np.array([1 if j in valid_moves else 0 for j in range(self.game.action_size)])
                
                spg_policy *= valid_moves
                if np.sum(spg_policy) == 0:
                    spg_policy = valid_moves / np.sum(valid_moves)
                else:
                    spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(value=spg_value)  
