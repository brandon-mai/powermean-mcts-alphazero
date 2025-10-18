import numpy as np
import math
import torch

class Node:
    def __init__(self, game, state, player, C, p, gamma, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.state = state
        self.player = player
        self.player_idx = 0 if player == 1 else 1
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.C = C
        self.p = p
        self.gamma = gamma
        
        self.children = [] # for compatibility purpose. Actually, this should be named opponent_nodes
        self.visit_count = visit_count
        self.q_node_values = np.array([0, 0], dtype=np.float32)
        self.v_node_values = np.array([0, 0], dtype=np.float32)

    def is_fully_expanded(self):
        return len(self.children) > 0 
    
    def select_opponent(self):
        best_node = None
        best_ucb = -np.inf
        
        for node in self.children:
            ucb = self.get_ucb(node)
            if ucb > best_ucb:
                best_node = node
                best_ucb = ucb
        return best_node
    
    def get_ucb(self, node):
        q_value = node.q_node_values[self.player_idx]
        return q_value + self.C * (math.pow(self.visit_count, 0.25) / math.sqrt(node.visit_count + 1)) * node.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                node_state = self.state.copy()
                node_state = self.game.get_next_state(node_state, action)
                node_state = self.game.change_perspective(node_state, player=-1)

                node = Node(
                    game=self.game,
                    state=node_state,
                    player=self.game.get_opponent(self.player),
                    C=self.C,
                    p=self.p,
                    gamma=self.gamma,
                    parent=self,
                    action_taken=action,
                    prior=prob
                )
                self.children.append(node)
        return node
    
    def backpropagate(self, update_player, final_reward=None, immediate_reward=0):
        if update_player != self.player and self.parent:
            self.parent.backpropagate(update_player)
            return

        if final_reward is not None:
            self.v_node_values[self.player_idx] = final_reward
        else:
            children = [child for node in self.children for child in node.children if child.visit_count > 0]
            
            power_sum = 0
            if len(children) == 0:
                for node in self.children:
                    weight = node.visit_count / (self.visit_count + 1)
                    powered = ((1 + self.gamma) * node.v_node_values[node.player_idx]) ** self.p
                    contribution = weight * powered
                    power_sum += contribution                    
            else:            
                for child in children:
                    weight = child.visit_count / (self.visit_count + 1)
                    powered = child.q_node_values[self.player_idx] ** self.p
                    contribution = weight * powered
                    power_sum += contribution
            self.v_node_values[self.player_idx] = power_sum ** (1.0 / self.p)

        if self.parent:
            if final_reward is not None:
                self.q_node_values[self.player_idx] = self.q_node_values[self.player_idx] + final_reward
            else:
                self.q_node_values[self.player_idx] = (
                    self.q_node_values[self.player_idx] * self.visit_count
                    + immediate_reward
                    + self.gamma * self.v_node_values[self.player_idx]
                ) / (self.visit_count + 1)
        self.visit_count += 1
        
        if self.parent:
            self.parent.backpropagate(update_player)

class Stochastic_Powermean_UCT:
    def __init__(self, game, model, C=1.41, p=1.5, gamma=0.95,
                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3, num_searches=25):
        self.name = "Stochastic_Powermean_UCT"
        self.game = game
        self.model = model

        self.C = C
        self.p = p
        self.gamma = gamma
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.num_searches = num_searches

    @torch.no_grad()
    def search(self, states, spGames):    
        policies, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
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
                state=states[i], 
                player=self.game.get_current_player(states[i]),
                C=self.C,
                p=self.p,
                gamma=self.gamma,
                visit_count=1
            )
            spg.root.expand(spg_policy)
        
        for search in range(self.num_searches):
            for spg in spGames:
                spg.node = None
                node = spg.root
                

                while node.is_fully_expanded():
                    node = node.select_opponent()

                value, is_terminal = self.game.get_value_and_terminated(
                    state=node.state, 
                    player=self.game.get_current_player(node.state)
                )
                
                if is_terminal:
                  node.backpropagate(update_player=node.player, final_reward=value)
                  node.parent.backpropagate(update_player=node.parent.player, final_reward=self.game.get_opponent_value(value))
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [i for i in range(len(spGames)) if spGames[i].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[i].node.state for i in expandable_spGames])
                policies, values = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
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
                node.backpropagate(update_player=node.player, final_reward=spg_value)  

                node.parent.backpropagate(update_player=node.parent.player, final_reward=self.game.get_opponent_value(spg_value))