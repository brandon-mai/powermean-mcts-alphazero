import numpy as np
import pyspiel
import copy

class Stochastic_Havannah():
    def __init__(self):
        self.name = "Stochastic_Havannah"
        self.num_player = 2

        self.game = pyspiel.load_game("havannah")
        self.action_size = self.game.num_distinct_actions()
        self.is_stochastic = True
        
        self.randomness = 0.50  

    def get_initial_state(self):
        state = self.game.new_initial_state()
        return copy.deepcopy(state)

    def get_current_player(self, state):
        player = state.current_player()
        return player

    def get_next_state(self, state, action):
        next_state = copy.deepcopy(state)
        
        actual_action = action
        
        if np.random.rand() < self.randomness:
            legal_moves = state.legal_actions()
            
            if len(legal_moves) > 0:
                actual_action = np.random.choice(legal_moves)
        next_state.apply_action(actual_action)  
        return next_state
    
    def get_next_absolute_state(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.apply_action(action)  
        return next_state

    def get_valid_moves(self, state):
        return state.legal_actions()

    def get_value_and_terminated(self, state, player):
        is_terminal = state.is_terminal()  
        if is_terminal:
            reward = state.returns()[player]
            reward = (reward + 1) / 2
        else:
            # get intermediate reward
            # because there is no intermedate reward, so by default intermedate reward is 0
            reward =  0
        return reward, is_terminal 

    def get_opponent(self, player):
        if player == 0:
            opponent_player = 1
        elif player == 1:
            opponent_player = 0
        else:
            raise ValueError(f"Invalid player value: {player}. Shoule be  0 or 1.")
        return opponent_player
        
    def get_opponent_value(self, value):
        return 1.0 - value

    def get_encoded_state(self, state):
        shape = self.game.observation_tensor_shape()

        if isinstance(state, list) or isinstance(state, tuple):  
            encoded_state = []
            for s in state:
                encoded_state.append(np.reshape(np.asarray(s.observation_tensor()), shape))
            encoded_state = np.stack(encoded_state).astype(np.float32)    
        else:
            encoded_state = np.reshape(np.asarray(state.observation_tensor()), shape) 
        return encoded_state

    def render(self, state):
        print(f"state:\n{state}")
        print(f"current_player: {state.current_player()}")
        print(f"legal_actions: {state.legal_actions()}")