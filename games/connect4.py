import numpy as np
import pyspiel

import copy

class ConnectFour():
    def __init__(self):
        self.name = "ConnectFour"
        self.num_player = 2

        self.game = pyspiel.load_game("connect_four")
        self.action_size = self.game.num_distinct_actions()
        
    def get_initial_state(self):
        state = self.game.new_initial_state()
        return copy.deepcopy(state)

    def get_current_player(self, state):
        player = state.current_player()
        return player

    def get_next_state(self, state, action):
        next_state = copy.deepcopy(state)
        next_state.apply_action(action)  
        return next_state

    def get_valid_moves(self, state):
        return state.legal_actions()

    def get_value_and_terminated(self, state, player):
        is_terminal = state.is_terminal()  
        if is_terminal:
            reward = state.returns()[player]
        else:
            # get intermediate reward
            reward = state.rewards()[player]
        # re-scale to [0, 1]
        reward = (reward + 1) / 2
        return reward, is_terminal 

    def get_opponent(self, player):
        if player == 0:
            opponent_player = 1
        elif player == 1:
            opponent_player = 0
        return opponent_player

    def get_opponent_value(self, value):
        return 1.0 - value

    def get_encoded_state(self, state):
        shape = self.game.observation_tensor_shape()

        if isinstance(state, list):  
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
