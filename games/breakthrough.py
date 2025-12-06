import numpy as np
import pyspiel

class Breakthrough():
    def __init__(self):
        self.name = "Breakthrough"
        self.num_player = 2
        self.game = pyspiel.load_game("breakthrough")
        self.action_size = self.game.num_distinct_actions()

        self.tensor_shape = tuple(self.game.observation_tensor_shape())
        
        self.num_planes = self.tensor_shape[0]
        self.row_count = self.tensor_shape[1]
        self.column_count = self.tensor_shape[2]        
        
        self.is_stochastic = False

    def get_initial_state(self):
        return self.game.new_initial_state()

    def get_current_player(self, state):
        return state.current_player()

    def get_next_state(self, state, action):
        next_state = state.clone()
        next_state.apply_action(action)  
        return next_state

    def get_valid_moves(self, state):
        return state.legal_actions()

    def get_value_and_terminated(self, state, player):
        if state.is_terminal():
            return (state.returns()[player] + 1) / 2, True
        return 0, False

    def get_opponent(self, player):
        return 1 - player 

    def get_encoded_state(self, state):
        if isinstance(state, (list, tuple)):
            batch_size = len(state)
            encoded = np.zeros((batch_size,) + self.tensor_shape, dtype=np.float32)
            for i, s in enumerate(state):
                encoded[i] = np.reshape(s.observation_tensor(), self.tensor_shape)
            return encoded
        else:
            return np.reshape(state.observation_tensor(), self.tensor_shape).astype(np.float32)

    def render(self, state):
        print(f"state:\n{state}")
        print(f"current_player: {state.current_player()}")
        print(f"legal_actions: {state.legal_actions()}")
