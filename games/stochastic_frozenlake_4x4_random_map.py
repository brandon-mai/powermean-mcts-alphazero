import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import copy

class Stochastic_FrozenLake_4x4_Random_Map:
    def __init__(self, reward_scale=10.0):
        self.name = "Stochastic_FrozenLake_4x4_Random_Map"
        self.num_player = 1 
        self.env_id = "FrozenLake-v1"
        self.size = 4
        
        self.tensor_shape = (3, self.size, self.size) # Agent, Holes, Goal
        
        self.action_size = 4 # Left, Down, Right, Up
        
        self.num_planes = self.tensor_shape[0]
        self.row_count = self.tensor_shape[1]
        self.column_count = self.tensor_shape[2]        
        
        self.is_stochastic = True
        self.randomness = 0.5
        self.reward_scale = reward_scale
        print(f"game: {self.env_id}")

    def get_initial_state(self):
        random_map = generate_random_map(size=self.size)
        env = gym.make(
            self.env_id, 
            desc=random_map, 
            is_slippery=self.is_stochastic,
            success_rate=self.randomness,
            render_mode=None)
        env.reset()
        return env

    def get_current_player(self, state):
        return 0

    def get_next_state(self, state, action):
        next_env = copy.deepcopy(state)
        
        actual_action = action
        
        obs, reward, terminated, truncated, info = next_env.step(actual_action)
        reward = reward * self.reward_scale
        
        next_env.custom_reward = reward
        next_env.custom_done = terminated or truncated
        
        return next_env

    def get_valid_moves(self, state):
        return list(range(self.action_size))

    def get_value_and_terminated(self, state, player):
        if hasattr(state, 'custom_done'):
            return state.custom_reward, state.custom_done
        return 0, False

    def get_opponent(self, player):
        return 0 

    def get_opponent_value(self, value):
        return value

    def get_encoded_state(self, state):
        if isinstance(state, list):
            encoded_batch = []
            for s in state:
                encoded_batch.append(self._encode_single_env(s))
            return np.array(encoded_batch, dtype=np.float32)
        else:
            return self._encode_single_env(state)
            
    def _encode_single_env(self, env):
        current_state_idx = env.unwrapped.s
        desc = env.unwrapped.desc
        nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol
        
        row = current_state_idx // ncol
        col = current_state_idx % ncol
        
        # Planes: 3
        # 0: Agent
        # 1: Holes
        # 2: Goal
        
        encoding = np.zeros((3, nrow, ncol), dtype=np.float32)
        
        # Agent
        encoding[0, row, col] = 1.0
        
        # Map features
        for r in range(nrow):
            for c in range(ncol):
                char = desc[r, c] # b'S', b'F', b'H', b'G'
                if char == b'H':
                    encoding[1, r, c] = 1.0
                elif char == b'G':
                    encoding[2, r, c] = 1.0
                    
        return encoding

    def render(self, state):
        env = gym.make(
             self.env_id, 
             desc=state.unwrapped.desc, 
             is_slippery=True,
             success_rate=self.randomness,
             render_mode="rgb_array")
        env.reset()
        env.unwrapped.s = state.unwrapped.s
        frame = env.render()
        env.close()
        return frame
