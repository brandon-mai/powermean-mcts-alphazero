import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
import copy

class Stochastic_MiniGrid:
    def __init__(self, env_id="MiniGrid-Empty-8x8-v0", reward_scale=10.0):
        self.name = "Stochastic_MiniGrid"
        self.num_player = 1 
        
        self.env_id = env_id
        self._dummy_env = gym.make(env_id, render_mode="rgb_array")
        self._dummy_env = ImgObsWrapper(self._dummy_env) 
        
        self.action_size = self._dummy_env.action_space.n 
        
        obs_shape = self._dummy_env.observation_space.shape
        self.tensor_shape = (obs_shape[2], obs_shape[0], obs_shape[1]) 
        
        self.num_planes = self.tensor_shape[0]
        self.row_count = self.tensor_shape[1]
        self.column_count = self.tensor_shape[2]        
        
        self.is_stochastic = True
        self.randomness = 0.25 
        self.reward_scale = reward_scale

    def get_initial_state(self):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env.reset(seed=42) 
        return env

    def get_current_player(self, state):
        return 0

    def get_next_state(self, state, action):
        next_env = copy.deepcopy(state)
        
        actual_action = action
        
        if np.random.rand() < self.randomness:
            legal_moves = range(self.action_size)
            actual_action = np.random.choice(legal_moves)
            
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
        if isinstance(state, (list, tuple, np.ndarray)) and len(state) > 0 and not hasattr(state[0], 'step'):
             pass

        if isinstance(state, list):
            encoded_batch = []
            for s in state:
                obs = s.gen_obs() 
                img = obs['image'] 
                img_transposed = np.transpose(img, (2, 0, 1))
                encoded_batch.append(img_transposed / 10.0)
                
            return np.array(encoded_batch, dtype=np.float32)
        else:
            obs = state.gen_obs()
            img = obs['image']
            img_transposed = np.transpose(img, (2, 0, 1))
            return (img_transposed / 10.0).astype(np.float32)

    def render(self, state):
        print(f"MiniGrid Env: {self.env_id}")
