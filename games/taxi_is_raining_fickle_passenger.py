import numpy as np
import gymnasium as gym
import copy

class Taxi_Is_Raining_Fickle_Passenger:
    def __init__(self, reward_scale=1.0, p_rain=0.1, p_fickle=0.3):
        self.name = "Taxi_Is_Raining_Fickle_Passenger"
        self.num_player = 1 
        self.env_id = "Taxi-v3"
        
        self.tensor_shape = (4, 5, 5) # Taxi, Passenger, Destination, InTaxiStatus
        
        self.action_size = 6 # down, up, right, left, pick up, drop off
        
        self.num_planes = self.tensor_shape[0]
        self.row_count = self.tensor_shape[1]
        self.column_count = self.tensor_shape[2]        
        
        self.is_stochastic = True
        self.p_rain = p_rain     # Prob of sliding left/right (0.1 each -> 0.8 success)
        self.p_fickle = p_fickle # Prob of changing dest on pickup
        self.reward_scale = reward_scale
        
        # Taxi locations for encoding
        self.locs = [(0,0), (0,4), (4,0), (4,3)]

    def get_initial_state(self):
        env = gym.make(self.env_id, render_mode=None)
        env.reset()
        return env

    def get_current_player(self, state):
        return 0

    def get_next_state(self, state, action):
        next_env = copy.deepcopy(state)
        
        actual_action = action
        
        # Raining Logic (Stochastic Movement)
        # Actions: 0:South, 1:North, 2:East, 3:West
        if action < 4: 
            if np.random.rand() < (2 * self.p_rain):
                # 0.2 chance to slide (0.1 left, 0.1 right)
                perpendiculars = {
                    0: [2, 3],
                    1: [2, 3],
                    2: [0, 1],
                    3: [0, 1]
                }
                actual_action = np.random.choice(perpendiculars[action])
        
        # Check pre-step state for Fickle logic
        pre_taxi_row, pre_taxi_col, pre_pass_idx, pre_dest_idx = next_env.unwrapped.decode(next_env.unwrapped.s)
        
        obs, reward, terminated, truncated, info = next_env.step(actual_action)
        
        # Fickle Passenger Logic
        post_taxi_row, post_taxi_col, post_pass_idx, post_dest_idx = next_env.unwrapped.decode(next_env.unwrapped.s)
        
        if pre_pass_idx != 4 and post_pass_idx == 4:
            # Passenger just entered taxi. Trigger Fickle change.
            if np.random.rand() < self.p_fickle:
                # Change destination to one of the other 3 locations
                possible_dests = [0, 1, 2, 3]
                if pre_dest_idx in possible_dests:
                    possible_dests.remove(pre_dest_idx)
                
                new_dest_idx = np.random.choice(possible_dests)
                
                # Update state
                new_s = next_env.unwrapped.encode(post_taxi_row, post_taxi_col, post_pass_idx, new_dest_idx)
                next_env.unwrapped.s = new_s
                
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
        s = env.unwrapped.s
        taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(s)
        
        encoding = np.zeros((4, 5, 5), dtype=np.float32)
        
        # Plane 0: Taxi Location
        encoding[0, taxi_row, taxi_col] = 1.0
        
        # Plane 1: Passenger Location
        if pass_idx < 4:
            r, c = self.locs[pass_idx]
            encoding[1, r, c] = 1.0
        else:
            # In taxi
            encoding[1, taxi_row, taxi_col] = 1.0
            
        # Plane 2: Destination Location
        dest_r, dest_c = self.locs[dest_idx]
        encoding[2, dest_r, dest_c] = 1.0
        
        # Plane 3: In Taxi Status (Global 1 if in taxi, else 0)
        if pass_idx == 4:
            encoding[3, :, :] = 1.0
            
        return encoding

    def render(self, state):
        # Create a temp env just for rendering to use rgb_array without tainting the main env
        env = gym.make(self.env_id, render_mode="rgb_array")
        env.reset()
        env.unwrapped.s = state.unwrapped.s
        frame = env.render()
        env.close()
        return frame
