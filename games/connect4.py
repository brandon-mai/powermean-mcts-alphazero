import numpy as np
from games.abstract_game import AbstractGame

class ConnectFour(AbstractGame):
    def __init__(self):
        super().__init__(name="ConnectFour", num_player=2)
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_current_player(self, state):
        count1 = np.sum(state == 1)
        count2 = np.sum(state == -1)
        return 1 if count1 <= count2 else -1

    def get_next_state(self, state, action):
        player = self.get_current_player(state)
        row = np.max(np.where(state[:, action] == 0))
        new_state = state.copy()
        new_state[row, action] = player
        return new_state

    def get_valid_moves(self, state) -> list:
        return [i for i in range(self.column_count) if state[0, i] == 0]

    def check_win(self, state, player: int) -> str:
        # Check horizontal, vertical, and both diagonals
        for i in range(self.row_count):
            for j in range(self.column_count):
                # Check horizontal
                if j + self.in_a_row <= self.column_count:
                    if all(state[i][j+k] == player for k in range(self.in_a_row)):
                        return "win"
                    if all(state[i][j+k] == -player for k in range(self.in_a_row)):
                        return "lose"
                
                # Check vertical
                if i + self.in_a_row <= self.row_count:
                    if all(state[i+k][j] == player for k in range(self.in_a_row)):
                        return "win"
                    if all(state[i+k][j] == -player for k in range(self.in_a_row)):
                        return "lose"
                
                # Check diagonal (top-left to bottom-right)
                if i + self.in_a_row <= self.row_count and j + self.in_a_row <= self.column_count:
                    if all(state[i+k][j+k] == player for k in range(self.in_a_row)):
                        return "win"
                    if all(state[i+k][j+k] == -player for k in range(self.in_a_row)):
                        return "lose"
                
                # Check diagonal (top-right to bottom-left)
                if i + self.in_a_row <= self.row_count and j - self.in_a_row + 1 >= 0:
                    if all(state[i+k][j-k] == player for k in range(self.in_a_row)):
                        return "win"
                    if all(state[i+k][j-k] == -player for k in range(self.in_a_row)):
                        return "lose"
        
        # Check draw
        if np.all(state != 0):
            return "draw"
        return "not_ended"

    def get_value_and_terminated(self, state, player):
        result = self.check_win(state, player)
        if result == "win":
            reward = 1.0
        elif result == "lose":
            reward = 0.0
        elif result == "draw":
            reward = 0.5
        elif result == "not_ended":
            reward = 0.0
        else:
            reward = 0.0
        if reward < 0:
            raise ValueError("Returned reward must be non-negative!")
        ended = result in ["win", "lose", "draw"]
        return reward, ended

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return 1.0 - value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

    def render(self, state):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nBoard:")
        for r in range(self.row_count):
            print(" | ".join(symbols[int(x)] for x in state[r]))
            if r < self.row_count - 1:
                print("-" * (self.column_count * 4 - 1))
        print()