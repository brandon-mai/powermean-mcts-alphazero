import numpy as np
from abstract_game import AbstractGame

class TicTacToe(AbstractGame):
    def __init__(self):
        super().__init__(name="TicTacToe", num_player=2)
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_current_player(self, state):
        count1 = np.sum(state == 1)
        count2 = np.sum(state == -1)
        return 1 if count1 <= count2 else -1

    def get_next_state(self, state, action):
        player = self.get_current_player(state)
        row = action // self.column_count
        column = action % self.column_count
        new_state = state.copy()
        new_state[row, column] = player
        return new_state

    def get_valid_moves(self, state) -> list:
        return [i for i in range(self.action_size) if state.flatten()[i] == 0]

    def check_win(self, state, player: int) -> str:
        if self._is_win(state, player):
            return "win"
        elif self._is_win(state, self.get_opponent(player)):
            return "lose"
        elif not any(v == 0 for v in state.flatten()):
            return "draw"
        return "not_ended"

    def get_value_and_terminated(self, state, player):
        result = self.check_win(state, player)
        if result == "win":
            reward = 1.0
        elif result == "lose":
            reward = 0.0
        elif result == "draw":
            reward = 0.0
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
        return -value

    def change_perspective(self, state, player):
        if player == 1:
            return state.copy()
        elif player == -1:
            new_state = state.copy()
            new_state[state == 1] = 99
            new_state[state == -1] = 1
            new_state[new_state == 99] = -1
            return new_state
        else:
            raise ValueError("player must be 1 or -1")

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == 1, state == 0, state == 2)
        ).astype(np.float32)
        return encoded_state

    def _is_win(self, state, player):
        # Check rows and columns
        for i in range(self.row_count):
            if np.all(state[i, :] == player):
                return True
        for j in range(self.column_count):
            if np.all(state[:, j] == player):
                return True
        # Check diagonals
        if np.all(np.diag(state) == player):
            return True
        if np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False