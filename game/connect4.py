import numpy as np
from abstract_game import AbstractGame

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
        count2 = np.sum(state == 2)
        return 1 if count1 <= count2 else 2

    def get_next_state(self, state, action):
        player = self.get_current_player(state)
        row = np.max(np.where(state[:, action] == 0))
        new_state = state.copy()
        new_state[row, action] = player
        return new_state

    def get_valid_moves(self, state) -> list:
        return [i for i in range(self.column_count) if state[0, i] == 0]

    def check_win(self, state, player: int) -> str:
        for action in range(self.column_count):
            if state[0, action] != 0:
                row = np.min(np.where(state[:, action] != 0))
                column = action
                p = state[row][column]
                if p == player and self._is_win(state, row, column, player):
                    return "win"
                elif p != player and self._is_win(state, row, column, p):
                    return "lose"
        if all(state[0, :] != 0):
            return "draw"
        return "not_ended"  

    def get_value_and_terminated(self, state, player):
        result = self.check_win(state, player)
        if result == "win":
            return 1.0, True
        elif result == "lose":
            return -1.0, True
        elif result == "draw":
            return 0.0, True
        elif result == "not_ended":
            return 0.0, False

    def get_opponent(self, player):
        return 2 if player == 1 else 1

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == 1, state == 0, state == 2)
        ).astype(np.float32)
        return encoded_state

    def _is_win(self, state, row, column, player):
        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1
        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )


