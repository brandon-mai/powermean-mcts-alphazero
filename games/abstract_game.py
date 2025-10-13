from abc import ABC, abstractmethod

class AbstractGame(ABC):
    @abstractmethod
    def get_initial_state(self):
        pass
    def __init__(self, name: str, num_player: int):
        self.name = name
        self.num_player = num_player

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def get_valid_moves(self, state) -> list:
        pass

    @abstractmethod
    def check_win(self, state, player: int) -> str:
        pass

    @abstractmethod
    def get_value_and_terminated(self, state, player):
        pass

    @abstractmethod
    def get_opponent(self, player):
        pass

    @abstractmethod
    def get_opponent_value(self, value, player):
        pass

    @abstractmethod
    def change_perspective(self, state, player):
        pass

    @abstractmethod
    def get_encoded_state(self, state):
        pass

    @abstractmethod
    def get_current_player(self, state):
        pass    
