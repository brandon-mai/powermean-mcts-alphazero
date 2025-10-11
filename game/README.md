## AbstractGame Implementation Guide
This guide describes how to implement a new game class by inheriting from `AbstractGame`. Please follow these requirements to ensure compatibility and consistency.
### Required Attributes
1. `self.name: str`  
	The name of the game.
2. `self.num_player: int`  
	The number of players in the game.
### Required Methods
1. `get_next_state(self, state, action)`  
	Returns the next state after performing the given action.
2. `get_valid_moves(self, state)`  
	Returns a list of valid actions for the given state.
3. `check_win(self, state, player: int)`  
	Returns one of the following strings: "win", "lose", "draw", or "not_ended" from the perspective of the specified player. "not_ended" means the game is still ongoing.
4. `get_value_and_terminated(self, state, player)`  
	Returns two values:
	- `reward`: a float representing the intermediate reward
	- `end`: a boolean indicating whether the game is over
5. `get_opponent(self, player)`  
	Returns the next player in turn order after the given player. For multiplayer games, this should cycle through all players (e.g., 1 → 2 → 3 → 4 → 1).
6. `get_opponent_value(self, value)`  
	Returns the value as seen from the perspective of opponent.
7. `change_perspective(self, state, player)`  
	Returns the state as seen from the perspective of the specified player.
8. `get_encoded_state(self, state)`  
	Returns a numpy object representing the encoded state.
9. `get_current_player(self, state)`  
	Returns the current player (as an integer) for the given state.
### Notes
- All methods must be implemented as described above.
- Attribute types must match the requirements.
- The implementation should be clear and easy to maintain.
This structure ensures that all game classes are consistent and compatible with the rest of the framework.
