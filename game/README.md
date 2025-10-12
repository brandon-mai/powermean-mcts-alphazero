## AbstractGame Implementation Guide
This guide describes how to implement a new game class by inheriting from `AbstractGame`. Please follow these requirements to ensure compatibility and consistency.
### Required Attributes
1. `self.name: str`  
	The name of the game.
2. `self.num_player: int`  
	The number of players in the game. Only 1-player or 2-player games are supported.

### Player Index Rules
- For 1-player games: player index must always be 1.
- For 2-player games: player index must be either 1 or -1.

### Reward Rules
- Returned reward must always be non-negative (>= 0). If your logic returns a negative reward, raise an error.
### Required Methods
1. `get_initial_state(self)`  
	Returns the initial state of the game as a numpy object.
2. `get_next_state(self, state, action)`  
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
	For 2-player games, returns the opponent of the given player (i.e., returns -player, where player is either 1 or -1).
    For 1-player games, just return player.
6. `get_opponent_value(self, value)`  
	Returns the value as seen from the perspective of opponent.
7. `change_perspective(self, state, player)`  
	Returns the state as seen from the perspective of the specified player to player 1's view.
8. `get_encoded_state(self, state)`  
	Returns a numpy object representing the encoded state. MUST handle state batch with np.swapaxes().
9. `get_current_player(self, state)`  
	Returns the current player (as an integer) for the given state.
### Notes
- All methods must be implemented as described above.
- Attribute types must match the requirements.
- The implementation should be clear and easy to maintain.
- The methods get_encoded_state and change_perspective should be able to handle stacked states (multi-state input), e.g., when the input is an array of multiple states, to ensure compatibility with operations using np.stack().

This structure ensures that all game classes are consistent and compatible with the rest of the framework.
