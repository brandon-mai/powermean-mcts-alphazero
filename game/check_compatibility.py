from typing import Any
import numpy as np
import inspect

# Import the game to check here. Replace with another game class as needed.
from tictactoe import TicTacToe as GameClass

from abstract_game import AbstractGame

REQUIRED_METHODS = [
    "get_next_state",
    "get_valid_moves",
    "check_win",
    "get_value_and_terminated",
    "get_opponent",
    "get_opponent_value",
    "change_perspective",
    "get_encoded_state",
    "get_current_player",
    "get_initial_state"
]

REQUIRED_ATTRIBUTES = ["name", "num_player"]

def check_interface(game_obj: Any) -> dict:
    report = {"ok": True, "messages": []}

    try:
        is_subclass = isinstance(game_obj, AbstractGame)
    except Exception:
        is_subclass = False
    report["is_subclass_of_AbstractGame"] = is_subclass
    if not is_subclass:
        report["ok"] = False
        report["messages"].append("Game instance is not an AbstractGame subclass (or check failed).")

    # Attributes
    for attr in REQUIRED_ATTRIBUTES:
        if not hasattr(game_obj, attr):
            report["ok"] = False
            report["messages"].append(f"Missing required attribute: {attr}")
        else:
            report["messages"].append(f"Found attribute: {attr} (value={getattr(game_obj, attr)})")

    # Methods
    for name in REQUIRED_METHODS:
        if not hasattr(game_obj, name) or not callable(getattr(game_obj, name)):
            report["ok"] = False
            report["messages"].append(f"Missing or non-callable required method: {name}")
        else:
            try:
                sig = inspect.signature(getattr(game_obj, name))
                report["messages"].append(f"Found method: {name}{sig}")
            except Exception:
                report["messages"].append(f"Found method: {name} (signature unavailable)")

    return report

def run_runtime_tests(game_obj: Any) -> dict:
    results = {"ok": True, "messages": []}

    state = game_obj.get_initial_state()
    results["initial_state_shape"] = state.shape

    # get_valid_moves
    try:
        valid = game_obj.get_valid_moves(state)
        results["valid_moves"] = valid
        if not isinstance(valid, (list, tuple, np.ndarray)):
            results["ok"] = False
            results["messages"].append("get_valid_moves did not return a list/tuple/ndarray")
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_valid_moves raised: {e}")
        return results

    # current player
    try:
        current = game_obj.get_current_player(state)
        results["current_player"] = current
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_current_player raised: {e}")
        return results

    # pick a valid move if any
    move = None
    try:
        if not isinstance(valid, list):
            raise TypeError("get_valid_moves must return a list!")
        if len(valid) == 0:
            results["messages"].append("No valid moves available on initial state")
        else:
            move = int(valid[0])
            results["picked_move"] = move
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"Error processing valid moves: {e}")
        return results

    # test get_next_state
    try:
        if move is not None:
            next_state = game_obj.get_next_state(state, move)
            results["next_state_shape"] = getattr(next_state, "shape", None)
        else:
            next_state = state
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_next_state raised: {e}")
        return results

    # test get_value_and_terminated
    try:
        reward, ended = game_obj.get_value_and_terminated(next_state, current)
        results["reward"] = reward
        results["ended"] = bool(ended)
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_value_and_terminated raised: {e}")

    # test check_win
    try:
        check = game_obj.check_win(next_state, current)
        results["check_win"] = check
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"check_win raised: {e}")

    # test opponent and opponent_value
    try:
        opp = game_obj.get_opponent(current)
        results["opponent"] = opp
        opp_val = game_obj.get_opponent_value(reward)
        results["opponent_value"] = opp_val
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_opponent/get_opponent_value raised: {e}")

    # test encoded state
    try:
        enc = game_obj.get_encoded_state(next_state)
        if not isinstance(enc, np.ndarray):
            results["ok"] = False
            results["messages"].append("get_encoded_state did not return a numpy array")
        else:
            results["encoded_shape"] = enc.shape
    except Exception as e:
        results["ok"] = False
        results["messages"].append(f"get_encoded_state raised: {e}")

    return results

def print_report(interface_report: dict, runtime_report: dict):
    print("\n=== Interface Check ===")
    for m in interface_report.get("messages", []):
        print(" -", m)
    print("Interface overall OK:", interface_report.get("ok"))

    print("\n=== Runtime Tests ===")
    for k, v in runtime_report.items():
        if k == "messages":
            for m in v:
                print(" -", m)
        else:
            print(f"{k}: {v}")
    print("Runtime overall OK:", runtime_report.get("ok"))

def main():
    print("Checking game class:", GameClass.__name__)
    try:
        game = GameClass()
    except Exception as e:
        print("Failed to instantiate game class:", e)
        return

    iface = check_interface(game)
    runtime = run_runtime_tests(game)
    print_report(iface, runtime)

if __name__ == "__main__":
    main()
