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
    num_player = getattr(game_obj, "num_player", None)
    if num_player not in [1, 2]:
        report["ok"] = False
        report["messages"].append(f"num_player must be 1 or 2, got {num_player}")
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

    import random
    state = game_obj.get_initial_state()
    results["initial_state_shape"] = state.shape
    step = 0
    ended = False
    history = []
    while not ended:
        try:
            valid = game_obj.get_valid_moves(state)
            if not isinstance(valid, list):
                raise TypeError("get_valid_moves must return a list!")
            if len(valid) == 0:
                results["messages"].append(f"Step {step}: No valid moves available.")
                break
            current = game_obj.get_current_player(state)
            # Check player index
            num_player = getattr(game_obj, "num_player", None)
            if num_player == 1 and current != 1:
                raise ValueError(f"For 1-player games, player index must be 1, got {current}")
            if num_player == 2 and current not in [1, -1]:
                raise ValueError(f"For 2-player games, player index must be 1 or -1, got {current}")
            move = random.choice(valid)
            next_state = game_obj.get_next_state(state, move)
            reward, ended = game_obj.get_value_and_terminated(next_state, current)
            # Check reward
            if reward < 0:
                raise ValueError(f"Returned reward must be non-negative! Got {reward} at step {step}")
            check = game_obj.check_win(next_state, current)
            opp = game_obj.get_opponent(current)
            opp_val = game_obj.get_opponent_value(reward)
            enc = game_obj.get_encoded_state(next_state)
            history.append({
                "step": step,
                "player": current,
                "move": move,
                "reward": reward,
                "ended": ended,
                "check_win": check,
                "opponent": opp,
                "opponent_value": opp_val,
                "encoded_shape": enc.shape if isinstance(enc, np.ndarray) else None
            })
            state = next_state
            step += 1
        except Exception as e:
            results["ok"] = False
            results["messages"].append(f"Step {step}: Exception: {e}")
            break
    results["history"] = history
    if history:
        results["final_reward"] = history[-1]["reward"]
        results["final_ended"] = history[-1]["ended"]
        results["final_check_win"] = history[-1]["check_win"]
    return results

def print_report(interface_report: dict, runtime_report: dict):
    print("\n==================== INTERFACE CHECK ====================")
    ok = interface_report.get("ok")
    for m in interface_report.get("messages", []):
        if any(w in m.lower() for w in ["missing", "error", "not", "fail"]):
            print(f"[ERROR] {m}")
        else:
            print(f"[OK] {m}")
    print(f"Interface overall: {'✅ PASS' if ok else '❌ FAIL'}")

    print("\n==================== RUNTIME TESTS =====================")
    ok = runtime_report.get("ok")
    for k, v in runtime_report.items():
        if k == "messages":
            for m in v:
                if "Exception" in m or "error" in m.lower() or "fail" in m.lower():
                    print(f"[ERROR] {m}")
                else:
                    print(f"[INFO] {m}")
        elif k == "history":
            print(f"Game steps: {len(v)}")
            if v:
                print("Last 3 steps:")
                for h in v[-3:]:
                    print(f"  Step {h['step']}: Player={h['player']}, Move={h['move']}, Reward={h['reward']}, Ended={h['ended']}, Win={h['check_win']}")
        else:
            print(f"{k}: {v}")
    print(f"Runtime overall: {'✅ PASS' if ok else '❌ FAIL'}")

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
