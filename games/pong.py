import numpy as np
from games.abstract_game import AbstractGame
import math
from dataclasses import dataclass
import logging
from typing import Optional, Tuple, Dict, List

try:
    from gymnasium import spaces
except ImportError:
    spaces = None  # type: ignore

# Constants for agent naming
AGENTS = ["player_0", "player_1"]
ATARI_AGENTS = ["first_0", "second_0"]

@dataclass
class PongConfig:
    width: int = 160
    height: int = 120
    paddle_width: int = 4
    paddle_height: int = 20
    paddle_step: int = 4  # pixels per step (discrete)
    ball_size: int = 4
    ball_speed: float = 3.0  # pixels per step
    ball_speed_increment: float = 1.05  # speedup after paddle hit
    max_score: int = 10
    max_steps: Optional[int] = None  # truncation threshold; None disables
    center_line: bool = True
    background_color: Tuple[int, int, int] = (0, 0, 0)
    foreground_color: Tuple[int, int, int] = (255, 255, 255)


class PongCore:
    """
    Minimal deterministic Pong engine with discrete-time stepping.
    - Two paddles: left (player 0) and right (player 1)
    - Actions: -1 (up), 0 (no-op), 1 (down)
    - Rendering: numpy RGB array
    """

    def __init__(self, config: Optional[PongConfig] = None, seed: Optional[int] = None) -> None:
        self._log = logging.getLogger(__name__ + ".PongCore")
        self.config = config or PongConfig()
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Dynamic state
        self.ball_pos = np.zeros(2, dtype=np.float32)  # x, y
        self.ball_vel = np.zeros(2, dtype=np.float32)  # vx, vy
        self.paddle_y = np.zeros(2, dtype=np.float32)  # left, right (top edge)
        self.score = np.zeros(2, dtype=np.int32)
        self.step_count: int = 0

        self.reset(seed)

    # ---------------------------- Public API ---------------------------- #
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.score[:] = 0

        # Center paddles vertically
        self.paddle_y[0] = (self.config.height - self.config.paddle_height) / 2
        self.paddle_y[1] = (self.config.height - self.config.paddle_height) / 2

        # Reset ball in the middle with random direction
        self._reset_ball(direction=None)
        self._log.debug(
            "reset: step=%d score=%s paddle_y=%s ball_pos=%s ball_vel=%s",
            self.step_count,
            self.score.tolist(),
            self.paddle_y.tolist(),
            self.ball_pos.tolist(),
            self.ball_vel.tolist(),
        )

    def step(self, left_action: int, right_action: int) -> Dict[str, bool]:
        """
        Perform one simulation step.
        Returns flags: {"scored": bool, "left_scored": bool, "right_scored": bool}
        """
        self.step_count += 1

        # Update paddles
        self._move_paddle(0, left_action)
        self._move_paddle(1, right_action)

        # Update ball position
        self.ball_pos += self.ball_vel
        self._log.debug(
            "step=%d actions(L,R)=(%d,%d) pre-coll ball_pos=%s ball_vel=%s paddles=%s",
            self.step_count,
            int(left_action),
            int(right_action),
            self.ball_pos.tolist(),
            self.ball_vel.tolist(),
            self.paddle_y.tolist(),
        )

        # Collide with top/bottom walls
        if self.ball_pos[1] <= 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] = abs(self.ball_vel[1])
            self._log.debug("bounce top: ball_pos=%s ball_vel=%s", self.ball_pos.tolist(), self.ball_vel.tolist())
        elif self.ball_pos[1] + self.config.ball_size >= self.config.height:
            self.ball_pos[1] = self.config.height - self.config.ball_size
            self.ball_vel[1] = -abs(self.ball_vel[1])
            self._log.debug("bounce bottom: ball_pos=%s ball_vel=%s", self.ball_pos.tolist(), self.ball_vel.tolist())

        # Check paddle collisions
        left_scored = False
        right_scored = False

        # Left paddle at x = 0 .. paddle_width
        if self.ball_pos[0] <= self.config.paddle_width:
            if self._ball_intersects_paddle(0):
                self._bounce_off_paddle(0)
                self._log.debug("hit left paddle: ball_pos=%s ball_vel=%s", self.ball_pos.tolist(), self.ball_vel.tolist())
            else:
                # Missed by left: right scores
                right_scored = True
                self.score[1] += 1
                self._reset_ball(direction="left")  # serve toward scorer
                self._log.debug("score right: score=%s", self.score.tolist())

        # Right paddle at x = width - paddle_width .. width
        elif self.ball_pos[0] + self.config.ball_size >= self.config.width - self.config.paddle_width:
            if self._ball_intersects_paddle(1):
                self._bounce_off_paddle(1)
                self._log.debug("hit right paddle: ball_pos=%s ball_vel=%s", self.ball_pos.tolist(), self.ball_vel.tolist())
            else:
                # Missed by right: left scores
                left_scored = True
                self.score[0] += 1
                self._reset_ball(direction="right")
                self._log.debug("score left: score=%s", self.score.tolist())

        scored = left_scored or right_scored
        self._log.debug(
            "post-step step=%d ball_pos=%s ball_vel=%s score=%s",
            self.step_count,
            self.ball_pos.tolist(),
            self.ball_vel.tolist(),
            self.score.tolist(),
        )
        return {"scored": scored, "left_scored": left_scored, "right_scored": right_scored}

    def is_terminated(self) -> bool:
        return bool(np.any(self.score >= self.config.max_score))

    def is_truncated(self) -> bool:
        if self.config.max_steps is None:
            return False
        return self.step_count >= int(self.config.max_steps)

    def observation_left(self) -> np.ndarray:
        return self._make_observation(perspective=0)

    def observation_right(self) -> np.ndarray:
        return self._make_observation(perspective=1)

    def all_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.observation_left(), self.observation_right()

    def render_rgb(self, scale: int = 1) -> np.ndarray:
        h = self.config.height
        w = self.config.width
        bg = np.array(self.config.background_color, dtype=np.uint8)
        fg = np.array(self.config.foreground_color, dtype=np.uint8)

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, :] = bg

        # Center line
        if self.config.center_line:
            frame[:, w // 2 - 1 : w // 2 + 1, :] = fg

        # Paddles
        pw = self.config.paddle_width
        ph = self.config.paddle_height
        # Left paddle
        y0 = int(round(self.paddle_y[0]))
        frame[y0 : y0 + ph, 0:pw, :] = fg
        # Right paddle
        y1 = int(round(self.paddle_y[1]))
        frame[y1 : y1 + ph, w - pw : w, :] = fg

        # Ball
        bx = int(round(self.ball_pos[0]))
        by = int(round(self.ball_pos[1]))
        bs = self.config.ball_size
        frame[by : by + bs, bx : bx + bs, :] = fg

        # Scoreboard (3x5 pixel digits)
        self._draw_scoreboard(frame, color=fg)

        if scale != 1:
            frame = self._nearest_neighbor_scale(frame, scale)
        return frame

    # --------------------------- Internal utils ------------------------- #
    def _reset_ball(self, direction: Optional[str]) -> None:
        # Place at center
        self.ball_pos[0] = (self.config.width - self.config.ball_size) / 2
        self.ball_pos[1] = (self.config.height - self.config.ball_size) / 2

        # Random launch angle biased horizontally
        angle = float(self.rng.uniform(low=-0.35 * math.pi, high=0.35 * math.pi))
        vx = math.cos(angle)
        vy = math.sin(angle)
        speed = self.config.ball_speed

        if direction == "left":
            vx = -abs(vx)
        elif direction == "right":
            vx = abs(vx)
        else:
            vx = math.copysign(abs(vx), float(self.rng.choice([-1, 1])))

        self.ball_vel[0] = speed * vx
        self.ball_vel[1] = speed * vy

    def _move_paddle(self, idx: int, action: int) -> None:
        # action in {-1, 0, 1}
        dy = float(self.config.paddle_step * int(np.sign(action)))
        self.paddle_y[idx] = float(np.clip(
            self.paddle_y[idx] + dy,
            0,
            self.config.height - self.config.paddle_height,
        ))

    def _ball_intersects_paddle(self, idx: int) -> bool:
        # Axis-aligned rectangle intersection
        pw = self.config.paddle_width
        ph = self.config.paddle_height
        bx, by = self.ball_pos
        bs = self.config.ball_size

        if idx == 0:
            rx0, rx1 = 0, pw
        else:
            rx0, rx1 = self.config.width - pw, self.config.width
        ry0 = self.paddle_y[idx]
        ry1 = ry0 + ph

        return not (
            bx + bs < rx0 or bx > rx1 or by + bs < ry0 or by > ry1
        )

    def _bounce_off_paddle(self, idx: int) -> None:
        # Reverse X velocity and tweak Y based on contact point
        ph = self.config.paddle_height
        pw = self.config.paddle_width
        bs = self.config.ball_size

        # Clamp ball to paddle edge to avoid tunneling
        if idx == 0:
            self.ball_pos[0] = float(pw)
            self.ball_vel[0] = abs(self.ball_vel[0])
        else:
            self.ball_pos[0] = float(self.config.width - pw - bs)
            self.ball_vel[0] = -abs(self.ball_vel[0])

        # Influence vertical component based on hit location
        paddle_center = self.paddle_y[idx] + ph / 2.0
        ball_center_y = self.ball_pos[1] + bs / 2.0
        offset = (ball_center_y - paddle_center) / (ph / 2.0)
        offset = float(np.clip(offset, -1.0, 1.0))

        # Blend some of offset into vy and normalize speed a bit
        speed = float(np.linalg.norm(self.ball_vel))
        if speed <= 1e-5:
            speed = self.config.ball_speed

        new_vx = math.copysign(max(1.0, abs(self.ball_vel[0])), self.ball_vel[0])
        new_vy = offset * speed

        vec = np.array([new_vx, new_vy], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vec *= speed * self.config.ball_speed_increment
        self.ball_vel[:] = vec

    def _make_observation(self, perspective: int) -> np.ndarray:
        """
        Vector observation normalized to [-1, 1].
        Contents (from the given player's perspective):
        [ball_x, ball_y, ball_vx, ball_vy, own_paddle_y, opp_paddle_y,
         own_score, opp_score]
        """
        w = float(self.config.width)
        h = float(self.config.height)
        bs = float(self.config.ball_size)
        ph = float(self.config.paddle_height)

        # Normalize positions to [-1, 1]
        bx = (self.ball_pos[0] + bs * 0.5) / w * 2.0 - 1.0
        by = (self.ball_pos[1] + bs * 0.5) / h * 2.0 - 1.0
        bvx = np.clip(self.ball_vel[0] / (self.config.ball_speed * 2.0), -1.0, 1.0)
        bvy = np.clip(self.ball_vel[1] / (self.config.ball_speed * 2.0), -1.0, 1.0)

        own_idx = perspective
        opp_idx = 1 - perspective
        own_py = (self.paddle_y[own_idx] + ph * 0.5) / h * 2.0 - 1.0
        opp_py = (self.paddle_y[opp_idx] + ph * 0.5) / h * 2.0 - 1.0

        own_score = float(self.score[own_idx]) / max(1.0, float(self.config.max_score)) * 2.0 - 1.0
        opp_score = float(self.score[opp_idx]) / max(1.0, float(self.config.max_score)) * 2.0 - 1.0

        obs = np.array([bx, by, bvx, bvy, own_py, opp_py, own_score, opp_score], dtype=np.float32)
        return obs

    @staticmethod
    def _nearest_neighbor_scale(img: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return img
        h, w = img.shape[:2]
        return img.repeat(scale, axis=0).repeat(scale, axis=1)

    # --------------------------- Drawing helpers ------------------------- #
    @staticmethod
    def _digit_bitmap(d: int) -> np.ndarray:
        # 3x5 font; '1' pixels filled
        DIGITS = {
            0: ("111",
                "101",
                "101",
                "101",
                "111"),
            1: ("010",
                "110",
                "010",
                "010",
                "111"),
            2: ("111",
                "001",
                "111",
                "100",
                "111"),
            3: ("111",
                "001",
                "111",
                "001",
                "111"),
            4: ("101",
                "101",
                "111",
                "001",
                "001"),
            5: ("111",
                "100",
                "111",
                "001",
                "111"),
            6: ("111",
                "100",
                "111",
                "101",
                "111"),
            7: ("111",
                "001",
                "010",
                "010",
                "010"),
            8: ("111",
                "101",
                "111",
                "101",
                "111"),
            9: ("111",
                "101",
                "111",
                "001",
                "111"),
        }
        arr = np.array([[c == '1' for c in row] for row in DIGITS[int(d) % 10]], dtype=bool)
        return arr

    def _draw_digit(self, frame: np.ndarray, x: int, y: int, d: int, color: np.ndarray) -> None:
        bm = self._digit_bitmap(d)
        h, w = frame.shape[:2]
        for r in range(bm.shape[0]):
            yy = y + r
            if yy < 0 or yy >= h:
                continue
            for c in range(bm.shape[1]):
                xx = x + c
                if xx < 0 or xx >= w:
                    continue
                if bm[r, c]:
                    frame[yy, xx, :] = color

    def _draw_number(self, frame: np.ndarray, x: int, y: int, value: int, color: np.ndarray) -> None:
        s = str(int(max(0, value)))
        if len(s) > 2:
            s = s[-2:]  # clamp display to last two digits
        cursor_x = x
        for i, ch in enumerate(s):
            self._draw_digit(frame, cursor_x, y, int(ch), color)
            cursor_x += 4  # 3px digit + 1px space

    def _draw_scoreboard(self, frame: np.ndarray, color: np.ndarray) -> None:
        w = frame.shape[1]
        # Place near top center with small margin
        y = 4
        center_x = w // 2
        left_value = int(self.score[0])
        right_value = int(self.score[1])

        # Compute left number width to right-align at center-2
        left_len = len(str(max(0, left_value)))
        left_total_w = left_len * 3 + (left_len - 1) * 1 if left_len > 0 else 0
        left_x = center_x - 2 - left_total_w
        right_x = center_x + 2

        self._draw_number(frame, left_x, y, left_value, color)
        self._draw_number(frame, right_x, y, right_value, color)


class PongAtariLikeParallelEnv:
    """Parallel API env mirroring PettingZoo Atari Pong specs.

    - Agents: ['first_0', 'second_0']
    - Action space: Discrete(6) with values [0..5]
      We map to PongCore actions as: 0->stay, 1->stay, 2->up, 3->down, 4->up, 5->down
    - Observation space: RGB uint8 with shape (210, 160, 3)
    """

    metadata = {"render_modes": ["rgb_array"], "name": "custom_pong_atari_like_v0"}

    def __init__(self, config: Optional[PongConfig] = None, render_mode: Optional[str] = None, seed: Optional[int] = None):
        if spaces is None:
            raise ImportError("gymnasium is required for PongAtariLikeParallelEnv")
        self._log = logging.getLogger(__name__ + ".PongAtariLikeParallelEnv")

        # Force Atari-like dimensions (W=160, H=210)
        cfg = config or PongConfig()
        cfg.width = 160
        cfg.height = 210
        self.core = PongCore(config=cfg, seed=seed)

        self.render_mode = render_mode
        self.possible_agents: List[str] = ATARI_AGENTS.copy()
        self.agents: List[str] = ATARI_AGENTS.copy()

        obs_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
        self.observation_spaces = {a: obs_space for a in ATARI_AGENTS}
        self.action_spaces = {a: spaces.Discrete(6) for a in ATARI_AGENTS}

    @staticmethod
    def _map_atari_action(a: int) -> int:
        # Map Atari Pong action id to PongCore action in {-1,0,1}
        # 0: NOOP -> 0, 1: FIRE -> 0, 2: UP -> -1, 3: DOWN -> +1,
        # 4: UP+FIRE -> -1, 5: DOWN+FIRE -> +1
        if int(a) in (2, 4):
            return -1
        if int(a) in (3, 5):
            return 1
        return 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.agents = ATARI_AGENTS.copy()
        if seed is not None:
            self.core.reset(seed=seed)
        else:
            self.core.reset()
        frame = self.core.render_rgb(scale=1)
        observations = {ATARI_AGENTS[0]: frame.copy(), ATARI_AGENTS[1]: frame.copy()}
        infos = {a: {"score_left": int(self.core.score[0]), "score_right": int(self.core.score[1])} for a in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        left_action = self._map_atari_action(int(actions.get(ATARI_AGENTS[0], 0)))
        right_action = self._map_atari_action(int(actions.get(ATARI_AGENTS[1], 0)))

        flags = self.core.step(left_action=left_action, right_action=right_action)

        terminated = self.core.is_terminated()
        truncated = self.core.is_truncated()

        frame = self.core.render_rgb(scale=1)
        observations = {ATARI_AGENTS[0]: frame.copy(), ATARI_AGENTS[1]: frame.copy()}

        rewards = {ATARI_AGENTS[0]: 0.0, ATARI_AGENTS[1]: 0.0}
        if flags["left_scored"]:
            rewards[ATARI_AGENTS[0]] += 1.0
            rewards[ATARI_AGENTS[1]] -= 1.0
        if flags["right_scored"]:
            rewards[ATARI_AGENTS[1]] += 1.0
            rewards[ATARI_AGENTS[0]] -= 1.0

        terminations = {a: bool(terminated) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        infos = {a: {"score_left": int(self.core.score[0]), "score_right": int(self.core.score[1])} for a in self.agents}

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return self.core.render_rgb(scale=1)
        return None

    def close(self):
        pass


class PongAtari(AbstractGame):
    """
    Pong game with Atari-like RGB observations (210, 160, 3).
    
    This version uses RGB images like the original Atari Pong for training.
    - State: RGB image (210, 160, 3) - uint8 [0, 255]
    - Actions: 6 Atari actions (0=NOOP, 1=FIRE, 2=UP, 3=DOWN, 4=UP+FIRE, 5=DOWN+FIRE)
    - Agents: ['first_0', 'second_0'] (Atari naming convention)
    
    This is suitable for CNN-based models like ResNet.
    """
    
    def __init__(self, config: PongConfig = None, seed: int = None):
        super().__init__(name="PongAtari", num_player=2)
        # Atari-like config with forced dimensions
        self.config = config or PongConfig(max_score=1, max_steps=2000)
        self.seed = seed
        self.env = None
        
        # For AlphaZero compatibility
        self.row_count = 210  # Height
        self.column_count = 160  # Width
        self.action_size = 6  # Atari Pong actions: 0-5
        
        # Track both players' actions for simultaneous execution
        self._pending_action = {1: 0, -1: 0}  # Default to NOOP
        
        # Internal state tracking
        self._reset_env()
        
    def _reset_env(self):
        """Reset the internal environment."""
        if self.env is not None:
            self.env.close()
        self.env = PongAtariLikeParallelEnv(config=self.config, seed=self.seed)
        obs, info = self.env.reset(seed=self.seed)
        self._current_obs = obs
        self._current_info = info
        self._done = False
        self._step_count = 0
        self._pending_action = {1: 0, -1: 0}
        self._current_player = 1
        
    def get_initial_state(self):
        """Return initial state as RGB observation."""
        self._reset_env()
        self._current_player = 1  # Start with player 1
        # Return the observation for first player (both see the same frame)
        return self._current_obs[ATARI_AGENTS[0]].copy()
    
    def get_current_player(self, state):
        """Return the current player (1 or -1)."""
        return self._current_player
    
    def get_next_state(self, state, action):
        """
        Apply action and return next state.
        Turn-based: alternates between players.
        """
        if self._done:
            return state
        
        # Store current player's action
        self._pending_action[self._current_player] = int(action)
        
        # Switch to the other player
        self._current_player = -self._current_player
        
        # If we've collected both actions (back to player 1), execute the step
        if self._current_player == 1:
            actions = {
                ATARI_AGENTS[0]: self._pending_action[1],   # first_0 = player 1
                ATARI_AGENTS[1]: self._pending_action[-1]   # second_0 = player -1
            }
            
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            self._current_obs = obs
            self._current_info = infos
            self._done = any(terminations.values()) or any(truncations.values())
            self._step_count += 1
        
        # Return the updated observation
        return self._current_obs[ATARI_AGENTS[0]].copy()
    
    def get_valid_moves(self, state):
        """All 6 Atari actions are always valid. Return as list."""
        return list(range(self.action_size))
    
    def check_win(self, state, player):
        """Check if the player won."""
        if not self._done:
            return False
        
        # Check if someone scored
        left_score = self._current_info[ATARI_AGENTS[0]]["score_left"]
        right_score = self._current_info[ATARI_AGENTS[0]]["score_right"]
        
        if player == 1:
            return left_score > right_score
        else:
            return right_score > left_score
    
    def get_value_and_terminated(self, state, player):
        """
        Return (value, terminated) tuple.
        Value is non-negative: 1 for win, 0 for draw/loss/ongoing.
        """
        if not self._done:
            return 0, False
        
        # Check scores
        left_score = self._current_info[ATARI_AGENTS[0]]["score_left"]
        right_score = self._current_info[ATARI_AGENTS[0]]["score_right"]
        
        if player == 1:
            if left_score > right_score:
                return 1, True  # Win
            else:
                return 0, True  # Loss or draw
        else:  # player == -1
            if right_score > left_score:
                return 1, True  # Win
            else:
                return 0, True  # Loss or draw
    
    def get_opponent(self, player):
        """Return the opponent player."""
        return -player
    
    def get_opponent_value(self, value):
        """Return value from opponent's perspective."""
        return 1 - value
    
    def change_perspective(self, state, player):
        """
        Change perspective of the state.
        For player -1 (right player), we flip the image horizontally.
        """
        if player == -1:
            return np.fliplr(state)
        return state
    
    def get_encoded_state(self, state):
        """
        Encode state for neural network.
        For RGB images (210, 160, 3), normalize to [0, 1] and transpose to (C, H, W).
        Handles both single state and batch of states.
        """
        # state can be:
        # - single: (210, 160, 3) in uint8 [0, 255]
        # - batch: (batch_size, 210, 160, 3) in uint8 [0, 255]
        
        # Normalize to [0, 1]
        encoded = state.astype(np.float32) / 255.0
        
        # Handle batch vs single state
        if len(state.shape) == 4:
            # Batch: (batch_size, H, W, C) -> (batch_size, C, H, W)
            encoded = np.transpose(encoded, (0, 3, 1, 2))
        else:
            # Single: (H, W, C) -> (C, H, W)
            encoded = np.transpose(encoded, (2, 0, 1))
        
        return encoded
    
    def __repr__(self):
        return "PongAtari"