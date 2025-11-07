"""
=====================================================================
STOCHASTIC MUZERO MCTS (Monte Carlo Tree Search)
=====================================================================

Stochastic MuZero mở rộng MuZero để xử lý environments có stochasticity
(ngẫu nhiên) như: 2048, games có dice, card games, real-world environments.

═══════════════════════════════════════════════════════════════════
PHÂN BIỆT CÁC LOẠI MCTS
═══════════════════════════════════════════════════════════════════

1. AlphaZero MCTS:
   - Sử dụng ENVIRONMENT THẬT để simulate
   - Biết chính xác rules của game
   - next_state = game.get_next_state(state, action)
   - Chỉ hoạt động với DETERMINISTIC environments
   
2. MuZero MCTS (Deterministic):
   - Sử dụng LEARNED MODEL để simulate  
   - KHÔNG cần biết rules của game
   - next_hidden_state, reward = model.dynamics(hidden_state, action)
   - Hoàn toàn "tưởng tượng" trong không gian latent
   - Vẫn chỉ cho DETERMINISTIC environments

3. Stochastic MuZero MCTS (Implementation này):
   - Mở rộng MuZero cho STOCHASTIC environments
   - Có 2 loại nodes: DECISION nodes và CHANCE nodes
   - Sử dụng AFTERSTATE representation
   - Xử lý được cả deterministic lẫn stochastic outcomes

═══════════════════════════════════════════════════════════════════
DECISION NODES vs CHANCE NODES
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│  DECISION NODE (Player's turn)                                  │
│  - Player chọn action a                                         │
│  - Node có policy over actions: π(a|s)                          │
│  - Expand tất cả possible actions                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ action a
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AFTERSTATE (sau action, trước chance)                          │
│  - State ngay sau khi player action                             │
│  - Nhưng trước khi environment respond                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ model predicts chance outcomes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CHANCE NODE (Environment's turn)                               │
│  - Environment produces stochastic outcome o                    │
│  - Node có probability distribution: p(o|s,a)                   │
│  - Sample theo probability (không phải player chọn)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ outcome o
                              ▼
                        NEXT DECISION NODE

Ví dụ trong game 2048:
- Decision node: Player chọn direction (up/down/left/right)
- Afterstate: Board sau khi tiles đã di chuyển và merge
- Chance node: Random tile (2 or 4) spawn ở random position
- Next state: Board với tile mới

═══════════════════════════════════════════════════════════════════
SO SÁNH VỚI LIGHTZERO IMPLEMENTATION
═══════════════════════════════════════════════════════════════════

LightZero (OpenDILab) implementation:
✅ Separate decision và chance nodes
✅ Afterstate dynamics network
✅ Batch processing cho efficient inference
✅ Min-max normalization cho values
✅ Support categorical distributions

Implementation này:
✅ Giữ nguyên core concepts từ LightZero
✅ Code dễ đọc, có comment chi tiết
✅ Flexible architecture support nhiều game types
⚠️ Chưa optimize batch processing như LightZero (sẽ thêm sau)

═══════════════════════════════════════════════════════════════════
ƯU ĐIỂM CỦA STOCHASTIC MUZERO
═══════════════════════════════════════════════════════════════════

1. ✅ Model hoá được stochastic environments chính xác
2. ✅ Học được optimal policy dưới uncertainty
3. ✅ Không cần biết probability distribution của environment
4. ✅ Áp dụng cho real-world: robotics, autonomous driving, etc.

═══════════════════════════════════════════════════════════════════
NHƯỢC ĐIỂM & CHALLENGES
═══════════════════════════════════════════════════════════════════

1. ⚠️ Phức tạp hơn deterministic MuZero
2. ⚠️ Training chậm hơn (2 loại networks)
3. ⚠️ Sample efficiency thấp hơn ở early training
4. ⚠️ Cần balance exploration giữa decisions và chances

═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import math
import copy

from muzero.model import categorical_to_scalar


class MinMaxStats:
    """Track min/max values for normalization during MCTS."""

    def __init__(self):
        self.minimum = float('inf')
        self.maximum = -float('inf')

    def update(self, value):
        if value is None:
            return
        value = float(value)
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if value is None:
            return 0.0
        if self.maximum > self.minimum:
            return float((value - self.minimum) / (self.maximum - self.minimum))
        return float(value)


class MuZeroNode:
    """
    ═══════════════════════════════════════════════════════════════════
    NODE TRONG STOCHASTIC MUZERO SEARCH TREE
    ═══════════════════════════════════════════════════════════════════
    
    Khác với AlphaZero Node:
    - Lưu HIDDEN STATE thay vì game state thật
    - Lưu PREDICTED REWARD từ dynamics model
    - Support cả DECISION nodes và CHANCE nodes
    
    Node Types:
    -----------
    1. DECISION NODE (is_chance=False):
       - Đại diện cho state mà player chọn action
       - Children = các actions khả dụng
       - Selection dựa trên UCB formula
       
    2. CHANCE NODE (is_chance=True):
       - Đại diện cho stochastic outcomes từ environment
       - Children = các possible outcomes
       - Selection dựa trên probability distribution
    
    Attributes:
    -----------
    hidden_state: torch.Tensor
        Hidden representation của state (từ model, không phải state thật)
        Shape: [channels, H, W] cho spatial hoặc [features] cho vector
        
    is_chance: bool
        True nếu đây là chance node (environment's turn)
        False nếu đây là decision node (player's turn)
        
    player: int
        Player hiện tại (0 hoặc 1) - chỉ có ý nghĩa với decision nodes
        
    prior: float
        Prior probability từ policy network (decision) hoặc 
        chance distribution (chance nodes)
        
    reward: float
        Predicted immediate reward khi transition đến node này
        Từ dynamics network: g(s,a) -> (s', r)
    """
    def __init__(self, hidden_state, player, prior=0, parent=None, 
                 action_taken=None, is_chance=False, min_max_stats=None):
        """
        Khởi tạo một node trong search tree
        
        Args:
            hidden_state: Hidden state từ model (torch.Tensor)
                         Không phải observation thật!
            player: Player hiện tại (0 hoặc 1)
            prior: Prior probability [0, 1]
                   - Decision node: từ policy network π(a|s)
                   - Chance node: từ chance distribution p(o|s,a)
            parent: Parent node (MuZeroNode hoặc None nếu root)
            action_taken: Action/outcome đã thực hiện để đến node này
                         - Decision node parent: integer action
                         - Chance node parent: integer outcome
            is_chance: Boolean flag
                      - False: decision node (player chọn action)
                      - True: chance node (environment produces outcome)
        """
        # ===== Core State Information =====
        self.hidden_state = hidden_state  # [C, H, W] hoặc [features]
        self.player = player              # 0 hoặc 1 (cho 2-player games)
        self.is_chance = is_chance        # Decision node hay Chance node?
        
        # ===== Tree Structure =====
        self.parent = parent              # Parent node (None nếu root)
        self.children = []                # List các child nodes
        self.action_taken = action_taken  # Action/outcome đưa đến node này
        
        # ===== Prior & Statistics =====
        self.prior = prior                # P(a|s) hoặc p(o|s,a)
        self.visit_count = 0              # Số lần node được visit
        self.value_sum = 0.0              # Tổng value qua các visits
        self.reward = 0.0                 # Predicted immediate reward r(s,a)

        # ===== Shared MinMax Statistics =====
        if min_max_stats is not None:
            self.min_max_stats = min_max_stats
        elif parent is not None and hasattr(parent, "min_max_stats"):
            self.min_max_stats = parent.min_max_stats
        else:
            self.min_max_stats = MinMaxStats()
    
    def is_expanded(self):  
        """
        Kiểm tra node đã được expand chưa
        
        Returns:
            bool: True nếu node đã có children, False nếu chưa
        """
        return len(self.children) > 0
    
    def value(self):
        """
        Tính giá trị trung bình (mean value) của node
        
        Q(s,a) = W(s,a) / N(s,a)
        
        Trong đó:
        - W(s,a) = value_sum: tổng values từ tất cả simulations
        - N(s,a) = visit_count: số lần node được visit
        
        Returns:
            float: Mean value [0, 1] hoặc 0 nếu chưa được visit
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def normalized_value(self):
        """Return normalized value using shared min/max stats."""
        if not hasattr(self, "min_max_stats") or self.min_max_stats is None:
            return self.value()
        return self.min_max_stats.normalize(self.value())
    
    def select_child(self, c_puct):
        """
        ════════════════════════════════════════════════════════════════
        SELECTION PHASE - Chọn child node tốt nhất
        ════════════════════════════════════════════════════════════════
        
        Behavior phụ thuộc vào node type:
        
        1. DECISION NODE (is_chance=False):
           Sử dụng PUCT (Predictor + Upper Confidence Bound for Trees):
           
           UCB(s,a) = Q(s,a) + c_puct · P(s,a) · √(N(s)) / (1 + N(s,a))
           
           Trong đó:
           - Q(s,a): Mean value của child (exploitation)
           - P(s,a): Prior probability từ policy network
           - N(s): Visit count của parent node (self)
           - N(s,a): Visit count của child node
           - c_puct: Exploration constant (thường ~1-2)
           
           Intuition:
           - Cao Q(s,a): action này cho value tốt → nên exploit
           - Cao P(s,a): policy network nghĩ action này tốt → tin model
           - Thấp N(s,a): chưa explore nhiều → nên thử thêm
           
        2. CHANCE NODE (is_chance=True):
           Sample theo PROBABILITY DISTRIBUTION:
           
           p(outcome) = prior probability từ model
           
           Không dùng UCB vì đây không phải decision của player.
           Environment sẽ produce outcome theo distribution của nó.
           
        Args:
            c_puct: Exploration constant (càng cao càng explore nhiều)
        
        Returns:
            MuZeroNode: Child node được chọn
        """
        # ===== CHANCE NODE: Sample theo probability =====
        if self.is_chance:
            # Lấy probability distribution từ priors
            probs = np.array([child.prior for child in self.children])
            
            # Normalize (đề phòng numerical errors)
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else:
                # Uniform nếu không có prior info
                probs = np.ones(len(self.children)) / len(self.children)
            
            # Sample outcome theo distribution
            idx = np.random.choice(len(self.children), p=probs)
            return self.children[idx]
        
        # ===== DECISION NODE: UCB Selection =====
        best_child = None
        best_ucb = -float('inf')
        
        for child in self.children:
            # Q(s,a): Giá trị trung bình (exploitation)
            q_value = child.normalized_value()
            
            # U(s,a): Exploration bonus
            # Tăng khi:
            # - Prior P(s,a) cao (policy network tin action này tốt)
            # - Parent được visit nhiều (có nhiều data để compare)
            # - Child ít được visit (chưa explore đủ)
            u_value = (c_puct * child.prior * 
                      math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            # PUCT score = exploitation + exploration
            ucb = q_value + u_value
            
            # Chọn child có UCB cao nhất
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child
    
    def expand(self, policy, hidden_state, model, game, use_chance=False):
        """
        ════════════════════════════════════════════════════════════════
        EXPANSION PHASE - Mở rộng node bằng cách tạo children
        ════════════════════════════════════════════════════════════════
        
        Expand behavior phụ thuộc vào node type:
        
        1. DECISION NODE (is_chance=False):
           Tạo child cho mỗi valid action:
           
           For each action a with P(a|s) > 0:
           - Dùng DYNAMICS model: (s,a) -> (s', r)
           - s' = next_hidden_state (learned representation)
           - r = predicted immediate reward
           - Tạo CHANCE node hoặc DECISION node tùy game
        
        2. CHANCE NODE (is_chance=True):
           Tạo child cho mỗi possible outcome:
           
           For each outcome o:
           - Dùng AFTERSTATE DYNAMICS: (afterstate, o) -> (s', r)
           - Tạo DECISION node với player tiếp theo
        
        So sánh với AlphaZero:
        - AlphaZero: next_state = game.step(state, action) [REAL ENV]
        - MuZero: next_hidden = model.dynamics(hidden, action) [LEARNED]
        
        Args:
            policy: Policy distribution [action_size] hoặc [chance_size]
                   - Decision node: π(a|s) từ policy network
                   - Chance node: p(o|s,a) từ chance predictor
            hidden_state: Hidden state hiện tại (torch.Tensor)
            model: MuZeroNetwork để inference
            game: Game object (CHỈ cho metadata!)
                 - game.action_size: số actions
                 - game.get_opponent(): đổi player
                 ⚠️ KHÔNG dùng game.step()! Dùng model.dynamics()!
            use_chance: Có dùng chance nodes không (cho stochastic games)
        """
        # ===== EXPAND DECISION NODE =====
        if not self.is_chance:
            for action in range(game.action_size):  # ← CHỈ dùng để lấy số actions!
                # Skip actions có probability = 0
                if policy[action] > 0:
                    # ════════════════════════════════════════════════
                    # ⚠️ KEY DIFFERENCE với AlphaZero:
                    # AlphaZero: next_state = game.step(state, action)
                    # MuZero:    next_hidden = model.dynamics(hidden, action)
                    # 
                    # MuZero KHÔNG biết game rules!
                    # Model TỰ HỌC cách predict next state & reward
                    # ════════════════════════════════════════════════
                    with torch.no_grad():
                        action_tensor = torch.tensor(
                            [action], 
                            device=hidden_state.device
                        )
                        
                        # g(s, a) -> (s', r)
                        # LEARNED DYNAMICS MODEL thay thế environment!
                        next_hidden_state, reward = model.dynamics(
                            hidden_state.unsqueeze(0),  # [1, C, H, W]
                            action_tensor                # [1]
                        )
                        
                        next_hidden_state = next_hidden_state.squeeze(0)  # [C, H, W]
                        
                        # Convert reward tensor to scalar
                        if isinstance(reward, torch.Tensor):
                            if reward.dim() > 1:
                                # Categorical reward: convert to scalar
                                reward = categorical_to_scalar(
                                    torch.softmax(reward, dim=-1),
                                    model.reward_support_range if hasattr(model, 'reward_support_range') else (-300, 301, 1)
                                )
                            reward = reward.item()
                    
                    # ===== TẠO CHILD NODE =====
                    if use_chance and hasattr(model, 'use_afterstate') and model.use_afterstate:
                        # Tạo CHANCE node (afterstate)
                        # Environment sẽ produce stochastic outcome
                        child = MuZeroNode(
                            hidden_state=next_hidden_state,
                            player=self.player,  # Chưa đổi player (afterstate)
                            prior=policy[action],
                            parent=self,
                            action_taken=action,
                            is_chance=True,  # Đây là chance node!
                            min_max_stats=self.min_max_stats
                        )
                    else:
                        # Tạo DECISION node (deterministic)
                        # Player kế tiếp sẽ chọn action
                        child = MuZeroNode(
                            hidden_state=next_hidden_state,
                            player=game.get_opponent(self.player),
                            prior=policy[action],
                            parent=self,
                            action_taken=action,
                            is_chance=False,
                            min_max_stats=self.min_max_stats
                        )
                    
                    child.reward = reward
                    self.children.append(child)
        
        # ===== EXPAND CHANCE NODE =====
        else:
            # Chance node expand: tạo child cho các possible outcomes
            for outcome_idx in range(len(policy)):
                if policy[outcome_idx] > 0:
                    # Dùng afterstate dynamics nếu có
                    with torch.no_grad():
                        if hasattr(model, 'afterstate_dynamics_network'):
                            # Create chance onehot [1, chance_space]
                            chance_onehot = torch.zeros(
                                1,
                                len(policy),
                                device=hidden_state.device
                            )
                            chance_onehot[0, outcome_idx] = 1.0
                            
                            # Afterstate dynamics: (afterstate, outcome) -> (s', r)
                            next_hidden_state, reward = model.afterstate_dynamics_network(
                                hidden_state.unsqueeze(0),
                                chance_onehot
                            )
                            
                            next_hidden_state = next_hidden_state.squeeze(0)
                            if isinstance(reward, torch.Tensor):
                                if reward.dim() > 1 and reward.shape[-1] > 1:
                                    reward = categorical_to_scalar(
                                        torch.softmax(reward, dim=-1),
                                        model.reward_support_range if hasattr(model, 'reward_support_range') else (-300, 301, 1)
                                    )
                                reward = reward.squeeze().item()
                        else:
                            # Fallback: dùng dynamics network thường
                            next_hidden_state = hidden_state
                            reward = 0.0
                    
                    # Tạo DECISION node cho player kế tiếp
                    child = MuZeroNode(
                        hidden_state=next_hidden_state,
                        player=game.get_opponent(self.player),
                        prior=policy[outcome_idx],
                        parent=self,
                        action_taken=outcome_idx,
                        is_chance=False,
                        min_max_stats=self.min_max_stats  # Back to decision node
                    )
                    child.reward = reward
                    self.children.append(child)
    
    def backpropagate(self, value, discount=0.997):
        """
        ════════════════════════════════════════════════════════════════
        BACKUP PHASE - Lan truyền value ngược lên tree
        ════════════════════════════════════════════════════════════════
        
        Sau khi evaluate leaf node, ta backpropagate value lên tree để
        update statistics của tất cả nodes trên path từ root đến leaf.
        
        MuZero Backup Formula:
        ----------------------
        Giá trị backup được tính theo n-step return:
        
        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·v_{t+n}
        
        Trong đó:
        - r_t: immediate reward tại step t (từ dynamics model)
        - γ (gamma): discount factor [0, 1]
        - v_{t+n}: value estimate tại leaf node (từ prediction network)
        
        Khác biệt với AlphaZero:
        -------------------------
        AlphaZero: G_t = v (chỉ backprop value)
        MuZero: G_t = r_t + γ·G_{t+1} (backprop value + rewards)
        
        Lý do: MuZero phải học rewards vì không có access vào env thật
        
        Two-Player Games:
        -----------------
        Trong 2-player zero-sum games, value được flip cho opponent:
        - Player 1's value = +v
        - Player 2's value = -v
        
        Chance Nodes:
        -------------
        Chance nodes KHÔNG flip value (không phải adversarial).
        Value được truyền trực tiếp từ child lên parent.
        
        Args:
            value: Value estimate từ leaf node [0, 1]
                   0 = thua, 0.5 = hòa, 1 = thắng
            discount: Discount factor γ cho rewards
        
        Update:
            - value_sum: tổng accumulated values
            - visit_count: số lần node được visit
        """
        # ===== UPDATE NODE STATISTICS =====
        self.value_sum += value
        self.visit_count += 1
        
        # ===== RECURSIVE BACKPROP TO PARENT =====
        if hasattr(self, "min_max_stats") and self.min_max_stats is not None:
            self.min_max_stats.update(self.value())

        if self.parent is not None:
            # Compute backed-up value cho parent:
            # G_parent = r_child + γ·G_child
            backed_up_value = self.reward + discount * value
            
            # ===== VALUE FLIPPING CHO OPPONENT =====
            # Trong 2-player zero-sum games:
            # - DECISION node: flip value (adversarial)
            # - CHANCE node: KHÔNG flip (stochastic, không adversarial)
            if not self.is_chance and self.parent.player != self.player:
                # Opponent's perspective: flip value
                # Player 1 thắng (+1) = Player 2 thua (-1)
                backed_up_value = -backed_up_value
            
            # Tiếp tục backprop lên parent
            self.parent.backpropagate(backed_up_value, discount)


class MuZeroMCTS:
    """
    ═══════════════════════════════════════════════════════════════════
    STOCHASTIC MUZERO MONTE CARLO TREE SEARCH
    ═══════════════════════════════════════════════════════════════════
    
    Hoàn toàn hoạt động trên LEARNED MODEL, không cần environment thật!
    
    Key Features:
    -------------
    1. ✅ Model-based planning: dùng dynamics model thay vì env thật
    2. ✅ Support stochastic environments: chance nodes
    3. ✅ Learned representations: hoạt động trong latent space
    4. ✅ Batch processing: efficient parallel inference
    
    MCTS Loop (4 phases):
    ----------------------
    
    Repeat num_searches times:
    
    1. SELECTION:
       - Bắt đầu từ root node
       - Chọn child theo UCB (decision) hoặc sample (chance)
       - Đi xuống cho đến leaf node (chưa expand)
    
    2. EXPANSION:
       - Dùng PREDICTION network: f(s) -> (π, v)
       - Tạo children cho leaf node với policy π
       - Dùng DYNAMICS model để predict next states
    
    3. EVALUATION:
       - Lấy value v từ prediction network
       - Đây là estimate của expected return từ leaf
    
    4. BACKUP:
       - Backpropagate value v lên tree
       - Update visit counts và value sums
       - Incorporate predicted rewards
    
    So sánh các loại MCTS:
    ----------------------
    
    ┌────────────┬──────────────┬─────────────┬────────────────┐
    │            │  AlphaZero   │  MuZero     │ Stochastic MZ  │
    ├────────────┼──────────────┼─────────────┼────────────────┤
    │ Env Access │   Required   │  Not needed │  Not needed    │
    │ Learned    │   Policy+V   │  P+V+Dyn+R  │  P+V+D+R+Ch    │
    │ Stochastic │      No      │     No      │     Yes        │
    │ Nodes      │   Decision   │  Decision   │  Dec + Chance  │
    │ Real World │      No      │  Possible   │  Best suited   │
    └────────────┴──────────────┴─────────────┴────────────────┘
    
    Hyperparameters:
    ----------------
    - num_searches: Số simulations (thường 50-800)
      + Nhiều hơn = better policy nhưng chậm hơn
      + Trade-off: quality vs speed
    
    - c_puct: Exploration constant (thường 1-2)
      + Cao hơn = explore nhiều hơn
      + Thấp hơn = exploit value estimates
    
    - dirichlet_noise: Add randomness ở root
      + Ensure exploration trong self-play
      + Prevent overfitting to current policy
    """
    def __init__(self, game, model, num_searches, c_puct=1.41, 
                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3,
                 discount=0.997, use_chance_nodes=False):
        """
        Khởi tạo Stochastic MuZero MCTS
        
        Args:
            game: Game object (CHỈ cho METADATA!)
                 ⚠️ QUAN TRỌNG: Game object ở đây CHỈ dùng cho:
                    - action_size: số actions có thể
                    - get_valid_moves(): mask invalid actions
                    - get_opponent(): đổi player (2-player games)
                    - get_current_player(): lấy player hiện tại
                 
                 ✅ MuZero KHÔNG dùng game.step() hay game rules!
                 ✅ Planning hoàn toàn dựa trên LEARNED MODEL
                 
            model: StochasticMuZeroNetwork (LEARNED MODEL)
                  - representation_network: h(o) -> s
                  - dynamics_network: g(s,a) -> (s', r)  ← Thay thế env.step()!
                  - prediction_network: f(s) -> (π, v)
                  - (optional) afterstate networks cho chance nodes
            
            num_searches: Số simulations mỗi lần search
                         - Mỗi simulation = 1 path từ root đến leaf
                         - Thường: 50-200 cho training, 800+ cho eval
            
            c_puct: PUCT exploration constant
                   - Controls exploration vs exploitation balance
                   - Thường: 1.25-2.0
                   - Formula: UCB = Q + c_puct * P * sqrt(N) / (1 + n)
            
            dirichlet_epsilon: Mixing weight cho Dirichlet noise
                              - Applied chỉ ở ROOT node
                              - π_root = (1-ε)·π + ε·Dir(α)
                              - Thường: 0.25
            
            dirichlet_alpha: Concentration parameter cho Dirichlet
                            - Lower = more concentrated (ít explore)
                            - Higher = more uniform (nhiều explore)
                            - Thường: 0.3 (Go), 0.15 (Chess), 0.03 (Shogi)
            
            discount: Discount factor γ cho rewards
                     - MuZero: G = r + γr' + γ²r'' + ... + γⁿv
                     - Thường: 0.997 (near 1.0 cho board games)
            
            use_chance_nodes: Enable stochastic MCTS với chance nodes
                             - True: support stochastic environments
                             - False: deterministic only (faster)
        """
        self.name = "Stochastic_MuZero_MCTS"
        
        # ===== Core Components =====
        self.game = game              # Game interface (metadata only)
        self.model = model            # MuZero neural networks
        
        # ===== MCTS Hyperparameters =====
        self.num_searches = num_searches      # Simulations per search
        self.c_puct = c_puct                  # Exploration constant
        self.discount = discount              # Reward discount factor γ
        
        # ===== Exploration Noise (Root Only) =====
        self.dirichlet_epsilon = dirichlet_epsilon  # Noise mixing weight
        self.dirichlet_alpha = dirichlet_alpha      # Dirichlet concentration
        
        # ===== Stochastic Support =====
        self.use_chance_nodes = use_chance_nodes    # Enable chance nodes?
    
    @torch.no_grad()
    def search(self, states, spGames):
        """
        ════════════════════════════════════════════════════════════════
        MAIN MCTS SEARCH FUNCTION
        ════════════════════════════════════════════════════════════════
        
        Thực hiện Monte Carlo Tree Search cho một batch states.
        Đây là core function kết nối tất cả components lại.
        
        Flow:
        -----
        1. INITIAL INFERENCE:
           - Encode observations → hidden states
           - Get initial policy & value predictions
           - Apply Dirichlet noise cho exploration
           - Create và expand root nodes
        
        2. SIMULATIONS (repeat num_searches times):
           For each simulation:
           a. SELECTION: Traverse tree từ root đến leaf
           b. EXPANSION: Expand leaf node với policy
           c. EVALUATION: Get value từ prediction network  
           d. BACKUP: Backpropagate value lên tree
        
        3. RETURN:
           - Updated root nodes với visit statistics
           - Action probabilities từ visit counts
        
        Batch Processing:
        -----------------
        - Search cho nhiều games đồng thời
        - Efficient: batch inference cho leaf nodes
        - Speeds up self-play significantly
        
        Args:
            states: List of game states [observations]
                   - Observations THẬT từ environment
                   - Sẽ được encode thành hidden states
                   - Length = batch_size
            
            spGames: List of SPG (Self-Play Game) objects
                    - Mỗi game có attribute .root cho MCTS tree
                    - Length = batch_size
                    - Will be updated in-place
        
        Returns:
            None (updates spGames[i].root in-place)
        """
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: INITIAL INFERENCE
        # ═══════════════════════════════════════════════════════════════
        # 
        # Chuyển observations thật (s) → hidden states (h)
        # Sử dụng REPRESENTATION NETWORK: h = h(o)
        # 
        # Lấy initial predictions:
        # - Policy π(a|s): probability distribution over actions  
        # - Value v(s): expected return from this state
        #
        # Đây là ĐIỂM KHỞI ĐẦU của tree search
        # ═══════════════════════════════════════════════════════════════
        
        # ===== ENCODE STATES =====
        # States từ environment thường là objects (pyspiel, gym, etc.)
        # Cần encode thành numpy arrays trước khi pass vào model
        # Game object CHỈ dùng ở đây để encode observations!
        encoded_states = self.game.get_encoded_state(states)
        
        # Model inference với encoded observations
        with torch.no_grad():
            inference_outputs = self.model.initial_inference(encoded_states)
        if isinstance(inference_outputs, (list, tuple)):
            if len(inference_outputs) == 5:
                hidden_states, policy_logits, values, _, _ = inference_outputs
            elif len(inference_outputs) == 4:
                hidden_states, policy_logits, values, _ = inference_outputs
            else:
                hidden_states, policy_logits, values = inference_outputs
        else:
            raise ValueError("Unexpected output from initial_inference")
        
        # ===== Convert Logits → Probabilities =====
        # Softmax: logits → valid probability distribution
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        
        # ═══════════════════════════════════════════════════════════════
        # DIRICHLET NOISE - Thêm Exploration Noise ở Root
        # ═══════════════════════════════════════════════════════════════
        # 
        # Tại sao cần noise?
        # - Prevent overfitting vào current policy
        # - Ensure đủ exploration trong self-play
        # - Giúp discover new strategies
        # 
        # Formula: π_root = (1-ε)·π_network + ε·Dir(α)
        # 
        # Trong đó:
        # - π_network: policy từ neural network
        # - Dir(α): Dirichlet distribution với concentration α
        # - ε: mixing weight (thường 0.25)
        # 
        # QUAN TRỌNG: Chỉ áp dụng ở ROOT, không ở internal nodes!
        # ═══════════════════════════════════════════════════════════════
        
        policies = (1 - self.dirichlet_epsilon) * policies + \
                   self.dirichlet_epsilon * np.random.dirichlet(
                       [self.dirichlet_alpha] * self.game.action_size, 
                       size=policies.shape[0]
                   )
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE ROOT NODES
        # ═══════════════════════════════════════════════════════════════
        # 
        # Tạo root node cho mỗi game trong batch
        # Root node = starting point của tree search
        # ═══════════════════════════════════════════════════════════════
        
        for i, spg in enumerate(spGames):
            spg_policy = policies[i]
            if isinstance(hidden_states, torch.Tensor):
                root_hidden_state = hidden_states[i].detach()
            else:
                root_hidden_state = torch.tensor(hidden_states[i], dtype=torch.float32, device=self.model.device)
            root_hidden_state = root_hidden_state.to(self.model.device)
            
            # ===== MASK INVALID MOVES =====
            # Game có thể có các actions không hợp lệ ở state này
            # Ví dụ: trong Connect4, cột đã đầy thì không chọn được
            valid_moves = self.game.get_valid_moves(states[i])
            valid_mask = np.array([1 if j in valid_moves else 0 
                                   for j in range(self.game.action_size)])
            
            # Apply mask: set probability = 0 cho invalid actions
            spg_policy *= valid_mask
            
            # Renormalize (đảm bảo sum = 1)
            if np.sum(spg_policy) == 0:
                # Edge case: tất cả actions invalid → uniform over valid
                spg_policy = valid_mask / np.sum(valid_mask)
            else:
                spg_policy /= np.sum(spg_policy)
            
            # ===== CREATE ROOT NODE =====
            # Root luôn là DECISION node (player's turn)
            spg.root = MuZeroNode(
                hidden_state=root_hidden_state,
                player=self.game.get_current_player(states[i]),
                is_chance=False  # Root luôn là decision node
            )
            
            # ===== EXPAND ROOT NODE =====
            # Tạo children cho tất cả valid actions
            # Mỗi child = một possible action từ root
            spg.root.expand(
                policy=spg_policy,
                hidden_state=root_hidden_state,
                model=self.model,
                game=self.game,
                use_chance=self.use_chance_nodes
            )
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: SIMULATION LOOP
        # ═══════════════════════════════════════════════════════════════
        # 
        # Thực hiện num_searches simulations để build search tree
        # Mỗi simulation gồm 4 phases: Selection → Expansion → Evaluation → Backup
        # ═══════════════════════════════════════════════════════════════
        
        for search in range(self.num_searches):
            # ═══════════════════════════════════════════════════════════
            # PHASE 2a: SELECTION
            # ═══════════════════════════════════════════════════════════
            # 
            # Traverse tree từ root xuống leaf node (chưa expand)
            # 
            # Decision node: chọn child có UCB cao nhất
            # Chance node: sample child theo probability
            # 
            # Dừng khi gặp leaf (node chưa expand)
            # ═══════════════════════════════════════════════════════════
            
            for spg in spGames:
                spg.node = None  # Reset node pointer
                node = spg.root
                
                # Traverse xuống tree cho đến leaf
                while node.is_expanded():
                    # select_child() tự động handle:
                    # - UCB cho decision nodes
                    # - Sampling cho chance nodes
                    node = node.select_child(self.c_puct)
                
                # Lưu leaf node để expand sau
                spg.node = node
            
            # ═══════════════════════════════════════════════════════════
            # PHASE 2b: EXPANSION & EVALUATION
            # ═══════════════════════════════════════════════════════════
            # 
            # Expand tất cả leaf nodes cùng lúc (batch processing)
            # 
            # Steps:
            # 1. Collect all leaf nodes' hidden states
            # 2. Batch inference: f(s) → (π, v)
            # 3. Expand each leaf với policy π
            # 4. Evaluate each leaf với value v
            # ═══════════════════════════════════════════════════════════
            
            # Collect expandable games (có leaf node)
            expandable_spGames = [
                i for i in range(len(spGames)) 
                if spGames[i].node is not None
            ]
            
            if len(expandable_spGames) > 0:
                node_policies = {}
                node_values = {}

                with torch.no_grad():
                    decision_entries = []  # (idx, hidden_state_tensor)
                    chance_entries = []

                    for idx in expandable_spGames:
                        node = spGames[idx].node
                        hidden_state = node.hidden_state
                        if isinstance(hidden_state, torch.Tensor):
                            hidden_tensor = hidden_state.detach().to(self.model.device)
                        else:
                            hidden_tensor = torch.tensor(hidden_state, dtype=torch.float32, device=self.model.device)

                        if node.is_chance and self.use_chance_nodes and getattr(self.model, 'use_afterstate', False):
                            chance_entries.append((idx, hidden_tensor))
                        else:
                            decision_entries.append((idx, hidden_tensor))

                    # Decision nodes: use policy/value head
                    if decision_entries:
                        decision_batch = torch.stack([entry[1] for entry in decision_entries])
                        policy_logits, values = self.model.prediction(decision_batch)
                        decision_policies = torch.softmax(policy_logits, dim=1).cpu().numpy()

                        if values.dim() > 1 and values.shape[-1] > 1:
                            values = categorical_to_scalar(
                                torch.softmax(values, dim=-1),
                                getattr(self.model, 'value_support_range', (-300, 301, 1))
                            )
                        decision_values = values.detach().cpu().numpy()

                        for i, (idx, _) in enumerate(decision_entries):
                            value_scalar = decision_values[i][0] if decision_values.ndim > 1 else decision_values[i]
                            node_policies[idx] = decision_policies[i]
                            node_values[idx] = float(value_scalar)

                    # Chance nodes: use afterstate prediction
                    if chance_entries:
                        chance_batch = torch.stack([entry[1] for entry in chance_entries])
                        chance_policy_logits, chance_value_logits = self.model.afterstate_prediction(chance_batch)
                        chance_policies = torch.softmax(chance_policy_logits, dim=1).cpu().numpy()

                        if chance_value_logits.dim() > 1 and chance_value_logits.shape[-1] > 1:
                            chance_value_logits = categorical_to_scalar(
                                torch.softmax(chance_value_logits, dim=-1),
                                getattr(self.model, 'value_support_range', (-300, 301, 1))
                            )
                        chance_values = chance_value_logits.detach().cpu().numpy()

                        for i, (idx, _) in enumerate(chance_entries):
                            value_scalar = chance_values[i][0] if chance_values.ndim > 1 else chance_values[i]
                            node_policies[idx] = chance_policies[i]
                            node_values[idx] = float(value_scalar)

                # ===== EXPAND & BACKUP =====
                for idx in expandable_spGames:
                    node = spGames[idx].node
                    spg_policy = node_policies.get(idx)
                    spg_value = node_values.get(idx, 0.0)

                    if spg_policy is None:
                        # Safety: skip if no policy (should not happen)
                        continue

                    # Ensure numpy array for downstream ops
                    spg_policy = np.asarray(spg_policy, dtype=np.float32)
                    if node.is_chance:
                        total_prob = np.sum(spg_policy)
                        if total_prob > 0:
                            spg_policy = spg_policy / total_prob
                    
                    # Normalize value về [0, 1] nếu cần
                    if spg_value < 0 or spg_value > 1:
                        spg_value = (spg_value + 1) / 2
                    spg_value = float(np.clip(spg_value, 0.0, 1.0))

                    node.expand(
                        policy=spg_policy,
                        hidden_state=node.hidden_state,
                        model=self.model,
                        game=self.game,
                        use_chance=self.use_chance_nodes
                    )

                    node.backpropagate(spg_value, discount=self.discount)
    
    def get_action_probs(self, root, temperature=1.0):
        """
        ════════════════════════════════════════════════════════════════
        EXTRACT ACTION PROBABILITIES từ Search Tree
        ════════════════════════════════════════════════════════════════
        
        Sau khi search xong, ta có visit counts cho mỗi action.
        Visit count cao = action được explore nhiều = likely tốt hơn.
        
        Convert visit counts → action probabilities:
        
        π(a) ∝ N(s,a)^(1/τ)
        
        Trong đó:
        - N(s,a): visit count của action a
        - τ (tau): temperature parameter
        
        Temperature Effects:
        --------------------
        
        τ → 0:  Deterministic (greedy)
                Chọn action có visit count cao nhất
                π(best_action) = 1, π(others) = 0
                Use: Evaluation, competitive play
        
        τ = 1:  Proportional to visit counts
                π(a) ∝ N(a)
                Balanced exploration-exploitation
                Use: Training (giữ diversity)
        
        τ > 1:  More uniform (more exploration)
                Làm mượt distribution
                Use: Early game, high exploration
        
        Args:
            root: Root node sau khi search
                 - Có visit statistics ở children
                 
            temperature: Temperature τ ∈ [0, ∞)
                        - 0: greedy (best action)
                        - 1: proportional (default)
                        - >1: more uniform
        
        Returns:
            action_probs: Probability distribution [action_size]
                         - Sum = 1.0
                         - Higher prob = better action
        """
        # Initialize probability array
        action_probs = np.zeros(self.game.action_size)
        
        # ===== COLLECT VISIT COUNTS =====
        # Mỗi child = một action đã được expand
        # Visit count = số lần action được explore trong MCTS
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        # ===== APPLY TEMPERATURE =====
        if temperature == 0:
            # ═══════════════════════════════════════════════════════════
            # GREEDY MODE (τ = 0)
            # ═══════════════════════════════════════════════════════════
            # 
            # Chọn action có visit count cao nhất
            # Deterministic: không có randomness
            # 
            # Use case:
            # - Evaluation games
            # - Competitive play
            # - Khi muốn best move (không explore)
            # ═══════════════════════════════════════════════════════════
            action = np.argmax(action_probs)
            action_probs = np.zeros(self.game.action_size)
            action_probs[action] = 1.0
        else:
            # ═══════════════════════════════════════════════════════════
            # TEMPERATURE SAMPLING (τ > 0)
            # ═══════════════════════════════════════════════════════════
            # 
            # Formula: π(a) ∝ N(a)^(1/τ)
            # 
            # Effect của temperature:
            # - τ < 1: Sharp distribution (favor best actions more)
            # - τ = 1: Linear với visit counts
            # - τ > 1: Smooth distribution (more uniform)
            # 
            # Use case:
            # - Self-play training (maintain diversity)
            # - Exploration phase
            # ═══════════════════════════════════════════════════════════
            action_probs = action_probs ** (1 / temperature)
            
            # Normalize (sum = 1)
            if np.sum(action_probs) > 0:
                action_probs /= np.sum(action_probs)
            else:
                # Edge case: no visits (shouldn't happen)
                action_probs = np.ones(self.game.action_size) / self.game.action_size
        
        return action_probs



