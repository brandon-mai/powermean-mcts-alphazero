"""
Quick MuZero Training - Cấu hình cực nhỏ để test nhanh
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from games import ConnectFour
from muzero.model import MuZeroNetwork, weights_init_normal
from muzero import MuZero
from mcts import MuZeroMCTS


def train_muzero_quick():
    """
    Training cực nhỏ - CHỈ ĐỂ TEST
    
    Thông số tối thiểu:
    - 2 iterations
    - 20 self-play games per iteration
    - 50 MCTS searches (thay vì 100)
    - 10 parallel games
    - 2 epochs
    - Batch size 32
    """
    print("="*80)
    print("🧪 QUICK MUZERO TEST - CẤU HÌNH TỐI THIỂU")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Game
    game = ConnectFour()
    
    # Model: MuZero network - cấu hình nhỏ
    observation_shape = (game.num_planes, game.row_count, game.column_count)
    
    model = MuZeroNetwork(
        observation_shape=observation_shape,  # (3, 6, 7) cho Connect4
        action_space_size=game.action_size,   # 7 columns
        num_res_blocks=2,      # CHỈ 2 ResBlocks (thay vì 4)
        num_channels=32,       # 32 channels (thay vì 64)
        # Connect4 là DETERMINISTIC - không cần chance/afterstate
        use_chance_encoder=False,
        use_afterstate=False,
        use_categorical=True,  # Dùng categorical distribution
        reward_support_range=(-10., 11., 1.),  # Nhỏ hơn cho Connect4
        value_support_range=(-10., 11., 1.),
        device=device
    )
    model.apply(weights_init_normal)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # MCTS - searches CỰC ÍT
    mcts = MuZeroMCTS(
        game=game,
        model=model,
        num_searches=25,       # CHỈ 25 searches (rất ít!)
        c_puct=1.41,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        use_chance_nodes=False  # Deterministic game
    )
    
    # Trainer - thông số CỰC TỐI THIỂU
    trainer = MuZero(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.002),
        game=game,
        mcts=mcts,
        num_parallel_games=5,         # CHỈ 5 games parallel (rất ít!)
        temperature=1.0,              # Temperature thấp hơn
        batch_size=16,                # Batch size nhỏ hơn
        num_iterations=1,             # CHỈ 1 iteration!
        num_selfPlay_iterations=5,    # CHỈ 5 games total (cực ít!)
        num_epochs=1,                 # CHỈ 1 epoch!
        num_unroll_steps=2,           # K=2 (tối thiểu)
        td_steps=3,                   # n=3 (tối thiểu)
        discount=0.99
    )
    
    print("\n🎯 Cấu hình training:")
    print(f"   - Iterations: 1")
    print(f"   - Self-play games: 5 total")
    print(f"   - MCTS searches: 25")
    print(f"   - Parallel games: 5")
    print(f"   - Training epochs: 1")
    print(f"   - Batch size: 16")
    print(f"   - Unroll steps: 2")
    print(f"   - TD steps: 3")
    print("="*80)
    
    # Start training
    print("\n🚀 Bắt đầu training...")
    trainer.learn()
    
    print("\n✅ Quick test hoàn thành!")
    print(f"💾 Checkpoint saved in: checkpoint/")
    
    return model, game, mcts


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK TEST MODE - Training với thông số tối thiểu")
    print("Mục đích: Kiểm tra code chạy đúng, KHÔNG optimize performance")
    print("="*80 + "\n")
    
    model, game, mcts = train_muzero_quick()
    
    print("\n" + "="*80)
    print("📊 NEXT STEPS:")
    print("="*80)
    print("\n1. Kiểm tra checkpoint:")
    print("   ls -lh checkpoint/Stochastic_MuZero_MCTS_ConnectFour_iteration_*.pt")
    print("\n2. Test model với evaluation:")
    print("   python3 evaluate/tournament.py \\")
    print("       --checkpoint1 checkpoint/Stochastic_MuZero_MCTS_ConnectFour_iteration_1.pt \\")
    print("       --checkpoint2 checkpoint/Stochastic_MuZero_MCTS_ConnectFour_iteration_2.pt \\")
    print("       --num_games 10")
    print("\n3. Play against AI:")
    print("   python3 evaluate/human_vs_alphazero.py \\")
    print("       --checkpoint checkpoint/Stochastic_MuZero_MCTS_ConnectFour_iteration_2.pt")
    print("\n" + "="*80)

