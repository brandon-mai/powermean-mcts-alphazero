"""
Ví dụ: Training Stochastic MuZero cho Connect Four

File này minh họa cách train MuZero từ đầu với các cấu hình khác nhau.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from games import ConnectFour
from muzero.model import MuZeroNetwork, weights_init_normal
from muzero import MuZero
from mcts import MuZeroMCTS


def train_muzero_basic():
    """
    Training cơ bản với default parameters
    
    Phù hợp cho:
    - Test nhanh
    - Debug
    - Làm quen với MuZero
    """
    print("="*80)
    print("🎮 TRAINING MUZERO - CẤU HÌNH CƠ BẢN")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Game
    game = ConnectFour()
    
    # Model: MuZero với 3 networks
    # MuZero KHÔNG cần game object - chỉ cần observation shape và action space!
    observation_shape = (game.num_planes, game.row_count, game.column_count)
    
    model = MuZeroNetwork(
        observation_shape=observation_shape,  # (3, 6, 7) cho Connect4
        action_space_size=game.action_size,   # 7 columns
        num_res_blocks=4,      # Ít ResBlocks vì có 3 networks
        num_channels=64,       # num_hidden -> num_channels
        # Connect4 là DETERMINISTIC game - không cần chance encoder!
        use_chance_encoder=False,
        use_afterstate=False,
        device=device
    )
    model.apply(weights_init_normal)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # MCTS
    mcts = MuZeroMCTS(
        game=game,
        model=model,
        num_searches=100,     # Ít searches để train nhanh
        c_puct=1.41,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    
    # Trainer
    trainer = MuZero(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        game=game,
        mcts=mcts,
        num_parallel_games=50,        # Ít games
        temperature=1.25,
        batch_size=64,                # Batch size nhỏ
        num_iterations=3,             # Chỉ 3 iterations
        num_selfPlay_iterations=100,  # Ít games
        num_epochs=3,
        num_unroll_steps=3,           # K=3
        td_steps=5,                   # n=5
        discount=0.99
    )
    
    # Start training
    trainer.learn()
    
    print("\n✅ Training hoàn thành!")


def train_muzero_advanced():
    """
    Training với cấu hình mạnh hơn
    
    Phù hợp cho:
    - Training serious
    - Đạt performance cao
    - Research
    """
    print("="*80)
    print("🚀 TRAINING MUZERO - CẤU HÌNH NÂNG CAO")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Game
    game = ConnectFour()
    
    # Model: Larger network
    observation_shape = (game.num_planes, game.row_count, game.column_count)
    
    model = MuZeroNetwork(
        observation_shape=observation_shape,
        action_space_size=game.action_size,
        num_res_blocks=6,      # Nhiều ResBlocks hơn
        num_channels=128,      # Nhiều channels
        use_chance_encoder=False,
        use_afterstate=False,
        use_categorical=True,
        reward_support_range=(-10., 11., 1.),
        value_support_range=(-10., 11., 1.),
        device=device
    )
    model.apply(weights_init_normal)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # MCTS: More searches
    mcts = MuZeroMCTS(
        game=game,
        model=model,
        num_searches=600,     # Nhiều searches
        c_puct=1.41,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    
    # Trainer
    trainer = MuZero(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001),
        game=game,
        mcts=mcts,
        num_parallel_games=100,       # Nhiều parallel games
        temperature=1.25,
        batch_size=128,
        num_iterations=10,            # 10 iterations
        num_selfPlay_iterations=500,  # Nhiều games
        num_epochs=5,
        num_unroll_steps=5,           # K=5
        td_steps=10,                  # n=10
        discount=0.997                # Discount cao cho long-term
    )
    
    # Start training
    trainer.learn()
    
    print("\n✅ Training hoàn thành!")


def continue_training_from_checkpoint():
    """
    Continue training từ một checkpoint có sẵn
    
    Hữu ích khi:
    - Training bị gián đoạn
    - Muốn train thêm iterations
    - Fine-tuning
    """
    print("="*80)
    print("🔄 CONTINUE TRAINING TỪ CHECKPOINT")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Game
    game = ConnectFour()
    
    # Model
    observation_shape = (game.num_planes, game.row_count, game.column_count)
    
    model = MuZeroNetwork(
        observation_shape=observation_shape,
        action_space_size=game.action_size,
        num_res_blocks=6,
        num_channels=128,
        use_chance_encoder=False,
        use_afterstate=False,
        use_categorical=True,
        reward_support_range=(-10., 11., 1.),
        value_support_range=(-10., 11., 1.),
        device=device
    )
    
    # Load checkpoint
    checkpoint_path = "checkpoint/MuZero_MCTS_ConnectFour_iteration_5.pt"
    if os.path.exists(checkpoint_path):
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"⚠️  Checkpoint không tồn tại: {checkpoint_path}")
        print("Khởi tạo model mới...")
        model.apply(weights_init_normal)
    
    # MCTS
    mcts = MuZeroMCTS(
        game=game,
        model=model,
        num_searches=600,
        c_puct=1.41,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3
    )
    
    # Trainer
    trainer = MuZero(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0005),  # Lower LR
        game=game,
        mcts=mcts,
        num_parallel_games=100,
        temperature=1.41,
        batch_size=128,
        num_iterations=5,            # Train thêm 5 iterations
        num_selfPlay_iterations=500,
        num_epochs=5,
        num_unroll_steps=5,
        td_steps=10,
        discount=0.997
    )
    
    # Continue training
    trainer.learn()
    
    print("\n✅ Training hoàn thành!")


def compare_hyperparameters():
    """
    So sánh ảnh hưởng của các hyperparameters
    
    Test các cấu hình:
    1. Unroll steps (K)
    2. TD steps (n)
    3. Latent dimension
    """
    print("="*80)
    print("🔬 SO SÁNH HYPERPARAMETERS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = ConnectFour()
    
    configs = [
        # Config 1: Small K, small n
        {
            'name': 'Small Unroll',
            'num_unroll_steps': 3,
            'td_steps': 5,
            'latent_dim': 16
        },
        # Config 2: Large K, large n
        {
            'name': 'Large Unroll',
            'num_unroll_steps': 7,
            'td_steps': 15,
            'latent_dim': 32
        },
        # Config 3: Balanced
        {
            'name': 'Balanced',
            'num_unroll_steps': 5,
            'td_steps': 10,
            'latent_dim': 24
        }
    ]
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"📊 Testing config: {config['name']}")
        print(f"   - Unroll steps: {config['num_unroll_steps']}")
        print(f"   - TD steps: {config['td_steps']}")
        print(f"   - Latent dim: {config['latent_dim']}")
        print(f"{'='*80}")
        
        # Model
        observation_shape = (game.num_planes, game.row_count, game.column_count)
        
        model = MuZeroNetwork(
            observation_shape=observation_shape,
            action_space_size=game.action_size,
            num_res_blocks=4,
            num_channels=64,
            use_chance_encoder=False,
            use_afterstate=False,
            use_categorical=True,
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            device=device
        )
        model.apply(weights_init_normal)
        
        # MCTS
        mcts = MuZeroMCTS(
            game=game,
            model=model,
            num_searches=100,
            c_puct=1.41,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3
        )
        
        # Trainer
        trainer = MuZero(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            game=game,
            mcts=mcts,
            num_parallel_games=50,
            temperature=1.25,
            batch_size=64,
            num_iterations=2,
            num_selfPlay_iterations=100,
            num_epochs=3,
            num_unroll_steps=config['num_unroll_steps'],
            td_steps=config['td_steps'],
            discount=0.99
        )
        
        # Train
        trainer.learn()
        
        print(f"\n✅ Config '{config['name']}' hoàn thành!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MuZero Training Examples")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['basic', 'advanced', 'continue', 'compare'],
        default='basic',
        help="""
        Chọn mode training:
        - basic: Cấu hình cơ bản, train nhanh
        - advanced: Cấu hình mạnh, performance cao
        - continue: Continue từ checkpoint
        - compare: So sánh hyperparameters
        """
    )
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        train_muzero_basic()
    elif args.mode == 'advanced':
        train_muzero_advanced()
    elif args.mode == 'continue':
        continue_training_from_checkpoint()
    elif args.mode == 'compare':
        compare_hyperparameters()

