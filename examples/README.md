# 📚 Examples - Ví Dụ Sử Dụng

Thư mục này chứa các ví dụ cách sử dụng các algorithms trong project.

## 🚀 Quick Start

### 1. Training MuZero Cơ Bản

```bash
cd examples
python train_muzero.py --mode basic
```

**Mô tả**: Training nhanh với cấu hình nhẹ để test và làm quen.

**Thời gian**: ~10-15 phút (CPU), ~3-5 phút (GPU)

---

### 2. Training MuZero Nâng Cao

```bash
python train_muzero.py --mode advanced
```

**Mô tả**: Training serious với cấu hình mạnh để đạt performance cao.

**Thời gian**: ~2-3 giờ (CPU), ~30-45 phút (GPU)

**Khuyến nghị**: Dùng GPU và để chạy qua đêm.

---

### 3. Continue Training từ Checkpoint

```bash
python train_muzero.py --mode continue
```

**Mô tả**: Tiếp tục training từ một checkpoint có sẵn.

**Use case**:
- Training bị gián đoạn
- Muốn train thêm iterations
- Fine-tuning model

---

### 4. So Sánh Hyperparameters

```bash
python train_muzero.py --mode compare
```

**Mô tả**: Test nhiều cấu hình hyperparameters khác nhau để tìm best config.

**Thời gian**: ~30-45 phút (CPU), ~10-15 phút (GPU)

---

## 🎮 Training với Main Script

Bạn cũng có thể dùng `train.py` ở root directory:

### AlphaZero với PUCT

```bash
python train.py \
    --algorithm PUCT \
    --game ConnectFour \
    --num_iterations 10 \
    --num_selfPlay_iterations 500 \
    --num_searches 600
```

### AlphaZero với Stochastic Powermean UCT

```bash
python train.py \
    --algorithm Stochastic_Powermean_UCT \
    --game ConnectFour \
    --num_iterations 10 \
    --num_selfPlay_iterations 500 \
    --num_searches 600 \
    --p 1.5 \
    --gamma 0.95
```

### Stochastic MuZero

```bash
python train.py \
    --algorithm MuZero \
    --game ConnectFour \
    --num_iterations 10 \
    --num_selfPlay_iterations 500 \
    --num_searches 600 \
    --num_unroll_steps 5 \
    --td_steps 10 \
    --discount 0.997
```

---

## 📊 Hyperparameters Guide

### MuZero Specific

| Parameter | Small | Medium | Large | Ý nghĩa |
|-----------|-------|--------|-------|---------|
| `num_unroll_steps` | 3 | 5 | 7 | Số steps unroll dynamics |
| `td_steps` | 5 | 10 | 15 | Số steps cho n-step returns |
| `latent_dim` | 16 | 32 | 64 | Dimension của latent variable |
| `num_resBlocks` | 4 | 6 | 9 | Số Residual Blocks |
| `num_hidden` | 64 | 128 | 256 | Số hidden channels |

### Training Speed vs Quality

**Fast Training** (để test):
```bash
--num_parallel_games 50
--num_iterations 3
--num_selfPlay_iterations 100
--num_searches 100
--batch_size 64
```

**Balanced** (khuyến nghị):
```bash
--num_parallel_games 100
--num_iterations 10
--num_selfPlay_iterations 500
--num_searches 600
--batch_size 128
```

**High Quality** (research):
```bash
--num_parallel_games 200
--num_iterations 20
--num_selfPlay_iterations 1000
--num_searches 1000
--batch_size 256
```

---

## 🔍 Monitoring Training

### Check Checkpoints

Checkpoints được lưu trong folder `checkpoint/`:

```
checkpoint/
├── MuZero_MCTS_ConnectFour_iteration_1.pt
├── MuZero_MCTS_ConnectFour_iteration_2.pt
└── ...
```

### Load và Evaluate

```python
import torch
from muzero.model import MuZeroNetwork
from games import ConnectFour

game = ConnectFour()
model = MuZeroNetwork(game, num_resBlocks=6, num_hidden=128, latent_dim=32, device='cpu')
model.load_state_dict(torch.load('checkpoint/MuZero_MCTS_ConnectFour_iteration_10.pt'))
model.eval()

# Evaluate
# ...
```

---

## 💡 Tips

### 1. **Start Small**
- Test với cấu hình nhỏ trước
- Verify code chạy đúng
- Sau đó scale up

### 2. **GPU Acceleration**
- MuZero rất compute-intensive
- Khuyến nghị dùng GPU
- CPU chạy được nhưng rất chậm

### 3. **Checkpoint Often**
- Training có thể mất nhiều giờ
- Checkpoints tự động save mỗi iteration
- Có thể continue từ bất kỳ checkpoint nào

### 4. **Monitor Losses**
- Policy loss: nên giảm dần
- Value loss: nên giảm dần
- Reward loss: quan trọng cho dynamics model
- Nếu loss không giảm → adjust hyperparameters

### 5. **Hyperparameter Tuning**
- `td_steps` nên lớn hơn `num_unroll_steps`
- `discount` gần 1.0 cho long-term planning
- `latent_dim` càng lớn → model càng expressive nhưng cũng overfitting dễ hơn

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Giảm**:
- `batch_size`
- `num_hidden`
- `num_resBlocks`
- `num_parallel_games`

### Training Too Slow

**Giảm**:
- `num_searches`
- `num_selfPlay_iterations`
- `num_unroll_steps`

### Model Not Learning

**Check**:
- Learning rate có phù hợp? (thử 0.001 hoặc 0.0001)
- Losses có NaN? → gradient clipping
- MCTS có đủ searches? (ít nhất 100)
- Training đủ lâu? (ít nhất 5 iterations)

---

## 📬 Questions?

Nếu gặp vấn đề, check:
1. **STOCHASTIC_MUZERO_EXPLAINED.md** - giải thích chi tiết
2. Code comments - mỗi phần đều có giải thích
3. GitHub issues

**Happy Learning!** 🎉

