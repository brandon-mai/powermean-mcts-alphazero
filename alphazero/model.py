import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class PongAtariResNet(nn.Module):
    """
    ResNet adapted for large images like PongAtari (210x160).
    Uses adaptive pooling to reduce spatial dimensions before policy/value heads.
    """
    def __init__(self, game, num_resBlocks=4, num_hidden=128, device='cpu'):
        super().__init__()
        self.game = game
        self.device = device
        
        # Start block: 3 -> num_hidden
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Residual blocks
        self.backBone = nn.ModuleList([
            ResBlock(num_hidden) for i in range(num_resBlocks)
        ])
        
        # Adaptive pooling to reduce to manageable size
        # e.g., (210, 160) -> (8, 8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Policy head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, game.action_size)  # Fixed size: 32 * 64 = 2048
        )
        
        # Value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 1),  # Fixed size: 3 * 64 = 192
            nn.Tanh()
        )
        
        self.to(device)
    
    def forward(self, state):
        """
        Forward pass - handles both inference (single state) and training (batch).
        
        For inference (MCTS): state is numpy array (210, 160, 3)
        For training: state is tensor batch (B, 3, 210, 160)
        """
        # Check if input is already a tensor (training mode)
        if isinstance(state, torch.Tensor):
            # Training mode: state is already encoded batch (B, 3, 210, 160)
            x = state
        else:
            # Inference mode: encode single state and add batch dimension
            encoded_state = self.game.get_encoded_state(state)
            x = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Start block
        x = self.startBlock(x)
        
        # Residual blocks
        for resBlock in self.backBone:
            x = resBlock(x)
        
        # Adaptive pooling to reduce size
        x = self.adaptive_pool(x)
        
        # Get policy and value
        policy_logits = self.policyHead(x)
        value = self.valueHead(x)
        
        # Return format depends on mode
        if isinstance(state, torch.Tensor):
            # Training mode: return tensors for batch
            return policy_logits, value
        else:
            # Inference mode: return single values
            value = value.item()
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().tolist()
            return value, policy
