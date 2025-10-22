import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from concurrent.futures import ThreadPoolExecutor

class MockNeuralNetwork(nn.Module):
    def __init__(self, game, device, num_workers=100, num_rollout=1000):
        super().__init__()
        self.game = game
        self.device = device
        self.num_workers = num_workers
        self.num_rollout = num_rollout
    
    def forward(self, states):
        value = self.simulate(states)
        policy = [[1.0 for i in range(self.game.action_size)] for _ in range(len(states))]
        return value, policy
    
    def _single_rollout(self, state):
        rollout_state = state.copy()
        rollout_player = -1
        while True:
            action = np.random.choice(self.game.get_valid_moves(rollout_state))
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            rollout_player = self.game.get_opponent(rollout_player)
    
    def simulate(self, states):
        values = []
        for state in states:
            value, is_terminal = self.game.get_value_and_terminated(
                state=state, 
                player=1)
            
            if is_terminal:
                values.appen(value)
                continue
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._single_rollout, state) for _ in range(self.num_rollout)]
                rollout_values = [future.result() for future in futures]
            values.append(np.mean(rollout_values))
            
        return values
    
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

    def forward(self, states):
        x = torch.tensor(self.game.get_encoded_state(states), device=self.device)

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
    
    def forward(self, states):
        """
        Forward pass - handles both inference (single state) and training (batch).
        
        For inference (MCTS): state is numpy array (210, 160, 3)
        For training: state is tensor batch (B, 3, 210, 160)
        """
        # Check if input is already a tensor (training mode)
        if isinstance(states, torch.Tensor):
            # Training mode: state is already encoded batch (B, 3, 210, 160)
            x = states
        else:
            # Inference mode: encode single state and add batch dimension
            encoded_state = self.game.get_encoded_state(states)
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
        if isinstance(states, torch.Tensor):
            # Training mode: return tensors for batch
            return policy_logits, value
        else:
            # Inference mode: return single values
            value = value.item()
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().tolist()
            return value, policy


def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


def print_weight_stats(model, name):
    print(f"--- {name} ---")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n}: mean={p.data.mean():.4f}, std={p.data.std():.4f}, shape={tuple(p.shape)}")


if __name__ == "__main__":
    class DummyGame:
        row_count = 6
        column_count = 7
        action_size = 7
        def get_encoded_state(self, state):
            return torch.zeros(6, 7)

    device = 'cpu'
    game = DummyGame()

    # Test ResNet
    resnet = ResNet(game, num_resBlocks=9, num_hidden=128, device=device)
    resnet.apply(weights_init_normal)
    print_weight_stats(resnet, "ResNet")

