import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.game = game
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([
            ResBlock(num_hidden) for i in range(num_resBlocks)
        ])
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
        self.device = device
        self.to(device)
    
    def forward(self, state):
        encoded_state = self.game.get_encoded_state(state)
        x = torch.tensor(encoded_state).unsqueeze(0)
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        value = value.item()
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().tolist()
        return value, policy


class MockNeuralNetwork(nn.Module):
    def __init__(self, game, args, num_workers=8, num_rollout=1000):
        super().__init__()
        self.game = game
        self.num_workers = num_workers
        self.num_rollout = num_rollout
        self.args = args
    
    def forward(self, state):
        value = self.simulate(state)
        policy = [1.0 for i in range(self.game.action_size)]
        return value, policy
    
    def _single_rollout(self, state):
        rollout_state = state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            rollout_player = self.game.get_opponent(rollout_player)
    
    def simulate(self, state):
        if isinstance(self.game, type(self.game)):
            rows, cols = np.where(state == -1)
            flat_indices = rows * self.game.column_count + cols
            action = np.random.choice(flat_indices)
        elif hasattr(self.game, 'row_count') and hasattr(self.game, 'column_count'):
            rows, cols = np.where(state == -1)
            action = np.random.choice(cols)
        value, is_terminal = self.game.get_value_and_terminated(state, action)
        value = self.game.get_opponent_value(value)
        if is_terminal:
            return value
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._single_rollout, state) for _ in range(self.num_rollout)]
            rollout_values = [future.result() for future in futures]
        return np.mean(rollout_values)
