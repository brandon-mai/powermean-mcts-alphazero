import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import copy

from concurrent.futures import ThreadPoolExecutor

class MockNeuralNetwork(nn.Module):
    def __init__(self, game, device, num_workers=100, num_rollout=1000):
        super().__init__()
        self.game = game
        self.device = device
        self.num_workers = num_workers
        self.num_rollout = num_rollout
    
    def forward(self, states):
        values = self.simulate(
            states=states)

        value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        batch_size = len(states)
        policy_tensor = torch.ones((batch_size, self.game.action_size), dtype=torch.float32, device=self.device)

        return policy_tensor, value_tensor
    
    def _single_rollout(self, state, player):
        rollout_state = copy.deepcopy(state)
        rollout_player = state.current_player()
        while True:
            action = np.random.choice(self.game.get_valid_moves(rollout_state))
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.get_value_and_terminated(
                state=rollout_state, 
                player=player
            )
            if is_terminal:
                return value

            rollout_player = self.game.get_opponent(rollout_player)
    
    def simulate(self, states):
        values = []
        for state in states:
            player = self.game.get_current_player(state=state)

            value, is_terminal = self.game.get_value_and_terminated(
                state=state, 
                player=player)
            
            if is_terminal:
                values.append(value)
                continue
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._single_rollout, state, player) for _ in range(self.num_rollout)]
                rollout_values = [future.result() for future in futures]
            values.append(np.mean(rollout_values))
            
        return values
    
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.game = game
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
        if isinstance(states, torch.Tensor):
            x = states
            
        elif isinstance(states, np.ndarray):
            x = torch.tensor(states, device=self.device, dtype=torch.float32)
            
        elif isinstance(states, (list, tuple)):
            if len(states) == 0:
                 # Handle empty batch
                 return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
            
            if isinstance(states[0], np.ndarray):
                x = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
            else:
                x = torch.tensor(self.game.get_encoded_state(states), device=self.device, dtype=torch.float32)
        else:
             x = torch.tensor(self.game.get_encoded_state(states), device=self.device, dtype=torch.float32)
        
        if x.dtype != torch.float32:
            x = x.float()

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

def weights_init_normal(m):
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