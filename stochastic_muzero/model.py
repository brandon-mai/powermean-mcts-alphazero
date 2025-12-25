import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embeddings.weight.t()))
            
        # Get nearest code
        encoding_indices = torch.argmin(distances, dim=1) # [B]
        quantized = self.embeddings(encoding_indices)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, encoding_indices, loss

class ChanceEncoder(nn.Module):
    def __init__(self, observation_channels, hidden_channels, chance_space_size, embedding_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(observation_channels, hidden_channels, kernel_size=3, padding=1, stride=2) # Downsample
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.resblocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(2)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_gap = nn.Linear(hidden_channels, embedding_dim)
        
        self.vq = VectorQuantizer(chance_space_size, embedding_dim)
        
    def forward(self, observation):
        x = F.relu(self.bn(self.conv(observation)))
        for block in self.resblocks:
            x = block(x)
        x = self.gap(x).view(x.shape[0], -1)
        x = self.fc_gap(x)
        quantized, indices, vq_loss = self.vq(x)
        return quantized, indices, vq_loss

def categorical_to_scalar(logits, support_range=(-300, 301, 1)):
    min_val, max_val, step = support_range
    support = torch.arange(min_val, max_val, step, 
                          device=logits.device, dtype=torch.float32)
    if logits.dim() == 1:
        return torch.sum(logits * support)
    return torch.sum(logits * support, dim=-1)

def scalar_to_categorical(scalar, support_range=(-300, 301, 1)):
    min_val, max_val, step = support_range
    support_size = (max_val - min_val) // step
    
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar, dtype=torch.float32)
    
    scalar = torch.clamp(scalar, min_val, max_val - step)
    scalar_normalized = (scalar - min_val) / step
    lower = torch.floor(scalar_normalized).long()
    upper = lower + 1
    upper = torch.clamp(upper, max=support_size - 1)
    
    upper_weight = scalar_normalized - lower.float()
    lower_weight = 1.0 - upper_weight
    
    batch_size = scalar.shape[0] if scalar.dim() > 0 else 1
    dist = torch.zeros(batch_size, support_size, device=scalar.device)
    
    if scalar.dim() == 0:
        dist[0, lower] = lower_weight
        dist[0, upper] = upper_weight
    else:
        batch_idx = torch.arange(batch_size, device=scalar.device)
        dist[batch_idx, lower] = lower_weight
        dist[batch_idx, upper] = upper_weight
    
    return dist


class ResBlock(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # Zero-gamma initialization for the last BN in each ResBlock
        # This makes the block act as identity at initialization
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class RepresentationNetwork(nn.Module):
    """h(observation) → hidden_state"""
    
    def __init__(self, observation_channels, hidden_channels, num_resblocks=4):
        super().__init__()
        self.conv = nn.Conv2d(observation_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.resblocks = nn.ModuleList([
            ResBlock(hidden_channels) for _ in range(num_resblocks)
        ])
        
    def forward(self, observation):
        x = F.relu(self.bn(self.conv(observation)))
        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(nn.Module):
    """g(hidden_state, action) → (next_hidden_state, reward)"""
    
    def __init__(self, hidden_channels, action_size, reward_support_size=601):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.action_size = action_size
        
        self.conv = nn.Conv2d(hidden_channels + action_size, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.resblocks = nn.ModuleList([
            ResBlock(hidden_channels) for _ in range(4)
        ])
        
        self.reward_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.reward_bn = nn.BatchNorm2d(32)
        self.reward_fc = None
        self.reward_support_size = reward_support_size
        
    def forward(self, hidden_state, action):
        batch_size = hidden_state.shape[0]
        h, w = hidden_state.shape[2], hidden_state.shape[3]
        
        action_onehot = F.one_hot(action.long(), self.action_size).float()
        action_planes = action_onehot.view(batch_size, self.action_size, 1, 1)
        action_planes = action_planes.expand(-1, -1, h, w)
        
        x = torch.cat([hidden_state, action_planes], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
        
        next_hidden_state = x * 0.5
        
        r = F.relu(self.reward_bn(self.reward_conv(x)))
        r = r.view(batch_size, -1)
        
        if self.reward_fc is None or self.reward_fc.in_features != r.shape[1]:
            self.reward_fc = nn.Linear(r.shape[1], self.reward_support_size).to(r.device)
            nn.init.kaiming_normal_(self.reward_fc.weight)
            if self.reward_fc.bias is not None:
                 nn.init.constant_(self.reward_fc.bias, 0)
        
        reward_logits = self.reward_fc(r)
             
        return next_hidden_state, reward_logits


class PredictionNetwork(nn.Module):
    """f(hidden_state) → (policy, value)"""
    
    def __init__(self, hidden_channels, action_size, value_support_size=601):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.action_size = action_size
        
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = None
        
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc = None
        self.value_support_size = value_support_size
        
    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        
        p = F.relu(self.policy_bn(self.policy_conv(hidden_state)))
        p = p.view(batch_size, -1)
        if self.policy_fc is None or self.policy_fc.in_features != p.shape[1]:
            self.policy_fc = nn.Linear(p.shape[1], self.action_size).to(p.device)
        policy_logits = self.policy_fc(p)
        
        v = F.relu(self.value_bn(self.value_conv(hidden_state)))
        v = v.view(batch_size, -1)
        if self.value_fc is None or self.value_fc.in_features != v.shape[1]:
            self.value_fc = nn.Linear(v.shape[1], self.value_support_size).to(v.device)
        value_logits = self.value_fc(v)
        
        return policy_logits, value_logits


class AfterstateDynamicsNetwork(nn.Module):
    """g(afterstate, chance) → (next_hidden_state)"""
    
    def __init__(self, hidden_channels, chance_space_size, reward_support_size=601):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.chance_space_size = chance_space_size
        
        self.conv = nn.Conv2d(hidden_channels + chance_space_size, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.resblocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(2)])
        
        self.reward_conv = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.reward_bn = nn.BatchNorm2d(16)
        self.reward_fc = None
        self.reward_support_size = reward_support_size
        
    def forward(self, afterstate, chance_onehot):
        batch_size = afterstate.shape[0]
        h, w = afterstate.shape[2], afterstate.shape[3]
        
        chance_planes = chance_onehot.view(batch_size, self.chance_space_size, 1, 1)
        chance_planes = chance_planes.expand(-1, -1, h, w)
        
        x = torch.cat([afterstate, chance_planes], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
            
        next_hidden_state = x * 0.5
        
        r = F.relu(self.reward_bn(self.reward_conv(x)))
        r = r.view(batch_size, -1)
        if self.reward_fc is None or self.reward_fc.in_features != r.shape[1]:
            self.reward_fc = nn.Linear(r.shape[1], self.reward_support_size).to(r.device)
            nn.init.kaiming_normal_(self.reward_fc.weight)
            if self.reward_fc.bias is not None:
                 nn.init.constant_(self.reward_fc.bias, 0)
        
        reward_logits = self.reward_fc(r)
        
        return next_hidden_state, reward_logits


class AfterstatePredictionNetwork(nn.Module):
    """f_after(afterstate) → (chance_probs, value)"""
    
    def __init__(self, hidden_channels, chance_space_size, value_support_size=601):
        super().__init__()
        self.chance_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.chance_bn = nn.BatchNorm2d(32)
        self.chance_fc = None
        self.chance_space_size = chance_space_size
        
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc = None
        self.value_support_size = value_support_size
        
    def forward(self, afterstate):
        batch_size = afterstate.shape[0]
        
        c = F.relu(self.chance_bn(self.chance_conv(afterstate)))
        c = c.view(batch_size, -1)
        if self.chance_fc is None or self.chance_fc.in_features != c.shape[1]:
            self.chance_fc = nn.Linear(c.shape[1], self.chance_space_size).to(c.device)
            nn.init.kaiming_normal_(self.chance_fc.weight)
            if self.chance_fc.bias is not None:
                nn.init.constant_(self.chance_fc.bias, 0)
        chance_logits = self.chance_fc(c)
        
        v = F.relu(self.value_bn(self.value_conv(afterstate)))
        v = v.view(batch_size, -1)
        if self.value_fc is None or self.value_fc.in_features != v.shape[1]:
            self.value_fc = nn.Linear(v.shape[1], self.value_support_size).to(v.device)
        value_logits = self.value_fc(v)
        
        return chance_logits, value_logits


class StochasticMuZeroNetwork(nn.Module):
    def __init__(self, observation_shape, action_size, hidden_channels=128,
                 num_resblocks=16, chance_space_size=32, use_afterstate=False,
                 support_size=601, device="cpu", **kwargs):
        super().__init__()
        
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.hidden_channels = hidden_channels
        self.chance_space_size = chance_space_size
        self.use_afterstate = use_afterstate
        self.device = device
        
        self.value_support_range = (-300, 301, 1)
        self.reward_support_range = (-300, 301, 1)
        self.support_size = support_size
        
        obs_channels = observation_shape[0]
        
        self.representation_network = RepresentationNetwork(
            obs_channels, hidden_channels, num_resblocks
        )
        self.dynamics_network = DynamicsNetwork(
            hidden_channels, action_size, support_size
        )
        self.prediction_network = PredictionNetwork(
            hidden_channels, action_size, support_size
        )
        
        if use_afterstate:
            self.chance_encoder = ChanceEncoder(
                obs_channels, hidden_channels, chance_space_size, embedding_dim=hidden_channels
            )
            
            self.afterstate_dynamics_network = AfterstateDynamicsNetwork(
                hidden_channels, chance_space_size
            )
            self.afterstate_prediction_network = AfterstatePredictionNetwork(
                hidden_channels, chance_space_size, support_size
            )
        
        self.to(device)
        
    def encode_chance(self, observation):
        """Get ground truth chance code from observation."""
        if not self.use_afterstate:
            return None, None, 0.0
        return self.chance_encoder(observation)
        
    def representation(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        elif not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            
        if observation.device != self.device:
            observation = observation.to(self.device)
            
        return self.representation_network(observation)
    
    def dynamics(self, hidden_state, action):
        if isinstance(action, (int, np.integer)):
            action = torch.tensor([action], device=self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, device=self.device)
            
        return self.dynamics_network(hidden_state, action)
    
    def prediction(self, hidden_state):
        return self.prediction_network(hidden_state)
    
    def afterstate_prediction(self, afterstate):
        if not self.use_afterstate:
            raise RuntimeError("Afterstate networks not enabled")
        return self.afterstate_prediction_network(afterstate)
    
    def afterstate_dynamics(self, afterstate, chance_onehot):
        if not self.use_afterstate:
            raise RuntimeError("Afterstate networks not enabled")
        return self.afterstate_dynamics_network(afterstate, chance_onehot)

    def initial_inference(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)
        
        batch_size = hidden_state.shape[0]
        reward_logits = torch.zeros(batch_size, self.support_size, device=self.device)
        
        if self.use_afterstate:
            chance_logits, _ = self.afterstate_prediction(hidden_state)
        else:
            chance_logits = torch.zeros(batch_size, self.chance_space_size, device=self.device)
        
        return hidden_state, policy_logits, value_logits, reward_logits, chance_logits
    
    def recurrent_inference(self, hidden_state, action):
        next_hidden, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden)
        
        if self.use_afterstate:
            chance_logits, _ = self.afterstate_prediction(next_hidden)
        else:
            batch_size = next_hidden.shape[0]
            chance_logits = torch.zeros(batch_size, self.chance_space_size, device=self.device)
        
        return next_hidden, policy_logits, value_logits, reward_logits, chance_logits
    
    def forward(self, observation):
        hidden = self.representation(observation)
        policy, value = self.prediction(hidden)
        return policy, value


def weights_init_stochastic_muzero(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
