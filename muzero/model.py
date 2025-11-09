"""
Stochastic MuZero Network Architecture - Full Implementation
Dựa trên LightZero implementation với support cho Atari, 2048, và board games

Kiến trúc đầy đủ gồm:
1. Representation Network: h(observation) -> hidden_state
2. Dynamics Network: g(hidden_state, action) -> (next_hidden_state, reward)
3. Prediction Network: f(hidden_state) -> (policy, value)
4. ChanceEncoder: Encode stochastic outcomes (cho games có chance nodes)
5. Afterstate Networks: Xử lý afterstates (state sau action, trước chance)
6. Categorical Distribution: Cho rewards/values (thay vì scalar)
7. Self-Supervised Learning: Optional, cho complex environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Sequence


def get_support_size(support_range):
    """Tính số lượng bins trong categorical distribution"""
    if isinstance(support_range, tuple) and len(support_range) == 3:
        start, end, step = support_range
        return len(torch.arange(start, end, step))
    return 1


def scalar_to_categorical(scalar, support_range):
    """Convert scalar value thành categorical distribution"""
    if isinstance(support_range, tuple) and len(support_range) == 3:
        start, end, step = support_range
        support = torch.arange(start, end, step, device=scalar.device)
        # Create one-hot distribution centered at scalar value
        distances = torch.abs(support - scalar)
        probs = F.softmax(-distances / step, dim=-1)
        return probs
    return scalar


def categorical_to_scalar(categorical, support_range):
    """Convert categorical distribution về scalar value"""
    if isinstance(support_range, tuple) and len(support_range) == 3:
        start, end, step = support_range
        support = torch.arange(start, end, step, device=categorical.device)
        return torch.sum(support * categorical, dim=-1, keepdim=True)
    return categorical


class OnehotArgmax(torch.autograd.Function):
    """
    Straight Through Estimator: one-hot argmax với gradient flow
    
    Cho phép backpropagation qua discrete sampling operation
    """
    @staticmethod
    def forward(ctx, input):
        """Forward: one-hot argmax"""
        return torch.zeros_like(input).scatter_(
            -1, 
            torch.argmax(input, dim=-1, keepdim=True), 
            1.
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward: gradient flows through unchanged"""
        return grad_output


class StraightThroughEstimator(nn.Module):
    """Wrapper cho OnehotArgmax"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return OnehotArgmax.apply(x)


class ChanceEncoderBackbone(nn.Module):
    """
    Chance Encoder Backbone: Encode observations thành chance encoding
    Support cả CNN (cho images) và MLP (cho vectors)
    """
    def __init__(self, observation_shape, chance_space_size, encoder_type='conv'):
        super().__init__()
        self.encoder_type = encoder_type
        self.chance_space_size = chance_space_size
        
        if encoder_type == 'conv':
            # CNN cho image observations (Atari, etc.)
            if isinstance(observation_shape, (list, tuple)) and len(observation_shape) == 3:
                C, H, W = observation_shape
                self.conv1 = nn.Conv2d(C * 2, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.fc1 = nn.Linear(64 * H * W, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, chance_space_size)
            else:
                raise ValueError(f"Invalid observation_shape for conv encoder: {observation_shape}")
        elif encoder_type == 'mlp':
            # MLP cho vector observations
            if isinstance(observation_shape, int):
                input_dim = observation_shape
            elif isinstance(observation_shape, (list, tuple)):
                input_dim = np.prod(observation_shape)
            else:
                raise ValueError(f"Invalid observation_shape for mlp encoder: {observation_shape}")
            
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, chance_space_size)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, x):
        if self.encoder_type == 'conv':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            B = x.shape[0]
            x = x.view(B, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:  # mlp
            if len(x.shape) > 2:
                B = x.shape[0]
                x = x.view(B, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        return x


class ChanceEncoder(nn.Module):
    """
    Chance Encoder: Encode observations thành discrete chance outcomes
    
    Sử dụng Straight Through Estimator để cho phép gradient flow
    """
    def __init__(self, observation_shape, chance_space_size, encoder_type='conv'):
        super().__init__()
        self.chance_space_size = chance_space_size
        self.encoder = ChanceEncoderBackbone(observation_shape, chance_space_size, encoder_type)
        self.onehot_argmax = StraightThroughEstimator()
    
    def forward(self, observations):
        """
        Args:
            observations: Observation tensor [batch, ...]
        
        Returns:
            chance_encoding: Continuous encoding [batch, chance_space_size]
            chance_onehot: One-hot discrete encoding [batch, chance_space_size]
        """
        chance_encoding = self.encoder(observations)
        chance_onehot = self.onehot_argmax(chance_encoding)
        return chance_encoding, chance_onehot


class MLP(nn.Module):
    """Multi-Layer Perceptron helper"""
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers, activation=nn.ReLU(), norm_type='BN',
                 output_activation=False, output_norm=False,
                 last_linear_layer_init_zero=False):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_channels, out_channels))
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
            
            if i < num_layers - 1 or output_activation:
                if norm_type == 'BN':
                    layers.append(nn.BatchNorm1d(hidden_channels if i < num_layers - 1 else out_channels))
                layers.append(activation)
        
        if output_norm and norm_type == 'BN':
            layers.append(nn.BatchNorm1d(out_channels))
        
        self.mlp = nn.Sequential(*layers)
        
        if last_linear_layer_init_zero and num_layers > 0:
            # Initialize last layer to zero
            last_layer = None
            for layer in reversed(self.mlp):
                if isinstance(layer, nn.Linear):
                    last_layer = layer
                    break
            if last_layer is not None:
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)
    
    def forward(self, x):
        return self.mlp(x)


class ResBlock(nn.Module):
    """Residual Block với BatchNorm"""
    def __init__(self, num_hidden, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.activation = activation
    
    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.activation(x)
        return x


class RepresentationNetwork(nn.Module):
    """
    Representation Network: h(observation) -> hidden_state
    
    Support flexible observation shapes (board games, Atari, 2048, etc.)
    """
    def __init__(self, observation_shape, num_resBlocks, num_hidden, 
                 downsample=False, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.observation_shape = observation_shape
        self.downsample = downsample
        
        # Determine input channels
        if isinstance(observation_shape, (list, tuple)) and len(observation_shape) == 3:
            C, H, W = observation_shape
            self.input_channels = C
        else:
            # For vector observations, use MLP
            self.input_channels = None
            self.use_mlp = True
            if isinstance(observation_shape, int):
                input_dim = observation_shape
            else:
                input_dim = np.prod(observation_shape)
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, num_hidden * 4),
                nn.ReLU(),
                nn.Linear(num_hidden * 4, num_hidden * 2),
                nn.ReLU(),
                nn.Linear(num_hidden * 2, num_hidden)
            )
            return
        
        self.use_mlp = False
        
        # Initial conv block
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            activation
        )
        
        # Downsampling nếu cần (cho Atari)
        if downsample:
            self.downsample_layers = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_hidden),
                activation,
                nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_hidden),
                activation
            )
        else:
            self.downsample_layers = None
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_hidden, activation) for _ in range(num_resBlocks)
        ])
    
    def forward(self, observations):
        """
        Args:
            observations: Observation tensor [batch, C, H, W] hoặc [batch, features]
        
        Returns:
            hidden_state: [batch, num_hidden, H, W] hoặc [batch, num_hidden]
        """
        if self.use_mlp:
            if len(observations.shape) > 2:
                B = observations.shape[0]
                observations = observations.view(B, -1)
            return self.mlp(observations)
        
        x = observations
        x = self.conv_block(x)
        
        if self.downsample_layers is not None:
            x = self.downsample_layers(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return x


class DynamicsNetwork(nn.Module):
    """
    Dynamics Network: g(hidden_state, action, chance) -> (next_hidden_state, reward)
    
    Support cả Gaussian latent (original) và ChanceEncoder (LightZero style)
    """
    def __init__(self, observation_shape, num_resBlocks, num_hidden,
                 reward_head_channels, reward_head_hidden_channels,
                 reward_support_size, flatten_input_size_for_reward_head,
                 use_chance_encoder=False, chance_space_size=None,
                 latent_dim=None, use_categorical=True,
                 last_linear_layer_init_zero=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.use_chance_encoder = use_chance_encoder
        self.use_categorical = use_categorical
        self.reward_support_size = reward_support_size
        self.use_gaussian_latent = (latent_dim is not None)
        self.latent_dim = latent_dim
        self.observation_shape = observation_shape
        self.activation = activation
        
        # Input: hidden_state + action_plane + (optional) chance_onehot
        if use_chance_encoder:
            input_channels = num_hidden + 1 + chance_space_size
        else:
            input_channels = num_hidden + 1
        
        # State decoder
        self.state_decoder = nn.Sequential(
            nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            activation
        )
        
        # Latent processing (nếu dùng Gaussian latent)
        if not use_chance_encoder and latent_dim is not None:
            # Encoder cho Gaussian latent
            if isinstance(observation_shape, (list, tuple)) and len(observation_shape) == 3:
                _, H, W = observation_shape
                spatial_size = H * W
            else:
                spatial_size = 64  # Default
            
            self.latent_encoder = nn.Sequential(
                nn.Conv2d(input_channels, num_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_hidden),
                activation,
                nn.Flatten(),
                nn.Linear(num_hidden * spatial_size, latent_dim * 2)
            )
            
            self.latent_projection = nn.Sequential(
                nn.Linear(latent_dim, num_hidden * spatial_size),
                activation
            )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_hidden, activation) for _ in range(num_resBlocks)
        ])
        
        # Reward head
        self.reward_head_conv = nn.Sequential(
            nn.Conv2d(num_hidden, reward_head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reward_head_channels),
            activation
        )
        
        self.reward_head_mlp = MLP(
            in_channels=flatten_input_size_for_reward_head,
            hidden_channels=reward_head_hidden_channels[0] if len(reward_head_hidden_channels) > 0 else 32,
            out_channels=reward_support_size,
            num_layers=len(reward_head_hidden_channels) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
    
    def forward(self, hidden_state, action, chance_onehot=None):
        """
        Args:
            hidden_state: [batch, num_hidden, H, W]
            action: [batch] integer actions
            chance_onehot: [batch, chance_space_size] optional, nếu dùng chance encoder
        
        Returns:
            next_hidden_state: [batch, num_hidden, H, W]
            reward: [batch, reward_support_size] nếu categorical, [batch, 1] nếu scalar
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        H, W = hidden_state.shape[2:]  # Only need H, W from shape
        
        # Create action plane
        action_plane = torch.zeros((batch_size, 1, H, W), device=device)
        action_size = action.max().item() + 1 if len(action) > 0 else 1
        for i in range(batch_size):
            action_value = action[i].item() / max(action_size, 1)
            action_plane[i, 0, :, :] = action_value
        
        # Concatenate inputs
        if self.use_chance_encoder and chance_onehot is not None:
            # Reshape chance_onehot to spatial
            chance_spatial = chance_onehot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x = torch.cat([hidden_state, action_plane, chance_spatial], dim=1)
        else:
            x = torch.cat([hidden_state, action_plane], dim=1)
        
        # Process latent (Gaussian case)
        if self.use_gaussian_latent:
            latent_params = self.latent_encoder(x)
            latent_dim = latent_params.shape[1] // 2
            mu = latent_params[:, :latent_dim]
            log_var = latent_params[:, latent_dim:]
            
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                latent_z = mu + eps * std
            else:
                latent_z = mu
            
            latent_spatial = self.latent_projection(latent_z)
            # Reshape to match hidden_state dimensions
            if isinstance(self.observation_shape, (list, tuple)) and len(self.observation_shape) == 3:
                _, obs_H, obs_W = self.observation_shape
            else:
                obs_H, obs_W = H, W
            
            latent_spatial = latent_spatial.view(batch_size, self.latent_dim, obs_H, obs_W)
            
            # Adjust if dimension mismatch
            if latent_spatial.shape[1] != hidden_state.shape[1] or latent_spatial.shape[2:] != hidden_state.shape[2:]:
                # Project to correct dimension
                if not hasattr(self, 'latent_dim_proj'):
                    self.latent_dim_proj = nn.Sequential(
                        nn.Conv2d(self.latent_dim, hidden_state.shape[1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_state.shape[1]),
                        self.activation
                    ).to(device)
                latent_spatial = self.latent_dim_proj(latent_spatial)
                
                # Resize if spatial dimensions don't match
                if latent_spatial.shape[2:] != hidden_state.shape[2:]:
                    latent_spatial = F.interpolate(latent_spatial, size=hidden_state.shape[2:], mode='bilinear', align_corners=False)
        else:
            latent_spatial = None
        
        # Decode state
        x = self.state_decoder(x)
        
        if latent_spatial is not None:
            x = x + latent_spatial
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        next_hidden_state = x
        
        # Reward prediction
        reward_features = self.reward_head_conv(next_hidden_state)
        reward_features_flat = reward_features.view(batch_size, -1)
        reward_logits = self.reward_head_mlp(reward_features_flat)
        
        if self.use_categorical and self.reward_support_size > 1:
            return next_hidden_state, reward_logits
        else:
            # Convert to scalar
            reward = categorical_to_scalar(
                F.softmax(reward_logits, dim=-1),
                (-300, 301, 1)  # Default support range
            )
            return next_hidden_state, reward


class PredictionNetwork(nn.Module):
    """
    Prediction Network: f(hidden_state) -> (policy, value)
    
    Support categorical distribution cho values
    """
    def __init__(self, observation_shape, action_space_size, num_resBlocks, num_hidden,
                 value_head_channels, policy_head_channels,
                 value_head_hidden_channels, policy_head_hidden_channels,
                 value_support_size, flatten_input_size_for_value_head,
                 flatten_input_size_for_policy_head,
                 use_categorical=True,
                 last_linear_layer_init_zero=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.use_categorical = use_categorical
        self.value_support_size = value_support_size
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_hidden, activation) for _ in range(num_resBlocks)
        ])
        
        # Value head
        self.value_head_conv = nn.Sequential(
            nn.Conv2d(num_hidden, value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_head_channels),
            activation
        )
        
        self.value_head_mlp = MLP(
            in_channels=flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels[0] if len(value_head_hidden_channels) > 0 else 32,
            out_channels=value_support_size,
            num_layers=len(value_head_hidden_channels) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        
        # Policy head
        self.policy_head_conv = nn.Sequential(
            nn.Conv2d(num_hidden, policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_head_channels),
            activation
        )
        
        self.policy_head_mlp = MLP(
            in_channels=flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels[0] if len(policy_head_hidden_channels) > 0 else 32,
            out_channels=action_space_size,
            num_layers=len(policy_head_hidden_channels) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
    
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: [batch, num_hidden, H, W] hoặc [batch, num_hidden]
        
        Returns:
            policy_logits: [batch, action_space_size]
            value_logits: [batch, value_support_size] nếu categorical, [batch, 1] nếu scalar
        """
        # Handle MLP case (vector observations)
        if len(hidden_state.shape) == 2:
            # Vector input, use MLP directly
            value_features = hidden_state
            policy_features = hidden_state
        else:
            # Conv case
            x = hidden_state
            for res_block in self.res_blocks:
                x = res_block(x)
            
            value_features = self.value_head_conv(x)
            policy_features = self.policy_head_conv(x)
            
            # Flatten
            batch_size = value_features.shape[0]
            value_features = value_features.view(batch_size, -1)
            policy_features = policy_features.view(batch_size, -1)
        
        # MLP heads
        value_logits = self.value_head_mlp(value_features)
        policy_logits = self.policy_head_mlp(policy_features)
        
        if self.use_categorical and self.value_support_size > 1:
            return policy_logits, value_logits
        else:
            # Convert to scalar
            value = categorical_to_scalar(
                F.softmax(value_logits, dim=-1),
                (-300, 301, 1)  # Default support range
            )
            return policy_logits, value


class AfterstateDynamicsNetwork(nn.Module):
    """
    Afterstate Dynamics Network: Xử lý transitions từ afterstate
    
    Afterstate = state sau khi player action nhưng trước khi chance outcome
    """
    def __init__(self, num_resBlocks, num_hidden,
                 reward_head_channels, reward_head_hidden_channels,
                 reward_support_size, flatten_input_size_for_reward_head,
                 chance_space_size, last_linear_layer_init_zero=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.chance_space_size = chance_space_size
        # Similar to DynamicsNetwork but for afterstate -> next state
        self.network = DynamicsNetwork(
            observation_shape=None,  # Not used in afterstate
            num_resBlocks=num_resBlocks,
            num_hidden=num_hidden,
            reward_head_channels=reward_head_channels,
            reward_head_hidden_channels=reward_head_hidden_channels,
            reward_support_size=reward_support_size,
            flatten_input_size_for_reward_head=flatten_input_size_for_reward_head,
            use_chance_encoder=True,
            chance_space_size=chance_space_size,
            use_categorical=True,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            activation=activation
        )
    
    def forward(self, afterstate, chance_onehot):
        """
        Args:
            afterstate: Afterstate hidden representation [batch, num_hidden, H, W]
            chance_onehot: Chance outcome [batch, chance_space_size]
        
        Returns:
            next_hidden_state: [batch, num_hidden, H, W]
            reward: [batch, reward_support_size]
        """
        batch_size = afterstate.shape[0]
        if chance_onehot.dim() == 1:
            chance_onehot = chance_onehot.unsqueeze(0)

        if chance_onehot.shape[1] != self.chance_space_size:
            raise ValueError(f"Expected chance_onehot dimension {self.chance_space_size}, got {chance_onehot.shape[1]}")

        chance_onehot = chance_onehot.to(afterstate.device)
        dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=afterstate.device)
        return self.network.forward(afterstate, dummy_actions, chance_onehot)


class AfterstatePredictionNetwork(nn.Module):
    """
    Afterstate Prediction Network: Predict policy và value từ afterstate
    """
    def __init__(self, chance_space_size, num_resBlocks, num_hidden,
                 value_head_channels, policy_head_channels,
                 value_head_hidden_channels, policy_head_hidden_channels,
                 value_support_size, flatten_input_size_for_value_head,
                 flatten_input_size_for_policy_head,
                 last_linear_layer_init_zero=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        # Similar to PredictionNetwork but outputs chance policy
        self.res_blocks = nn.ModuleList([
            ResBlock(num_hidden, activation) for _ in range(num_resBlocks)
        ])
        
        self.value_head_conv = nn.Sequential(
            nn.Conv2d(num_hidden, value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_head_channels),
            activation
        )
        
        self.policy_head_conv = nn.Sequential(
            nn.Conv2d(num_hidden, policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_head_channels),
            activation
        )
        
        self.value_head_mlp = MLP(
            in_channels=flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels[0] if len(value_head_hidden_channels) > 0 else 32,
            out_channels=value_support_size,
            num_layers=len(value_head_hidden_channels) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        
        self.policy_head_mlp = MLP(
            in_channels=flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels[0] if len(policy_head_hidden_channels) > 0 else 32,
            out_channels=chance_space_size,
            num_layers=len(policy_head_hidden_channels) + 1,
            activation=activation,
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
    
    def forward(self, afterstate):
        """
        Args:
            afterstate: [batch, num_hidden, H, W]
        
        Returns:
            chance_policy_logits: [batch, chance_space_size]
            afterstate_value_logits: [batch, value_support_size]
        """
        x = afterstate
        for res_block in self.res_blocks:
            x = res_block(x)
        
        value_features = self.value_head_conv(x)
        policy_features = self.policy_head_conv(x)
        
        batch_size = value_features.shape[0]
        value_features = value_features.view(batch_size, -1)
        policy_features = policy_features.view(batch_size, -1)
        
        chance_policy_logits = self.policy_head_mlp(policy_features)
        afterstate_value_logits = self.value_head_mlp(value_features)
        
        return chance_policy_logits, afterstate_value_logits


class StochasticMuZeroNetwork(nn.Module):
    """
    ═══════════════════════════════════════════════════════════════════════════
    STOCHASTIC MUZERO NETWORK - Full Implementation
    ═══════════════════════════════════════════════════════════════════════════
    
    Kiến trúc linh hoạt hỗ trợ cả deterministic và stochastic environments.
    
    Support:
    --------
    - Deterministic games: Connect4, Chess, Go (use_chance_encoder=False)
    - Stochastic games: 2048, Backgammon, Poker (use_chance_encoder=True)
    - Board games: Connect4, Chess, Breakthrough
    - Atari games: với downsample=True
    - Games với chance nodes: 2048, dice games
    
    Components:
    -----------
    1. Representation Network: h(observation) → hidden_state
    2. Dynamics Network: g(hidden_state, action) → (next_hidden_state, reward)
    3. Prediction Network: f(hidden_state) → (policy, value)
    4. [Optional] Chance Encoder: encode stochastic outcomes
    5. [Optional] Afterstate Networks: xử lý afterstates (trước chance events)
    
    Architecture Modes:
    -------------------
    
    📌 MODE 1: Deterministic MuZero (cho Connect4, Chess, Go)
    --------------------------------------------------------
    use_chance_encoder=False
    use_afterstate=False
    use_gaussian_latent=False
    
    → Chỉ có 3 networks cơ bản: Representation + Dynamics + Prediction
    → Phù hợp cho games KHÔNG có random elements
    
    📌 MODE 2: Stochastic MuZero (cho 2048, Poker, Backgammon)
    ----------------------------------------------------------
    use_chance_encoder=True
    use_afterstate=True
    use_gaussian_latent=False (hoặc True)
    
    → Có đầy đủ 5+ networks để model stochastic transitions
    → Phù hợp cho games CÓ random elements (dice, card shuffle, tile spawn)
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    def __init__(self, 
                 # ===== CORE PARAMETERS (Required) =====
                 observation_shape,
                 # Shape của observation input
                 # - Spatial: (channels, height, width) ví dụ: (3, 6, 7) cho Connect4
                 # - Vector: (features,) ví dụ: (128,) cho feature vector
                 
                 action_space_size,
                 # Số lượng actions có thể
                 # - Connect4: 7 (7 columns)
                 # - Chess: ~4672 (all possible moves)
                 # - 2048: 4 (up, down, left, right)
                 
                 # ===== STOCHASTIC PARAMETERS =====
                 chance_space_size=2,
                 # Số lượng possible chance outcomes (chỉ dùng khi use_chance_encoder=True)
                 # - 2048: ~16 (số positions có thể spawn tile)
                 # - Dice game: 6 (1-6)
                 # - Backgammon: ~36 (dice combinations)
                 # ⚠️ IGNORED nếu use_chance_encoder=False
                 
                 # ===== NETWORK ARCHITECTURE =====
                 num_res_blocks=1,
                 # Số ResNet blocks trong mỗi network
                 # - Small: 1-2 (fast, ít params)
                 # - Medium: 4-6 (balanced)
                 # - Large: 9-16 (slow, nhiều params)
                 # Trade-off: accuracy vs speed vs memory
                 
                 num_channels=64,
                 # Số hidden channels trong networks
                 # - Small: 32-64 (fast)
                 # - Medium: 64-128 (balanced)
                 # - Large: 128-256 (powerful)
                 # Ảnh hưởng lớn đến model size: O(num_channels²)
                 
                 # ===== HEAD ARCHITECTURE =====
                 reward_head_channels=16,
                 # Channels cho reward prediction head
                 
                 value_head_channels=16,
                 # Channels cho value prediction head
                 
                 policy_head_channels=16,
                 # Channels cho policy prediction head
                 
                 reward_head_hidden_channels=[32],
                 # Hidden layers trong reward MLP
                 # - [32]: 1 hidden layer với 32 units
                 # - [64, 32]: 2 hidden layers
                 
                 value_head_hidden_channels=[32],
                 # Hidden layers trong value MLP
                 
                 policy_head_hidden_channels=[32],
                 # Hidden layers trong policy MLP
                 
                 # ===== CATEGORICAL DISTRIBUTION =====
                 reward_support_range=(-300., 301., 1.),
                 # Support range cho categorical reward distribution
                 # Format: (min, max, step)
                 # - (-300, 301, 1): 601 bins từ -300 đến 300
                 # - (-10, 11, 1): 21 bins từ -10 đến 10 (cho Connect4)
                 # Nhỏ hơn = ít bins = faster, ít memory
                 
                 value_support_range=(-300., 301., 1.),
                 # Support range cho categorical value distribution
                 # Tương tự reward_support_range
                 
                 # ===== FEATURE TOGGLES (Bật/Tắt các tính năng) =====
                 use_chance_encoder=True,
                 # 🎲 Bật Chance Encoder cho stochastic games
                 # - True: Dùng cho 2048, Poker, Backgammon (có random)
                 # - False: Dùng cho Connect4, Chess, Go (deterministic)
                 # ⚠️ Nếu False → ignore chance_space_size, use_afterstate
                 
                 use_afterstate=True,
                 # 🎯 Bật Afterstate Networks (cần use_chance_encoder=True)
                 # Afterstate = state SAU action TRƯỚC chance outcome
                 # - True: Model afterstates riêng (chính xác hơn cho stochastic)
                 # - False: Không model afterstates (đơn giản hơn)
                 # Example (2048):
                 #   State → Action (slide) → Afterstate → Chance (spawn) → Next State
                 
                 use_categorical=True,
                 # 📊 Dùng categorical distribution thay vì scalar
                 # - True: Values/Rewards là distributions (chính xác hơn, LightZero style)
                 # - False: Values/Rewards là scalars (đơn giản hơn, original MuZero)
                 # Categorical distribution giúp model học tốt hơn trong trường hợp:
                 #   + Multi-modal distributions
                 #   + Long-tail rewards
                 #   + Stochastic environments
                 
                 use_gaussian_latent=False,
                 # 🌀 Dùng Gaussian latent variables trong dynamics
                 # - True: Dynamics có latent variables z ~ N(μ, σ²) (stochastic)
                 # - False: Dynamics deterministic (đơn giản)
                 # ⚠️ Experimental feature, thường để False
                 
                 latent_dim=None,
                 # Dimension của latent variables (nếu use_gaussian_latent=True)
                 # - None: không dùng latent
                 # - 16, 32, 64: latent dimension
                 
                 downsample=False,
                 # 🖼️ Downsample input observations (cho Atari)
                 # - True: Giảm spatial resolution (84x84 → 21x21)
                 # - False: Giữ nguyên resolution
                 # Dùng cho Atari để giảm computation
                 
                 self_supervised_learning_loss=False,
                 # 🔬 Thêm self-supervised learning losses
                 # - True: Thêm auxiliary losses (contrastive, reconstruction)
                 # - False: Chỉ dùng RL losses
                 # ⚠️ Chưa implement đầy đủ
                 
                 device='cpu'):
                 # 🖥️ Device để chạy model
                 # - 'cpu': CPU (chậm)
                 # - 'cuda': GPU (nhanh)
                 # - 'cuda:0', 'cuda:1': Specific GPU
        super().__init__()
        
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size
        self.device = device
        self.use_chance_encoder = use_chance_encoder
        self.use_afterstate = use_afterstate
        self.use_categorical = use_categorical
        self.reward_support_range = reward_support_range
        self.value_support_range = value_support_range
        
        # Support sizes
        if use_categorical:
            self.reward_support_size = get_support_size(reward_support_range)
            self.value_support_size = get_support_size(value_support_range)
        else:
            self.reward_support_size = 1
            self.value_support_size = 1
        
        # Calculate flatten sizes
        if isinstance(observation_shape, (list, tuple)) and len(observation_shape) == 3:
            _, H, W = observation_shape
            if downsample:
                H = math.ceil(H / 16)
                W = math.ceil(W / 16)
            
            flatten_input_size_for_reward_head = reward_head_channels * H * W
            flatten_input_size_for_value_head = value_head_channels * H * W
            flatten_input_size_for_policy_head = policy_head_channels * H * W
        else:
            # Vector observations
            flatten_input_size_for_reward_head = reward_head_channels * 64
            flatten_input_size_for_value_head = value_head_channels * 64
            flatten_input_size_for_policy_head = policy_head_channels * 64
        
        # 1. Representation Network
        self.representation_network = RepresentationNetwork(
            observation_shape=observation_shape,
            num_resBlocks=num_res_blocks,
            num_hidden=num_channels,
            downsample=downsample
        )
        
        # 2. Chance Encoder (nếu cần)
        if use_chance_encoder:
            encoder_type = 'conv' if isinstance(observation_shape, (list, tuple)) and len(observation_shape) == 3 else 'mlp'
            self.chance_encoder = ChanceEncoder(
                observation_shape=observation_shape,
                chance_space_size=chance_space_size,
                encoder_type=encoder_type
            )
        
        # 3. Dynamics Network
        self.dynamics_network = DynamicsNetwork(
            observation_shape=observation_shape,
            num_resBlocks=num_res_blocks,
            num_hidden=num_channels,
            reward_head_channels=reward_head_channels,
            reward_head_hidden_channels=reward_head_hidden_channels,
            reward_support_size=self.reward_support_size,
            flatten_input_size_for_reward_head=flatten_input_size_for_reward_head,
            use_chance_encoder=use_chance_encoder,
            chance_space_size=chance_space_size if use_chance_encoder else None,
            latent_dim=latent_dim if use_gaussian_latent else None,
            use_categorical=use_categorical
        )
        
        # 4. Prediction Network
        self.prediction_network = PredictionNetwork(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            num_resBlocks=num_res_blocks,
            num_hidden=num_channels,
            value_head_channels=value_head_channels,
            policy_head_channels=policy_head_channels,
            value_head_hidden_channels=value_head_hidden_channels,
            policy_head_hidden_channels=policy_head_hidden_channels,
            value_support_size=self.value_support_size,
            flatten_input_size_for_value_head=flatten_input_size_for_value_head,
            flatten_input_size_for_policy_head=flatten_input_size_for_policy_head,
            use_categorical=use_categorical
        )
        
        # 5. Afterstate Networks (nếu cần)
        if use_afterstate:
            self.afterstate_dynamics_network = AfterstateDynamicsNetwork(
                num_resBlocks=num_res_blocks,
                num_hidden=num_channels,
                reward_head_channels=reward_head_channels,
                reward_head_hidden_channels=reward_head_hidden_channels,
                reward_support_size=self.reward_support_size,
                flatten_input_size_for_reward_head=flatten_input_size_for_reward_head,
                chance_space_size=chance_space_size
            )
            
            self.afterstate_prediction_network = AfterstatePredictionNetwork(
                chance_space_size=chance_space_size,
                num_resBlocks=num_res_blocks,
                num_hidden=num_channels,
                value_head_channels=value_head_channels,
                policy_head_channels=policy_head_channels,
                value_head_hidden_channels=value_head_hidden_channels,
                policy_head_hidden_channels=policy_head_hidden_channels,
                value_support_size=self.value_support_size,
                flatten_input_size_for_value_head=flatten_input_size_for_value_head,
                flatten_input_size_for_policy_head=flatten_input_size_for_policy_head
            )
        
        # 6. Self-Supervised Learning (optional)
        self.self_supervised_learning_loss = self_supervised_learning_loss
        if self_supervised_learning_loss:
            # TODO: Add self-supervised components
            pass
        
        self.to(device)
    
    def initial_inference(self, observations):
        """
        Initial inference: observation -> hidden_state -> (policy, value)
        
        Args:
            observations: Observation tensor [batch, ...]
        
        Returns:
            hidden_state: [batch, num_channels, H, W] hoặc [batch, num_channels]
            policy_logits: [batch, action_space_size]
            value: [batch, value_support_size] hoặc [batch, 1]
            chance_encoding, chance_onehot: (optional) nếu use_chance_encoder
        """
        # Encode observation
        if isinstance(observations, torch.Tensor):
            observations = observations.to(self.device)
        elif isinstance(observations, np.ndarray):
            # Check if object array (nested structure)
            if observations.dtype == object:
                # Try to stack individual items
                observations = np.stack([np.array(obs, dtype=np.float32) for obs in observations])
            observations = torch.from_numpy(observations).float().to(self.device)
        elif isinstance(observations, list):
            # Convert list to numpy then tensor
            observations = np.stack([np.array(obs, dtype=np.float32) for obs in observations])
            observations = torch.from_numpy(observations).float().to(self.device)
        else:
            # Fallback: try to convert whatever it is
            observations = torch.tensor(observations, device=self.device, dtype=torch.float32)
        
        hidden_state = self.representation_network(observations)
        
        # Chance encoding (nếu cần)
        if self.use_chance_encoder:
            chance_encoding, chance_onehot = self.chance_encoder(observations)
        else:
            chance_encoding, chance_onehot = None, None
        
        # Prediction
        policy_logits, value = self.prediction_network(hidden_state)
        
        if chance_encoding is not None:
            return hidden_state, policy_logits, value, chance_encoding, chance_onehot
        return hidden_state, policy_logits, value
    
    def recurrent_inference(self, hidden_state, action, chance_onehot=None):
        """
        Recurrent inference: (hidden_state, action) -> (next_hidden_state, reward, policy, value)
        
        Args:
            hidden_state: [batch, num_channels, H, W] hoặc [batch, num_channels]
            action: [batch] integer actions
            chance_onehot: [batch, chance_space_size] optional
        
        Returns:
            next_hidden_state: [batch, num_channels, H, W]
            reward: [batch, reward_support_size] hoặc [batch, 1]
            policy_logits: [batch, action_space_size]
            value: [batch, value_support_size] hoặc [batch, 1]
        """
        # Dynamics
        next_hidden_state, reward = self.dynamics_network(hidden_state, action, chance_onehot)
        
        # Prediction
        policy_logits, value = self.prediction_network(next_hidden_state)
        
        return next_hidden_state, reward, policy_logits, value
    
    def dynamics(self, hidden_state, action, chance_onehot=None):
        """
        Wrapper for dynamics_network - API compatibility
        
        Args:
            hidden_state: [batch, num_channels, H, W]
            action: [batch] integer actions
            chance_onehot: [batch, chance_space_size] optional
        
        Returns:
            next_hidden_state, reward
        """
        return self.dynamics_network(hidden_state, action, chance_onehot)
    
    def prediction(self, hidden_state):
        """
        Wrapper for prediction_network - API compatibility
        
        Args:
            hidden_state: [batch, num_channels, H, W]
        
        Returns:
            policy_logits, value
        """
        return self.prediction_network(hidden_state)
    
    def afterstate_prediction(self, afterstate_hidden_state):
        """Wrapper for afterstate prediction network."""
        if not self.use_afterstate:
            raise ValueError("Afterstate networks not enabled")
        if not isinstance(afterstate_hidden_state, torch.Tensor):
            afterstate_hidden_state = torch.tensor(afterstate_hidden_state, dtype=torch.float32, device=self.device)
        else:
            afterstate_hidden_state = afterstate_hidden_state.to(self.device)
        if afterstate_hidden_state.dim() == 3:
            afterstate_hidden_state = afterstate_hidden_state.unsqueeze(0)
        return self.afterstate_prediction_network(afterstate_hidden_state)

    def afterstate_dynamics(self, afterstate_hidden_state, chance_onehot):
        """Wrapper for afterstate dynamics network."""
        if not self.use_afterstate:
            raise ValueError("Afterstate networks not enabled")
        if not isinstance(afterstate_hidden_state, torch.Tensor):
            afterstate_hidden_state = torch.tensor(afterstate_hidden_state, dtype=torch.float32, device=self.device)
        else:
            afterstate_hidden_state = afterstate_hidden_state.to(self.device)
        if afterstate_hidden_state.dim() == 3:
            afterstate_hidden_state = afterstate_hidden_state.unsqueeze(0)

        if not isinstance(chance_onehot, torch.Tensor):
            chance_onehot = torch.tensor(chance_onehot, dtype=torch.float32, device=self.device)
        else:
            chance_onehot = chance_onehot.to(self.device)
        if chance_onehot.dim() == 1:
            chance_onehot = chance_onehot.unsqueeze(0)

        return self.afterstate_dynamics_network(afterstate_hidden_state, chance_onehot)

    def afterstate_inference(self, hidden_state, action):
        """
        Afterstate inference: (hidden_state, action) -> afterstate -> (chance_policy, value)
        
        Chỉ dùng nếu use_afterstate=True
        """
        if not self.use_afterstate:
            raise ValueError("Afterstate networks not enabled")
        
        if not isinstance(hidden_state, torch.Tensor):
            hidden_state = torch.tensor(hidden_state, dtype=torch.float32, device=self.device)
        else:
            hidden_state = hidden_state.to(self.device)

        squeeze_hidden = False
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.unsqueeze(0)
            squeeze_hidden = True
        elif hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long, device=self.device)
        else:
            action = action.to(self.device)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        next_hidden_state, reward = self.dynamics_network(hidden_state, action)
        chance_policy_logits, afterstate_value_logits = self.afterstate_prediction_network(next_hidden_state)

        if squeeze_hidden:
            next_hidden_state = next_hidden_state.squeeze(0)
            if isinstance(reward, torch.Tensor):
                reward = reward.squeeze(0)
            chance_policy_logits = chance_policy_logits.squeeze(0)
            afterstate_value_logits = afterstate_value_logits.squeeze(0)

        return next_hidden_state, reward, chance_policy_logits, afterstate_value_logits


# Backward compatibility alias
MuZeroNetwork = StochasticMuZeroNetwork


def weights_init_normal(m):
    """Khởi tạo weights theo phân phối chuẩn"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        if m.bias is not None:
            m.bias.data.fill_(0)
