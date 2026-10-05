"""
Masked Actor-Critic Neural Network for Monopoly Reinforcement Learning.
Optimized for CUDA execution on PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 439, act_dim: int = 77, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Policy head (logits for 117 discrete actions)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

        # Value head (expected relative value / win probability)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Forward pass with action masking.
        Illegal actions receive large negative logits (-1e9) before softmax.
        """
        features = self.encoder(obs)
        logits = self.actor(features)

        # Action masking
        large_neg = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask > 0, logits, large_neg)

        dist = torch.distributions.Categorical(logits=masked_logits)
        value = self.critic(features)

        return dist, value

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Inference helper for single-step selection."""
        with torch.no_grad():
            dist, value = self.forward(obs, action_mask)
            if deterministic:
                action = torch.argmax(dist.logits, dim=-1).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=obs.device)).item()
            return action, log_prob, value.item()
