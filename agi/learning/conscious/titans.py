import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch import NeuralMemory
import torch

from agi.learning.symbolic_actions import SymbolicActions, GovernancePolicy


class Titans(nn.Module):
    def __init__(self, latent_size, chunk_size, device, policy_dim, hidden_dim=None):
        super().__init__()
        self.mem = NeuralMemory(dim=latent_size, chunk_size=chunk_size).to(device)

        # Policy and value heads (simplified)
        self.policy_head = nn.Linear(latent_size, policy_dim).to(device)
        self.value_head = nn.Linear(latent_size, 1).to(device)

    def forward(self, x):
        # Handle both context (for memory) and flattened state (for policy)
        if len(x.shape) == 3:  # Context tensor [batch, seq, features]
            retrieved, _ = self.mem(x)
            retrieved = retrieved[:, -1, :]  # Take last timestep
        else:  # Flattened state [batch, features]
            retrieved = x

        # Policy and value outputs (simplified)
        policy = F.softmax(self.policy_head(retrieved), dim=-1)
        value = torch.tanh(self.value_head(retrieved))

        return policy, value
