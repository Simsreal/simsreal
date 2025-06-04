import torch.nn as nn
import torch.nn.functional as F


class SymbolicActions(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        # Return logits for symbolic actions
        return self.fc(x)


class GovernancePolicy(nn.Module):
    def __init__(self, hidden_dim, n_intrinsics):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_intrinsics)

    def forward(self, x):
        # Return policy distribution over intrinsics
        return F.softmax(self.fc(x), dim=-1)