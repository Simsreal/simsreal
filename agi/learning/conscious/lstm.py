import torch
import torch.nn as nn

from agi.learning.symbolic_actions import SymbolicActions, GovernancePolicy
from agi.learning.emotions import PAD


class LSTM(nn.Module):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        n_layers,
        device,
        batch_size,
        n_movement_actions,
        n_intrinsics=None,
        batch_first=True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=batch_first,
        ).to(device)

        self.h = torch.zeros((n_layers, batch_size, hidden_dim), dtype=torch.float32).to(device)
        self.c = torch.zeros((n_layers, batch_size, hidden_dim), dtype=torch.float32).to(device)

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.movement_actions = SymbolicActions(hidden_dim, n_movement_actions)
        self.emotions = PAD(hidden_dim)
        
        if n_intrinsics:
            self.governance = GovernancePolicy(hidden_dim, n_intrinsics)
        else:
            self.governance = None

    def forward(self, ctx):
        self.h = self.h.detach()
        self.c = self.c.detach()

        out, (h, c) = self.lstm(ctx, (self.h, self.c))
        self.h = h
        self.c = c
        out = out[:, -1, :]
        
        movement_logits = self.movement_actions(out)
        emotions = self.emotions(out)
        
        outputs = {
            "movement_logits": movement_logits,
            "emotions": emotions,
        }
        
        if self.governance:
            governance = self.governance(out)
            outputs["governance"] = governance
            
        return outputs
