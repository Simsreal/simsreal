import torch
import torch.nn as nn

from agi.learning.efforts import Torques
from agi.learning.emotions import PAD


class LSTM(nn.Module):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        n_layers,
        device,
        batch_size,
        n_actuators,
        batch_first=True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=batch_first,
        ).to(device)

        self.h = torch.zeros(
            (
                n_layers,
                batch_size,
                hidden_dim,
            ),
            dtype=torch.float32,
        ).to(device)

        self.c = torch.zeros(
            (
                n_layers,
                batch_size,
                hidden_dim,
            ),
            dtype=torch.float32,
        ).to(device)

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.torques = Torques(hidden_dim, n_actuators)
        self.pad = PAD(hidden_dim)

    def forward(self, ctx):
        self.h = self.h.detach()
        self.c = self.c.detach()

        out, (h, c) = self.lstm(ctx, (self.h, self.c))
        self.h = h
        self.c = c
        out = out[:, -1, :]
        torques = self.torques(out)
        emotions = self.pad(out)
        return {
            "torques": torques,
            "emotions": emotions,
        }
