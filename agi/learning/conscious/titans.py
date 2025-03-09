import torch.nn as nn

from titans_pytorch import NeuralMemory

from agi.learning.efforts import Torques
from agi.learning.emotions import PAD


class Titans(nn.Module):
    def __init__(
        self,
        latent_size,
        chunk_size,
        device,
        n_actuators,
    ):
        super().__init__()
        self.mem = NeuralMemory(
            dim=latent_size,
            chunk_size=chunk_size,
        ).to(device)

        self.torques = Torques(latent_size, n_actuators).to(device)
        self.pad = PAD(latent_size).to(device)

    def forward(self, ctx):
        retrieved, _ = self.mem(ctx)
        retrieved = retrieved[:, -1, :]

        torques = self.torques(retrieved)
        emotions = self.pad(retrieved)
        return {
            "torques": torques,
            "emotions": emotions,
        }
