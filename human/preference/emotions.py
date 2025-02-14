import torch
import torch.nn as nn


class PAD(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x
