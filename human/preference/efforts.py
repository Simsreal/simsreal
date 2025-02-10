import torch.nn as nn


class Torques(nn.Module):
    def __init__(self, hidden_dim, n_actuators):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_actuators)

    def forward(self, x):
        return self.fc(x)
