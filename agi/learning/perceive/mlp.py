import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, emb_dim=32, input_dim=2, hidden_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, emb_dim)
        self.fc_logvar = nn.Linear(hidden_dim, emb_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # x: [B, N, 2]
        B, N, D = x.shape
        x_flat = x.view(B * N, D)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        r = self.decode(z)
        r = r.view(B, N, D)
        mu = mu.view(B, N, -1).mean(dim=1)      # [B, emb_dim]
        logvar = logvar.view(B, N, -1).mean(dim=1)
        return r, mu, logvar
