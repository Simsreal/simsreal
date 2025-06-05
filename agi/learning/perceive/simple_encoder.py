import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List


class SimpleRaycastEncoder(nn.Module):
    """Simple encoder for 24-ray raycast data - no VAE complexity needed"""

    def __init__(self, n_rays=24, feature_dim=32):
        super().__init__()
        self.n_rays = n_rays
        self.feature_dim = feature_dim

        # Simple MLP to encode raycast features
        self.encoder = nn.Sequential(
            nn.Linear(n_rays * 3, 64),  # distance + type + angle for each ray
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, raycast_data):
        """
        Args:
            raycast_data: dict with 'distances', 'types', 'angles' each of length n_rays
        Returns:
            encoded features: tensor of shape (batch_size, feature_dim)
        """
        device = next(self.parameters()).device  # Get device from model parameters

        distances = torch.tensor(
            raycast_data["distances"], dtype=torch.float32, device=device
        )
        types = torch.tensor(raycast_data["types"], dtype=torch.float32, device=device)
        angles = torch.tensor(
            raycast_data["angles"], dtype=torch.float32, device=device
        )

        # Pad to n_rays if needed
        if len(distances) < self.n_rays:
            padding = self.n_rays - len(distances)
            distances = torch.cat(
                [distances, torch.full((padding,), 100.0, device=device)]
            )
            types = torch.cat([types, torch.zeros(padding, device=device)])
            angles = torch.cat([angles, torch.zeros(padding, device=device)])
        elif len(distances) > self.n_rays:
            distances = distances[: self.n_rays]
            types = types[: self.n_rays]
            angles = angles[: self.n_rays]

        # Normalize distances to 0-1 range
        distances = torch.clamp(distances / 100.0, 0, 1)

        # Normalize types to 0-1 range (0-6 object types)
        types = types / 6.0

        # Normalize angles to 0-1 range
        angles = (angles + np.pi) / (2 * np.pi)

        # Concatenate all features
        features = torch.cat([distances, types, angles], dim=0)

        if len(features.shape) == 1:
            features = features.unsqueeze(0)  # Add batch dimension

        return self.encoder(features)


class DiscreteStateEncoder(nn.Module):
    """Encode agent state as discrete categorical features"""

    def __init__(self, feature_dim=16):
        super().__init__()
        self.feature_dim = feature_dim

        # Discrete state mappings
        self.agent_states = {0: "normal", 1: "pain", 2: "unknown", 3: "falling"}
        self.threat_levels = ["none", "low", "medium", "high"]
        self.goal_levels = ["none", "detected", "close", "very_close"]

        # Simple embedding for discrete states
        self.state_embedding = nn.Embedding(len(self.agent_states), 4)
        self.threat_embedding = nn.Embedding(len(self.threat_levels), 4)
        self.goal_embedding = nn.Embedding(len(self.goal_levels), 4)

        # Combine embeddings
        self.combiner = nn.Linear(12 + 4, feature_dim)  # 3*4 embeddings + 4 continuous

    def forward(self, context: Dict[str, Any]):
        """Create discrete state representation"""
        device = next(self.parameters()).device

        # Agent state (discrete)
        agent_state = context.get("agent_state", 0)
        agent_state_tensor = torch.tensor([agent_state], dtype=torch.long).to(device)
        agent_embed = self.state_embedding(agent_state_tensor)

        # Threat level (discrete based on detection counts)
        threats = context.get("detection_summary", {})
        threat_count = threats.get("enemy", {}).get("count", 0) + threats.get(
            "trap", {}
        ).get("count", 0)
        threat_level = min(threat_count, 3)  # 0-3 levels
        threat_tensor = torch.tensor([threat_level], dtype=torch.long).to(device)
        threat_embed = self.threat_embedding(threat_tensor)

        # Goal level (discrete based on detection counts)
        goals = context.get("detection_summary", {})
        goal_count = goals.get("goal", {}).get("count", 0) + goals.get("food", {}).get(
            "count", 0
        )
        goal_level = min(goal_count, 3)  # 0-3 levels
        goal_tensor = torch.tensor([goal_level], dtype=torch.long).to(device)
        goal_embed = self.goal_embedding(goal_tensor)

        # Continuous features (normalized)
        hit_point = context.get("hit_point", 100) / 100.0
        hunger = context.get("hunger", 0.0)

        # Simple position encoding (discretized)
        pos = context.get("agent_position", {"x": 0, "y": 0, "z": 0})
        pos_x = pos["x"] / 100.0  # Normalize position
        pos_y = pos["y"] / 100.0

        continuous_features = (
            torch.tensor([hit_point, hunger, pos_x, pos_y], dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
        )

        # Combine all features
        combined = torch.cat(
            [
                agent_embed.flatten().unsqueeze(0),
                threat_embed.flatten().unsqueeze(0),
                goal_embed.flatten().unsqueeze(0),
                continuous_features,
            ],
            dim=1,
        )

        return self.combiner(combined)
