import torch
import numpy as np
from typing import List, Dict, Any


def process_line_of_sight(line_of_sight: List[Dict[str, Any]], agent_orientation: float = 0.0) -> Dict[str, Any]:
    """
    Process line_of_sight data to extract raycast information
    FOV is 120 degrees (-60 to +60 degrees) relative to agent's facing direction
    
    Args:
        line_of_sight: List of raycast hit data
        agent_orientation: Agent's current orientation in degrees (from Unity)
    """
    obstacle_distances = []
    obstacle_angles = []
    enemy_distances = []
    enemy_angles = []
    empty_distances = []
    empty_angles = []

    distances = []
    angles = []
    types = []

    total_rays = len(line_of_sight)

    for i, ray in enumerate(line_of_sight):
        distance = ray.get("Distance", -1.0)
        ray_type = ray.get("Type", 0)

        # Calculate relative angle for 120° FOV (-60° to +60°)
        if total_rays > 0:
            # Map ray index to relative angle within 120° FOV
            relative_angle = -60.0 + (i * 120.0 / (total_rays - 1)) if total_rays > 1 else 0.0
        else:
            relative_angle = 0.0
        
        # Store the relative angle (will be combined with agent orientation in mindmap_builder)
        distances.append(distance)
        angles.append(relative_angle)
        types.append(ray_type)

        if ray_type == 1 and distance > 0:  # Obstacle
            obstacle_distances.append(distance)
            obstacle_angles.append(relative_angle)
        elif ray_type == 2:  # Enemy
            enemy_distances.append(distance)
            enemy_angles.append(relative_angle)
        elif distance == -1.0:  # Empty/no hit
            empty_distances.append(100.0)  # Max range for visualization
            empty_angles.append(relative_angle)

    obstacle_count = len(obstacle_distances)
    enemy_count = len(enemy_distances)
    empty_count = len(empty_distances)

    return {
        "total_rays": total_rays,
        "obstacle_count": obstacle_count,
        "enemy_count": enemy_count,
        "empty_count": empty_count,
        "obstacle_distances": obstacle_distances,
        "obstacle_angles": obstacle_angles,
        "enemy_distances": enemy_distances,
        "enemy_angles": enemy_angles,
        "empty_distances": empty_distances,
        "empty_angles": empty_angles,
        "distances": distances,
        "angles": angles,
        "types": types,
        "agent_orientation": agent_orientation,  # Store for reference
    }


def create_raycast_matrices(raycast_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raycast data into structured matrices organized by type
    """
    matrices = {
        "obstacle_matrix": create_distance_angle_matrix(
            raycast_data.get("obstacle_distances", []),
            raycast_data.get("obstacle_angles", []),
        ),
        "enemy_matrix": create_distance_angle_matrix(
            raycast_data.get("enemy_distances", []),
            raycast_data.get("enemy_angles", []),
        ),
        "empty_matrix": create_distance_angle_matrix(
            raycast_data.get("empty_distances", []),
            raycast_data.get("empty_angles", []),
        ),
        "full_scan_matrix": create_distance_angle_matrix(
            raycast_data.get("distances", []),
            raycast_data.get("angles", []),
            raycast_data.get("types", []),
        ),
        "summary": {
            "total_rays": raycast_data.get("total_rays", 0),
            "obstacle_count": raycast_data.get("obstacle_count", 0),
            "enemy_count": raycast_data.get("enemy_count", 0),
            "empty_count": raycast_data.get("empty_count", 0),
        },
    }

    return matrices


def create_distance_angle_matrix(
    distances: List[float], angles: List[float], types: List[int] = None
) -> torch.Tensor:
    """
    Create a matrix from distances and angles
    Returns: torch.Tensor of shape [N, 2] or [N, 3] if types included
    """
    if not distances or not angles:
        return torch.zeros((0, 3 if types else 2), dtype=torch.float32)

    distances = torch.tensor(distances, dtype=torch.float32)
    angles = torch.tensor(angles, dtype=torch.float32)

    if types is not None:
        types = torch.tensor(types, dtype=torch.float32)
        return torch.stack([distances, angles, types], dim=1)
    else:
        return torch.stack([distances, angles], dim=1)
