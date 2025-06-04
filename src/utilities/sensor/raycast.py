import torch
import numpy as np
from typing import List, Dict, Any


def process_line_of_sight(line_of_sight: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process line_of_sight data to extract raycast information
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
        
        # Calculate angle based on ray index (assuming evenly distributed rays)
        angle = (i * 360.0 / total_rays) if total_rays > 0 else 0.0
        
        distances.append(distance)
        angles.append(angle)
        types.append(ray_type)
        
        if ray_type == 1 and distance > 0:  # Obstacle
            obstacle_distances.append(distance)
            obstacle_angles.append(angle)
        elif ray_type == 2:  # Enemy
            enemy_distances.append(distance)
            enemy_angles.append(angle)
        elif distance == -1.0:  # Empty/no hit
            empty_distances.append(100.0)  # Max range for visualization
            empty_angles.append(angle)
    
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


def create_distance_angle_matrix(distances: List[float], angles: List[float], types: List[int] = None) -> torch.Tensor:
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