import torch
import numpy as np
from typing import Dict, Any


def create_lidar_vision_tensor(raycast_matrices: Dict[str, Any]) -> torch.Tensor:
    """
    Convert LiDAR data into a vision-like tensor for the perceiver
    Creates a 2D "image" representation of the LiDAR scan
    """
    # Create a simple 2D representation
    # Channel 0: Obstacle distances
    # Channel 1: Enemy distances
    # Channel 2: Empty space distances

    height, width = 64, 64
    vision_tensor = torch.zeros((3, height, width), dtype=torch.float32)

    # Process obstacle data
    obstacle_matrix = raycast_matrices.get("obstacle_matrix", torch.zeros((0, 2)))
    if obstacle_matrix.shape[0] > 0:
        vision_tensor[0] = lidar_to_2d_grid(obstacle_matrix, height, width)

    # Process enemy data
    enemy_matrix = raycast_matrices.get("enemy_matrix", torch.zeros((0, 2)))
    if enemy_matrix.shape[0] > 0:
        vision_tensor[1] = lidar_to_2d_grid(enemy_matrix, height, width)

    # Process empty space data
    empty_matrix = raycast_matrices.get("empty_matrix", torch.zeros((0, 2)))
    if empty_matrix.shape[0] > 0:
        vision_tensor[2] = lidar_to_2d_grid(empty_matrix, height, width)

    return vision_tensor


def lidar_to_2d_grid(distance_angle_matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Convert distance-angle pairs to a 2D grid representation
    """
    grid = torch.zeros((height, width), dtype=torch.float32)

    if distance_angle_matrix.shape[0] == 0:
        return grid

    distances = distance_angle_matrix[:, 0]
    angles = distance_angle_matrix[:, 1]

    # Convert polar coordinates to cartesian and map to grid
    center_x, center_y = width // 2, height // 2
    max_distance = 100.0  # Assume max LiDAR range

    for i in range(len(distances)):
        distance = distances[i].item()
        angle = angles[i].item()

        # Convert to cartesian coordinates
        x = distance * np.cos(np.radians(angle))
        y = distance * np.sin(np.radians(angle))

        # Map to grid coordinates
        grid_x = int(center_x + (x / max_distance) * (width // 2))
        grid_y = int(center_y + (y / max_distance) * (height // 2))

        # Ensure within bounds
        if 0 <= grid_x < width and 0 <= grid_y < height:
            grid[grid_y, grid_x] = distance / max_distance  # Normalize distance

    return grid 