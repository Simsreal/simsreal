import json
import torch
import zmq
import numpy as np
from loguru import logger


def ctx_parser(runtime_engine):
    logger.info("starting ctx parser")
    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(
        f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"
    )
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    perceiver_shm = runtime_engine.get_shared_memory("perceiver")
    motivator_shm = runtime_engine.get_shared_memory("motivator")
    memory_manager_shm = runtime_engine.get_shared_memory("memory_manager")

    while True:
        frame: dict = sub.recv_json()
        agent_status = json.loads(frame["robot_state"])

        # Parse raycast data
        raycast_data = agent_status.get("raycast_data", {})

        logger.info(
            f"Agent: Pos=({agent_status['x']:.2f}, {agent_status['y']:.2f}, {agent_status['z']:.2f}), "
            f"HP={agent_status['hit_point']}, "
            f"Rays={raycast_data.get('total_rays', 0)}, "
            f"Obstacles={raycast_data.get('obstacle_count', 0)}, "
            f"Enemies={raycast_data.get('enemy_count', 0)}"
        )

        # Create structured raycast matrices
        raycast_matrices = create_raycast_matrices(raycast_data)

        # Add raycast matrices to agent status
        agent_status["raycast_matrices"] = raycast_matrices

        # Put the enhanced agent status in robot_state queue
        motivator_shm["robot_state"].put(agent_status)

        # Extract movement commands
        movement_commands = torch.tensor(
            [
                agent_status.get("movement_x", 0.0),
                agent_status.get("movement_y", 0.0),
                agent_status.get("orientation", 0.0),
            ]
        ).unsqueeze(0)

        memory_manager_shm["torque"].put(movement_commands)

        # Create vision tensor from raycast data (convert LiDAR to "vision")
        lidar_vision = create_lidar_vision_tensor(raycast_matrices)
        perceiver_shm["vision"].put(lidar_vision)

        # Create placeholder force data
        empty_force = torch.zeros(1, dtype=torch.float32)
        motivator_shm["force_on_geoms"].put(empty_force)


def create_raycast_matrices(raycast_data):
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


def create_distance_angle_matrix(distances, angles, types=None):
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


def create_lidar_vision_tensor(raycast_matrices):
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


def lidar_to_2d_grid(distance_angle_matrix, height, width):
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
