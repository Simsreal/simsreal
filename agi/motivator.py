from importlib import import_module
import numpy as np

from agi.memory.store import MemoryStore
from src.utilities.queues.queue_util import try_get


def motivator(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    robot_props = runtime_engine.get_metadata("robot_props")
    intrinsics_module = import_module("agi.intrinsics")
    instrinsic_lookup = {
        name: getattr(intrinsics_module, name) for name in intrinsics_module.__all__
    }
    intrinsics = cfg["intrinsics"]
    intrinsic_indices = runtime_engine.get_metadata("intrinsic_indices")
    episodic_memory_cfg = cfg["memory_management"]["episodic_memory"]
    live_memory_cfg = cfg["memory_management"]["live_memory"]
    vector_size = runtime_engine.get_metadata("latent_size")
    motivator_shm = runtime_engine.get_shared_memory("motivator")
    brain_shm = runtime_engine.get_shared_memory("brain")
    device = runtime_engine.get_metadata("device")

    episodic_memory_store = MemoryStore(
        vector_size,
        episodic_memory_cfg,
        reset=False,
        create=False,
    )

    live_memory_store = MemoryStore(
        vector_size,
        live_memory_cfg,
        reset=False,
        create=False,
    )

    motivators = {}

    for intrinsic in intrinsics:
        motivators[intrinsic] = instrinsic_lookup[intrinsic](
            id=intrinsic_indices[intrinsic],
            live_memory_store=live_memory_store,
            episodic_memory_store=episodic_memory_store,
        )

    while True:
        latent = try_get(motivator_shm["latent"], device)
        robot_state: dict = try_get(motivator_shm["robot_state"])
        governance = try_get(motivator_shm["governance"], device)
        force_on_geoms = try_get(motivator_shm["force_on_geoms"], device)
        emotion = try_get(motivator_shm["emotion"], device)

        if (
            latent is None
            or robot_state is None
            or governance is None
            or force_on_geoms is None
            or emotion is None
        ):
            continue

        # Handle different robot types
        if robot_props.get("data_type") == "simple_agent":
            # Create simple agent physics state
            agent_physics = create_simple_agent_physics(robot_state)
        else:
            # Legacy MuJoCo handling (if needed)
            agent_physics = create_mujoco_physics(robot_state, cfg)

        information = {
            "latent": latent,
            "emotion": emotion,
            "governance": governance,
            "force_on_geoms": force_on_geoms,
            "agent_state": robot_state,  # Add raw agent state
            "raycast_matrices": robot_state.get(
                "raycast_matrices", {}
            ),  # Add raycast data
        }

        for intrinsic in intrinsics:
            motivators[intrinsic].guide(
                information=information,
                brain_shm=brain_shm,
                physics=agent_physics,  # Pass simplified physics
            )


def create_simple_agent_physics(robot_state):
    """
    Create a simplified physics-like object for simple agent
    This replaces the MuJoCo Physics object with agent-specific data
    """

    class SimpleAgentPhysics:
        def __init__(self, agent_state):
            self.agent_state = agent_state

            # Agent position and movement
            self.position = np.array(
                [
                    agent_state.get("x", 0.0),
                    agent_state.get("y", 0.0),
                    agent_state.get("z", 0.0),
                ]
            )

            # Agent velocity
            self.velocity = np.array(
                [
                    agent_state.get("velocity_x", 0.0),
                    agent_state.get("velocity_y", 0.0),
                    0.0,  # No Z velocity for 2D agent
                ]
            )

            # Movement commands (equivalent to control inputs)
            self.movement_commands = np.array(
                [
                    agent_state.get("movement_x", 0.0),
                    agent_state.get("movement_y", 0.0),
                    agent_state.get("orientation", 0.0),
                ]
            )

            # Health/status
            self.hit_points = agent_state.get("hit_point", 100)

            # Raycast/sensor data
            self.raycast_data = agent_state.get("raycast_data", {})
            self.raycast_matrices = agent_state.get("raycast_matrices", {})

            # Create simplified data structure similar to MuJoCo
            self.data = SimpleAgentData(self)

    class SimpleAgentData:
        def __init__(self, physics):
            # Position (equivalent to qpos)
            self.qpos = physics.position

            # Velocity (equivalent to qvel)
            self.qvel = physics.velocity

            # Control inputs (equivalent to ctrl)
            self.ctrl = physics.movement_commands

            # Additional agent-specific data
            self.hit_points = physics.hit_points
            self.raycast_data = physics.raycast_data
            self.raycast_matrices = physics.raycast_matrices

            # Sensor data organized by type
            self.obstacle_distances = np.array(
                physics.raycast_data.get("obstacle_distances", [])
            )
            self.obstacle_angles = np.array(
                physics.raycast_data.get("obstacle_angles", [])
            )
            self.enemy_distances = np.array(
                physics.raycast_data.get("enemy_distances", [])
            )
            self.enemy_angles = np.array(physics.raycast_data.get("enemy_angles", []))
            self.empty_distances = np.array(
                physics.raycast_data.get("empty_distances", [])
            )
            self.empty_angles = np.array(physics.raycast_data.get("empty_angles", []))

            # Summary statistics
            self.total_rays = physics.raycast_data.get("total_rays", 0)
            self.obstacle_count = physics.raycast_data.get("obstacle_count", 0)
            self.enemy_count = physics.raycast_data.get("enemy_count", 0)
            self.empty_count = physics.raycast_data.get("empty_count", 0)

    return SimpleAgentPhysics(robot_state)


def create_mujoco_physics(robot_state, cfg):
    """
    Legacy MuJoCo physics creation (kept for backward compatibility)
    """
    try:
        from dm_control.mujoco import Physics

        physics = Physics.from_xml_path(cfg["robot"]["mjcf_path"])
        qpos = robot_state["qpos"]
        qvel = robot_state["qvel"]

        with physics.reset_context():
            physics.data.qpos[:] = np.array(qpos)
            physics.data.qvel[:] = np.array(qvel)

        return physics
    except Exception as e:
        print(f"Warning: Could not create MuJoCo physics: {e}")
        # Fall back to simple agent physics
        return create_simple_agent_physics(robot_state)
