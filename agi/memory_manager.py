import time

import torch
from loguru import logger

from agi.memory.store import MemoryStore
from src.utilities.queues.queue_util import try_get


def memory_manager(runtime_engine, mem_type):
    cfg = runtime_engine.get_metadata("config")
    latent_dim = runtime_engine.get_metadata("latent_size")
    mem_manage_cfg = cfg["memory_management"]
    memory_cfg = mem_manage_cfg[mem_type]
    robot_props = runtime_engine.get_metadata("robot_props")
    device = runtime_engine.get_metadata("device")
    latent_slices = runtime_engine.get_metadata("latent_slices")
    memory_manager_shm = runtime_engine.get_shared_memory("memory_manager")
    brain_shm = runtime_engine.get_shared_memory("brain")
    motivator_shm = runtime_engine.get_shared_memory("motivator")
    governor_shm = runtime_engine.get_shared_memory("governor")

    memory = MemoryStore(
        latent_dim,
        memory_cfg,
        reset=memory_cfg["reset"],
    )

    live_memory = None
    if mem_type == "episodic_memory":
        live_memory = MemoryStore(
            latent_dim,
            mem_manage_cfg["live_memory"],
            reset=False,
            create=False,
        )

    last_memory = 0
    last_memory_decay = 0

    latent = torch.zeros(
        (
            1,
            latent_dim,
        ),
        dtype=torch.float32,
    )

    emotion = torch.zeros(
        (
            1,
            cfg["emotion"]["pad_dim"],
        ),
        dtype=torch.float32,
    )

    # Handle different robot types
    if robot_props.get("data_type") == "simple_agent":
        # For simple agent, use movement commands instead of torques
        # [movement_x, movement_y, orientation]
        movement_commands = torch.zeros(
            (
                1,
                3,  # movement_x, movement_y, orientation
            ),
            dtype=torch.float32,
        )
        logger.info("Using simple agent mode - movement commands (x, y, orientation)")
    else:
        # For MuJoCo robots with actuators
        movement_commands = torch.zeros(
            (
                1,
                robot_props.get("n_actuators", 1),  # Default to 1 if not found
            ),
            dtype=torch.float32,
        )
        logger.info(f"Using robot mode - {robot_props.get('n_actuators', 1)} actuators")

    while True:
        if time.time() - last_memory > 1 / memory_cfg["hz"]:
            last_memory = time.time()
            if mem_type == "live_memory":
                try:
                    id = int(time.time() * 10e6)
                    vision_latent = try_get(memory_manager_shm["vision_latent"], device)
                    emerged_emotion = try_get(memory_manager_shm["emotion"], device)
                    emerged_movement = try_get(
                        memory_manager_shm["torque"], device
                    )  # Still using "torque" queue name for compatibility

                    if vision_latent is not None:
                        latent[latent_slices["vision"]] = vision_latent

                    if emerged_emotion is not None:
                        emotion = emerged_emotion.clone()

                    if emerged_movement is not None:
                        # Handle movement commands for simple agent
                        if robot_props.get("data_type") == "simple_agent":
                            # Expect movement commands: [movement_x, movement_y, orientation]
                            if emerged_movement.shape[-1] >= 3:
                                movement_commands = emerged_movement[:, :3].clone()
                            elif emerged_movement.shape[-1] == 2:
                                # If only 2D movement, pad with zero orientation
                                movement_commands = torch.cat(
                                    [
                                        emerged_movement,
                                        torch.zeros(
                                            emerged_movement.shape[0],
                                            1,
                                            device=emerged_movement.device,
                                        ),
                                    ],
                                    dim=-1,
                                )
                            else:
                                # If single value, assume it's movement_x
                                movement_commands = torch.cat(
                                    [
                                        emerged_movement,
                                        torch.zeros(
                                            emerged_movement.shape[0],
                                            2,
                                            device=emerged_movement.device,
                                        ),
                                    ],
                                    dim=-1,
                                )
                        else:
                            movement_commands = emerged_movement.clone()

                    memory.memorize(
                        id=id,
                        latent=latent.squeeze(0).cpu().numpy().tolist(),
                        emotion=emotion.squeeze(0).cpu().numpy().tolist(),
                        efforts=movement_commands.squeeze(0)
                        .cpu()
                        .numpy()
                        .tolist(),  # Store movement commands as "efforts"
                    )
                    brain_shm["latent"].put(latent)
                    motivator_shm["latent"].put(latent)
                    motivator_shm["emotion"].put(emotion)
                    governor_shm["emotion"].put(emotion)

                except Exception as e:
                    logger.warning(f"memory loss: {e}")

            if live_memory is not None:
                consolidated = live_memory.consolidate("emotion_intensity")
                if len(consolidated) and len(consolidated[0]):
                    memory.memorize_points(consolidated[0])  # type: ignore

        if time.time() - last_memory_decay > memory_cfg["decay_every"]:
            last_memory_decay = time.time()
            if mem_type == "live_memory":
                memory.decay_on_retain_time()
            elif mem_type == "episodic_memory":
                memory.decay_on_capacity("emotion_intensity")
            else:
                logger.warning("Unable to decay memory")
