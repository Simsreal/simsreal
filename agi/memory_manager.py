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
    # robot_props = runtime_engine.get_metadata("robot_props")
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

    torque = torch.zeros(
        (
            1,
            # robot_props["n_actuators"],
            3,  # treat output as x, y, and orientation
        ),
        dtype=torch.float32,
    )

    while True:
        if time.time() - last_memory > 1 / memory_cfg["hz"]:
            last_memory = time.time()
            if mem_type == "live_memory":
                try:
                    id = int(time.time() * 10e6)
                    vision_latent = try_get(memory_manager_shm["vision_latent"], device)
                    emerged_emotion = try_get(memory_manager_shm["emotion"], device)
                    emerged_command = try_get(memory_manager_shm["command"], device)

                    # TODO: change the vision to new format
                    if vision_latent is not None:
                        latent[latent_slices["vision"]] = vision_latent

                    if emerged_emotion is not None:
                        emotion = emerged_emotion.clone()

                    if emerged_command is not None:
                        command = emerged_command.clone()

                    memory.memorize(
                        id=id,
                        latent=latent.squeeze(0).cpu().numpy().tolist(),
                        emotion=emotion.squeeze(0).cpu().numpy().tolist(),
                        efforts=command.squeeze(0).cpu().numpy().tolist(), # not sure, the command is not a physical matrix
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
