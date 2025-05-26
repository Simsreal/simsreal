from importlib import import_module

import numpy as np

from agi.memory.store import MemoryStore
from src.utilities.queues.queue_util import try_get


def motivator(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
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
        emotion = try_get(motivator_shm["emotion"], device)

        if (
            latent is None
            or robot_state is None
            or governance is None
            or force_on_geoms is None
            or emotion is None
        ):
            continue

        # FIXME: unused variables/information, remove if not needed
        # not sure if these are needed
        # x, y, z, hit_point
        # pass exact coordinates to motivators may impact generalization, ignore?
        x = robot_state["x"]
        y = robot_state["y"]
        z = robot_state["z"]
        line_of_sight = robot_state["line_of_sight"]

        information = {
            "latent": latent,
            "emotion": emotion,
            "governance": governance,
            
        }

        for intrinsic in intrinsics:
            motivators[intrinsic].guide(
                information=information,
                brain_shm=brain_shm,
            )
