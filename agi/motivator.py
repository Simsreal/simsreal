from importlib import import_module

from dm_control.mujoco import Physics
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
    physics = Physics.from_xml_path(cfg["robot"]["mjcf_path"])
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

        qpos = robot_state["qpos"]
        qvel = robot_state["qvel"]

        with physics.reset_context():
            physics.data.qpos[:] = np.array(qpos)  # type: ignore
            physics.data.qvel[:] = np.array(qvel)  # type: ignore

        information = {
            "latent": latent,
            "emotion": emotion,
            "governance": governance,
            "force_on_geoms": force_on_geoms,
        }

        for intrinsic in intrinsics:
            motivators[intrinsic].guide(
                information=information,
                brain_shm=brain_shm,
                physics=physics,
            )
