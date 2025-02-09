import time

from dm_control.mujoco import Physics

from human.intrinsics import (
    Boredom,
    CognitiveDissonance,
    FearOfPain,
    FearOfUnknown,
    MereExposure,
)
from human.memory.store import MemoryStore


def motivator_proc(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    instrinsic_lookup = {
        "fear_of_pain": FearOfPain,
        "mere_exposure": MereExposure,
        "fear_of_unknown": FearOfUnknown,
        "cognitive_dissonance": CognitiveDissonance,
        "boredom": Boredom,
    }
    intrinsics = cfg["intrinsics"]
    physics = Physics.from_xml_path(cfg["robot"]["mjcf_path"])
    intrinsic_indices = runtime_engine.get_metadata("intrinsic_indices")

    episodic_memory_cfg = cfg["memory_management"]["episodic_memory"]
    live_memory_cfg = cfg["memory_management"]["live_memory"]
    vector_size = runtime_engine.get_shm("latent").shape[-1]

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
        with physics.reset_context():
            physics.data.qpos[:] = runtime_engine.get_shm("qpos").numpy()  # type: ignore
            physics.data.qvel[:] = runtime_engine.get_shm("qvel").numpy()  # type: ignore

        for intrinsic in intrinsics:
            motivators[intrinsic].guide(
                runtime_engine.shared_memory,
                runtime_engine.shared_queues,
                physics,
            )
        time.sleep(1 / cfg["running_frequency"])
