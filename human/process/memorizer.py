import time

from human.memory.store import MemoryStore


def memory_manager_proc(runtime_engine, mem_type):
    cfg = runtime_engine.get_metadata("config")
    latent_dim = runtime_engine.get_shm("latent").shape[-1]
    mem_manage_cfg = cfg["memory_management"]
    memory_cfg = mem_manage_cfg[mem_type]

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

    while True:
        if time.time() - last_memory > 1 / memory_cfg["hz"]:
            last_memory = time.time()
            if mem_type == "live_memory":
                try:
                    id = int(time.time() * 10e6)
                    memory.memorize(
                        id=id,
                        latent=runtime_engine.get_shm("latent")
                        .squeeze(0)
                        .numpy()
                        .tolist(),
                        emotion=runtime_engine.get_shm("emotions")
                        .squeeze(0)
                        .numpy()
                        .tolist(),
                        efforts=runtime_engine.get_shm("torques")
                        .squeeze(0)
                        .numpy()
                        .tolist(),
                    )
                except Exception as e:
                    print(f"memory loss: {e}")

            if live_memory is not None:
                consolidated = live_memory.consolidate("emotion_intesity")
                if len(consolidated) and len(consolidated[0]):
                    memory.memorize_points(consolidated[0])  # type: ignore

        if time.time() - last_memory_decay > memory_cfg["decay_every"]:
            last_memory_decay = time.time()
            if mem_type == "live_memory":
                memory.decay_on_retain_time()
            elif mem_type == "episodic_memory":
                memory.decay_on_capacity("emotion_intesity")

            else:
                print("Unable to decay memory")
