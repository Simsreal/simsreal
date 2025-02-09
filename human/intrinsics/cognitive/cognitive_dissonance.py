import numpy as np
from scipy.spatial.distance import cdist

from human.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class CognitiveDissonance(Intrinsic):
    k = 10
    alpha = 0.1

    def impl(
        self,
        shm,
        queues,
        physics=None,
    ):
        if not self.memory_is_available:
            return

        try:
            memory = self.episodic_memory_store.recall_all(payloads=["emotion"])
        except Exception:
            return

        memorized_emotions = np.array(
            [m.payload["emotion"] for m in memory if m.payload is not None],
            dtype=np.float32,
        )

        dist = cdist(
            shm["emotions"].clone().numpy(),
            memorized_emotions,
            metric="cosine",
        )
        closest_indices = np.argsort(dist, axis=1)[:, : self.k][0]
        memories = [memory[i].vector for i in closest_indices]
        avg_memory = np.mean(np.array(memories, dtype=np.float32), axis=0)
        norm_memory = avg_memory / np.linalg.norm(avg_memory)
        shm["latent"].copy_(
            self.alpha * shm["latent"] + (1 - self.alpha) * norm_memory,
            non_blocking=True,
        )

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=[])
