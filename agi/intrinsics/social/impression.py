from collections import deque
import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class Impression(Intrinsic):
    number_of_recall = 5

    def impl(
        self,
        shm,
        guidances,
        physics=None,
    ):
        if not self.memory_is_available:
            return

        try:
            recalled = self.episodic_memory_store.recall(
                shm["latent"].squeeze(0).numpy().tolist(), self.number_of_recall
            )
        except Exception:
            return

        emotions_tensor = torch.tensor(
            [pt.payload["emotion"] for pt in recalled if pt.payload is not None],
            dtype=torch.float32,
        )

        if emotions_tensor.size(0) == 0:
            return

        emotions = torch.mean(emotions_tensor, dim=0).unsqueeze(0)
        self.add_guidance("emotion", emotions * self.activeness_fn(shm))

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
