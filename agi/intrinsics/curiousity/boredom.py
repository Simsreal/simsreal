from collections import deque

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class Boredom(Intrinsic):
    boredom_threshold = 0.05

    def impl(
        self,
        shm,
        guidances,
        physics=None,
    ):
        if not self.memory_is_available:
            return

        try:
            memory = self.live_memory_store.recall_all(["emotion"])
        except Exception:
            return

        emotions = torch.tensor(
            [m.payload["emotion"] for m in memory if m.payload is not None],
            dtype=torch.float32,
        )
        avg_emotion = torch.mean(emotions, dim=0)
        neutral_emotion = self.pad_vector("neutral", unsqueeze=False)

        dist_to_boredom = torch.norm(avg_emotion - neutral_emotion, p=2)
        if dist_to_boredom < self.boredom_threshold:
            self.add_guidance("emotion", "bored")

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
