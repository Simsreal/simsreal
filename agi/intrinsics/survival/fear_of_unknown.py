from typing import Dict

from collections import deque

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfUnknown(Intrinsic):
    min_familiarity_wanted = 0.3
    number_of_recall = 10

    def impl(
        self,
        information: Dict[str, torch.Tensor],
        physics=None,
    ):
        if not self.memory_is_available:
            return

        try:
            recalled = self.episodic_memory_store.recall(
                information["latent"].squeeze(0).cpu().numpy().tolist(),
                self.number_of_recall,
            )
        except Exception:
            return

        familiarities = torch.tensor([pt.score for pt in recalled], dtype=torch.float32)
        familarity = torch.mean(familiarities)

        if torch.isnan(familarity):
            return

        self.brain_shm["emotion"].put(
            self.pad_vector(
                "fearful"
                if familarity.item() < self.min_familiarity_wanted
                else "neutral",
            )
        )

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
