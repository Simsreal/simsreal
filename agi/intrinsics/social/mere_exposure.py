from collections import deque

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class MereExposure(Intrinsic):
    number_of_recall = 5
    acceptance_threshold = 0.9
    exposure_weight = 0.1

    def impl(
        self,
        information,
        brain_shm,
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

        emotions_tensor = torch.tensor(
            [pt.payload["emotion"] for pt in recalled if pt.payload is not None],
            dtype=torch.float32,
        )

        if torch.isnan(familarity):
            return

        if emotions_tensor.size(0) == 0:
            return

        emotions = torch.mean(emotions_tensor, dim=0).unsqueeze(0)
        emotions = (
            self.exposure_weight * emotions
            + (1 - self.exposure_weight) * information["emotion"].cpu()
        )
        self.add_guidance(
            "emotion", emotions * self.activeness_fn(information["governance"])
        )

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
