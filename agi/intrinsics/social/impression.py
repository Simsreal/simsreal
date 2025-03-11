from collections import deque
from typing import Dict

import torch
from loguru import logger

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class Impression(Intrinsic):
    number_of_recall = 5

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

        except Exception as e:
            logger.warning(e)
            return

        emotions_tensor = torch.tensor(
            [pt.payload["emotion"] for pt in recalled if pt.payload is not None],
            dtype=torch.float32,
        )

        if emotions_tensor.size(0) == 0:
            return

        emotions = torch.mean(emotions_tensor, dim=0).unsqueeze(0)
        self.brain_shm["emotion"].put(
            emotions * self.activeness_fn(information["governance"])
        )

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
