from collections import deque
from typing import Dict

import numpy as np
import torch
from scipy.spatial.distance import cdist
from loguru import logger

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class CognitiveDissonance(Intrinsic):
    k = 10
    alpha = 0.9

    def impl(
        self,
        information: Dict[str, torch.Tensor],
        brain_shm,
        physics=None,
    ):
        if not self.memory_is_available:
            return

        try:
            memory = self.episodic_memory_store.recall_all(payloads=["emotion"])
        except Exception as e:
            logger.warning(e)
            return

        memorized_emotions = np.array(
            [m.payload["emotion"] for m in memory if m.payload is not None],
            dtype=np.float32,
        )

        dist = cdist(
            information["emotion"].clone().cpu().numpy(),
            memorized_emotions,
            metric="cosine",
        )
        closest_indices = np.argsort(dist, axis=1)[:, : self.k][0]
        memories = [memory[i].vector for i in closest_indices]
        avg_memory = np.mean(np.array(memories, dtype=np.float32), axis=0)
        norm_memory = avg_memory / np.linalg.norm(avg_memory)

        if np.isnan(norm_memory).any():
            return

        dissonance = (
            self.alpha * information["latent"].cpu() + (1 - self.alpha) * norm_memory
        )

        brain_shm["latent"].put(dissonance)

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
