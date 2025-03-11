from collections import deque
from typing import Dict

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfPain(Intrinsic):
    """
    TODO:
    instead of a hardcoded acceptable force
    acceptable forces to be defined thoroughly depending robot states.
    just make sure they are measured in Newtons with reliable contact sensors.
    """

    acceptable_forceN = 100

    def impl(
        self,
        information: Dict[str, torch.Tensor],
        physics=None,
    ):
        forces = information["force_on_geoms"] > self.acceptable_forceN
        painful = torch.any(forces).item()
        self.brain_shm["emotion"].put(
            self.pad_vector("fearful" if painful else "neutral")
        )

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
