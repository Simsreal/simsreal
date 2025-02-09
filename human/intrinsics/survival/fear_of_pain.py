import torch

from human.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfPain(Intrinsic):
    """
    TODO:
    instead of a hardcoded acceptable force
    acceptable forces to be defined thoroughly depending robot states.
    just make sure they are measured in Newtons with reliable contact sensors.
    """

    acceptable_forceN = 100

    def impact(
        self,
        shm,
        queues,
        physics=None,
    ):
        forces = shm["force_on_geoms"] > self.acceptable_forceN
        painful = torch.any(forces).item()
        queues["emotions_q"].put(
            self.pad_vector("fearful" if painful else "neutral") * self.activeness(shm)
        )

    def generate_motion_trajectory(self):
        return MotionTrajectory(trajectory=[])
