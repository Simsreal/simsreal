from collections import deque

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class OrientingReflex(Intrinsic):
    def impl(self, information, brain_shm):
        pass

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
