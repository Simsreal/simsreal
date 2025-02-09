# import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
from dm_control.utils.inverse_kinematics import IKResult, qpos_from_site_pose

from human.memory.store import MemoryStore
from utilities.emotions.pad import emotion_look_up


@dataclass
class MotionCheckpoint:
    reward_emotion: torch.Tensor
    pos: torch.Tensor | np.ndarray
    eta: float


@dataclass
class MotionTrajectory:
    trajectory: Deque[MotionCheckpoint]


class Intrinsic(ABC):
    def __init__(self, id, live_memory_store, episodic_memory_store):
        self.id = id
        self.live_memory_store: MemoryStore = live_memory_store
        self.episodic_memory_store: MemoryStore = episodic_memory_store

        self.priorities: List[Tuple[float, MotionTrajectory]] = []

    @property
    def memory_is_available(self) -> bool:
        try:
            self.episodic_memory_store.size
            self.live_memory_store.size
            return True
        except Exception:
            return False

    def importance(self, shm) -> float:
        """
        decides how much the intrinsic should be prioritized.

        If not implemented, treat activeness as importance.
        """
        return self.activeness(shm)

    def activeness(self, shm) -> float:
        """
        intrinsic governance output.

        :param shm: Dict[str, torch.Tensor]
        :return: activeness: float
        """
        return shm["governance"][self.id].item()

    def pad_vector(
        self,
        emotion: str,
        unsqueeze: bool = True,
    ) -> torch.Tensor:
        if unsqueeze:
            return emotion_look_up[emotion].unsqueeze(0)
        return emotion_look_up[emotion]

    def solve_ik(
        self,
        physics,
        site_name,
        target_pos,
        joint_names,
    ) -> IKResult:
        return qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            joint_names=joint_names,
        )

    def guide(
        self,
        shm,
        guidances,
        physics=None,
    ):
        self.impl(shm, guidances, physics)

    @abstractmethod
    def impl(
        self,
        shm,
        guidances,
        physics=None,
    ):
        """
        :param shm: Dict[str, torch.Tensor]
        :param guidances: Dict[str, mp.Queue]
        :param physics: dm_control.physics.Physics
        """
        raise NotImplementedError

    @abstractmethod
    def generate_motion_trajectory(self) -> MotionTrajectory:
        raise NotImplementedError
