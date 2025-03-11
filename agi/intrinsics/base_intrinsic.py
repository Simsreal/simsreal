from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Deque, Dict

import numpy as np
import torch
from dm_control.utils.inverse_kinematics import IKResult, qpos_from_site_pose

from agi.memory.store import MemoryStore
from src.utilities.emotion.pad import emotion_look_up

# from src.utilities.tools.retry import retry


@dataclass
class MotionCheckpoint:
    reward_emotion: torch.Tensor
    pos: torch.Tensor | np.ndarray
    eta: float


@dataclass
class MotionTrajectory:
    trajectory: Deque[MotionCheckpoint]


class Intrinsic(ABC):
    priority_queue_size = 1000

    def __init__(
        self,
        id,
        live_memory_store,
        episodic_memory_store,
        brain_shm,
    ):
        self.id = id
        self.live_memory_store: MemoryStore = live_memory_store
        self.episodic_memory_store: MemoryStore = episodic_memory_store
        self.brain_shm = brain_shm

        self.emotions = PriorityQueue(maxsize=self.priority_queue_size)

        self.activeness = 0.0
        self.importance = 0.0

    @property
    def memory_is_available(self) -> bool:
        try:
            self.episodic_memory_store.size
            self.live_memory_store.size
            return True
        except Exception:
            return False

    def importance_fn(self) -> float:
        return self.activeness

    def activeness_fn(self, governance: torch.Tensor) -> float:
        return -governance[self.id].item()

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
        information: Dict[str, torch.Tensor],
        physics=None,
    ):
        self.activeness = self.activeness_fn(information["governance"])
        self.importance = self.importance_fn()
        self.impl(information, physics)

    @abstractmethod
    def impl(
        self,
        information: Dict[str, torch.Tensor],
        physics=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def generate_motion_trajectory(self) -> MotionTrajectory:
        raise NotImplementedError
