from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dm_control.utils.inverse_kinematics import IKResult
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

from human.memory.store import MemoryStore
from utilities.emotion.pad import emotion_look_up


@dataclass
class MotionCheckpoint:
    reward_emotion: torch.Tensor
    pos: torch.Tensor | np.ndarray
    eta: float


@dataclass
class MotionTrajectory:
    trajectory: List[MotionCheckpoint]


class Intrinsic(ABC):
    def __init__(self, id, live_memory_store, episodic_memory_store):
        self.id = id
        self.live_memory_store: MemoryStore = live_memory_store
        self.episodic_memory_store: MemoryStore = episodic_memory_store

        self.motion_trajectory: MotionTrajectory | None = None

    @property
    def memory_is_available(self) -> bool:
        try:
            self.episodic_memory_store.size
            self.live_memory_store.size
            return True
        except Exception:
            return False

    def activeness(self, shm) -> float:
        return shm["governance"][self.id].item()

    def pad_vector(
        self,
        emotion,
        unsqueeze=True,
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

    @abstractmethod
    def impact(
        self,
        shm,
        queues,
        physics=None,
    ):
        # physics is useful for motion generation
        raise NotImplementedError

    @abstractmethod
    def generate_motion_trajectory(self) -> MotionTrajectory:
        raise NotImplementedError
