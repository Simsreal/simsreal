from abc import ABC
from abc import abstractmethod
import random
from dataclasses import dataclass
from queue import Empty, PriorityQueue
from typing import Deque, Dict, Tuple

import numpy as np
import torch
from dm_control.utils.inverse_kinematics import IKResult
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

from human.memory.store import MemoryStore
from utilities.emotions.pad import emotion_look_up
from utilities.tools.retry import retry


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

    def __init__(self, id, live_memory_store, episodic_memory_store):
        self.id = id
        self.live_memory_store: MemoryStore = live_memory_store
        self.episodic_memory_store: MemoryStore = episodic_memory_store

        self.priorities: Dict[
            str, PriorityQueue[Tuple[float, torch.Tensor | MotionTrajectory]]
        ] = {
            "emotion": PriorityQueue(maxsize=self.priority_queue_size),
            "motion": PriorityQueue(maxsize=self.priority_queue_size),
        }

        # runtime
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

    def importance_fn(
        self,
        shm,
    ) -> float:
        """
        decides how much the intrinsic should be prioritized at current time step.
        default to self.activeness.
        """
        return -self.activeness

    def activeness_fn(self, shm) -> float:
        """
        decides the extent of the intrinsic's influence on the agent.
        default to corresponding governance output.

        :param shm: Dict[str, torch.Tensor]
        :return: activeness: float
        """
        return -shm["governance"][self.id].item()

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

    @retry
    def add_guidance(
        self,
        guidance_type: str,
        guidance: torch.Tensor | str,
    ):
        if guidance_type == "emotion":
            if isinstance(guidance, str):
                emotion = self.pad_vector(guidance)
            else:
                emotion = guidance
            self.priorities[guidance_type].put(
                (
                    self.importance + random.random() * 1e-9,
                    emotion * self.activeness,
                )
            )
        else:
            print(f"Invalid guidance type: {guidance_type}")

    def guide(
        self,
        runtime_engine,
        physics=None,
    ):
        shm = runtime_engine.shared_memory
        guidances = runtime_engine.shared_guidances

        self.activeness = self.activeness_fn(shm)
        self.importance = self.importance_fn(shm)
        self.impl(shm, guidances, physics)

        try:
            emotion_guidance = self.priorities["emotion"].get_nowait()[1]
        except Empty:
            emotion_guidance = None

        print(self.priorities["emotion"].queue)
        # try:
        #     motion_guidance = self.priorities["motion"].get_nowait()[1]
        # except Empty:
        #     motion_guidance = None

        if emotion_guidance is not None:
            guidances["emotion"].put(emotion_guidance)

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
