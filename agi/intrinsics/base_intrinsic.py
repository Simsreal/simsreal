from abc import ABC, abstractmethod
import random
from dataclasses import dataclass
from queue import Empty, PriorityQueue
from typing import Deque, Dict, Tuple, Any, List

import numpy as np
from loguru import logger
import torch

from agi.memory.store import MemoryStore
from src.utilities.emotion.pad import emotion_look_up
from src.utilities.tools.retry import retry


@dataclass
class MotionCheckpoint:
    reward_emotion: float  # Simplified to single float
    pos: torch.Tensor | np.ndarray
    eta: float


@dataclass
class MotionTrajectory:
    trajectory: Deque[MotionCheckpoint]


@dataclass
class SimulationState:
    """State tracking for MCTS simulation"""
    step: int
    accumulated_reward: float
    context: Dict[str, Any]
    action_sequence: List[str]


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

        self.activeness = 0.0
        self.importance = 0.0
        self.reward = 0.0  # Current step reward
        
        # Enhanced simulation tracking
        self.simulation_history: List[SimulationState] = []
        self.accumulated_simulation_reward = 0.0
        self.simulation_active = False

    @property
    def memory_is_available(self) -> bool:
        try:
            self.episodic_memory_store.size
            self.live_memory_store.size
            return True
        except Exception:
            return False

    def importance_fn(self) -> float:
        """
        decides how much the intrinsic should be prioritized at current time step.
        default to self.activeness.
        """
        return self.activeness

    def activeness_fn(self, governance: torch.Tensor) -> float:
        """
        decides the extent of the intrinsic's influence on the agent.
        default to corresponding governance output.

        :param governance: torch.Tensor
        :return: activeness: float
        """
        return -governance[self.id].item()

    def pad_vector(
        self,
        emotion: str,
        unsqueeze: bool = True,
    ) -> torch.Tensor:
        if unsqueeze:
            return emotion_look_up[emotion].unsqueeze(0)
        return emotion_look_up[emotion]

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
            logger.warning(f"Invalid guidance type: {guidance_type}")

    def start_simulation(self, initial_context: Dict[str, Any]):
        """Start a new MCTS simulation sequence"""
        self.simulation_active = True
        self.simulation_history = []
        self.accumulated_simulation_reward = 0.0
        
        # Store initial state
        initial_state = SimulationState(
            step=0,
            accumulated_reward=0.0,
            context=initial_context.copy(),
            action_sequence=[]
        )
        self.simulation_history.append(initial_state)

    def simulate_step(self, context: Dict[str, Any], action: str, step: int) -> float:
        """Simulate one step in MCTS and return step reward"""
        if not self.simulation_active:
            self.start_simulation(context)
        
        # Evaluate intrinsic for this simulation step
        old_reward = self.reward
        self.impl(context, {}, None)
        step_reward = self.get_reward()
        
        # Accumulate reward
        self.accumulated_simulation_reward += step_reward
        
        # Store simulation state
        sim_state = SimulationState(
            step=step,
            accumulated_reward=self.accumulated_simulation_reward,
            context=context.copy(),
            action_sequence=self.simulation_history[-1].action_sequence + [action]
        )
        self.simulation_history.append(sim_state)
        
        # Restore original reward (don't affect real state)
        self.reward = old_reward
        
        return step_reward

    def get_simulation_reward(self) -> float:
        """Get accumulated reward from current simulation"""
        return self.accumulated_simulation_reward if self.simulation_active else 0.0

    def end_simulation(self) -> float:
        """End simulation and return total accumulated reward"""
        if not self.simulation_active:
            return 0.0
        
        total_reward = self.accumulated_simulation_reward
        self.simulation_active = False
        self.simulation_history = []
        self.accumulated_simulation_reward = 0.0
        
        return total_reward

    def get_simulation_trajectory(self) -> List[SimulationState]:
        """Get the current simulation trajectory"""
        return self.simulation_history.copy()

    def guide(
        self,
        context: Dict[str, Any],
        brain_shm,
        physics=None,
    ):
        # For backward compatibility, if governance is available, use it for activeness
        governance = context.get("governance")
        if governance is not None:
            self.activeness = self.activeness_fn(governance)
        else:
            self.activeness = 1.0  # Default activeness
            
        self.importance = self.importance_fn()
        self.impl(context, brain_shm, physics)

        try:
            emotion_guidance = self.priorities["emotion"].get_nowait()[1]
        except Empty:
            emotion_guidance = None

        if emotion_guidance is not None:
            brain_shm["emotion"].put(emotion_guidance)

    @abstractmethod
    def impl(
        self,
        context: Dict[str, Any],
        brain_shm,
        physics=None,
    ):
        """
        :param context: Dict[str, Any] - Raw context data from the environment
        :param physics: dm_control.physics.Physics
        """
        raise NotImplementedError

    @abstractmethod
    def generate_motion_trajectory(self) -> MotionTrajectory:
        raise NotImplementedError

    def get_reward(self) -> float:
        """
        Return simplified reward as single float in range [-1, +1]
        """
        return max(-1.0, min(1.0, self.reward))
