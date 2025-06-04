from collections import deque
from typing import Dict

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfPain(Intrinsic):
    """
    Fear of pain intrinsic that responds to agent damage state.
    When agent state = 1, it indicates the agent is experiencing pain/damage.
    """

    def impl(
        self,
        information: Dict[str, torch.Tensor],
        brain_shm,
        physics=None,
    ):
        # Check if agent_state is available and indicates pain (state = 1)
        agent_state = information.get("agent_state", 0)

        # Convert to scalar if it's a tensor
        if torch.is_tensor(agent_state):
            state_value = agent_state.item()
        else:
            state_value = agent_state

        # State = 1 indicates pain/damage
        painful = state_value == 1

        self.add_guidance("emotion", "fearful" if painful else "neutral")

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
