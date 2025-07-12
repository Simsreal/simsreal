from collections import deque
from typing import Dict, Any

import torch

from agi.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfPain(Intrinsic):
    """
    Fear of pain intrinsic that responds to agent damage/status state.
    Reward system:
    - Status 0 (normal): +0.1 (slight positive for being healthy)
    - Status 1 (fell down): -0.5 (moderate negative, recoverable)
    - Status 2 (won): +1.0 (maximum positive reward)
    - Status 3 (dead): -1.0 (maximum negative reward)
    """

    def impl(
        self,
        context: Dict[str, Any],
        brain_shm,
        physics=None,
    ):
        # Get agent state from raw context or parsed context
        raw_context = context.get("raw_context", context)
        agent_state = raw_context.get("state", 0)

        # Convert to scalar if it's a tensor
        if torch.is_tensor(agent_state):
            state_value = agent_state.item()
        else:
            state_value = int(agent_state)

        # Calculate reward based on status
        if state_value == 0:  # Normal
            self.reward = 0.1
            emotion = "neutral"
        elif state_value == 1:  # Fell down (recoverable)
            self.reward = -0.5
            emotion = "fearful"
        elif state_value == 2:  # Won
            self.reward = 1.0
            emotion = "joyful"
        elif state_value == 3:  # Dead (unrecoverable)
            self.reward = -1.0
            emotion = "fearful"
        else:  # Unknown state
            self.reward = 0.0
            emotion = "neutral"

        # Add emotional guidance
        self.add_guidance("emotion", emotion)

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=deque())
