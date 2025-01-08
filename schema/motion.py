from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class MotionCheckpoint:
    reward_emotion: torch.Tensor
    pos: torch.Tensor | np.ndarray
    eta: float


@dataclass
class MotionTrajectory:
    trajectory: List[MotionCheckpoint]
