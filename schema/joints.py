from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class DJointState:
    name: str
    pos: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    xpos: np.ndarray
    parent_geoms: List[str] | None = field(default=None)
    child_geoms: List[str] | None = field(default=None)
