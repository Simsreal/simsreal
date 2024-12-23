from dataclasses import dataclass

import numpy as np


@dataclass
class DGeom:
    name: str
    geom_id: int
    body_id: int
    geom_type: int
    size: np.ndarray
    friction: np.ndarray
    rgba: np.ndarray
    xpos: np.ndarray
    xmat: np.ndarray
    # quat: np.ndarray
