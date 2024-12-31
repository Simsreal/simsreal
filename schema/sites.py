from dataclasses import dataclass

import numpy as np


@dataclass
class DSite:
    name: str
    site_id: int
    body_id: int
    size: np.ndarray
    rgba: np.ndarray
    xpos: np.ndarray
    xmat: np.ndarray
