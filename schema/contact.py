from dataclasses import dataclass

import numpy as np


@dataclass
class MujocoContact:
    H: np.ndarray
    dim: int
    dist: float
    efc_address: int
    elem: np.ndarray
    exclude: int
    flex: np.ndarray
    frame: np.ndarray
    friction: np.ndarray
    geom: np.ndarray
    geom1: int
    geom2: int
    includemargin: float
    mu: float
    pos: np.ndarray
    solimp: np.ndarray
    solref: np.ndarray
    solreffriction: np.ndarray
    vert: np.ndarray
