from typing import List

from pydantic import BaseModel


class IKComputeRequest(BaseModel):
    """
    Requierd arguments:
    timestamp: float
    qpos: List[float]
    qvel: List[float]
    site_name: str
    target_pos: List[float]
    joint_names: List[str]
    """

    timestamp: float
    qpos: List[float]
    qvel: List[float]
    site_name: str
    target_pos: List[float]
    joint_names: List[str]


class IKComputeResponse(BaseModel):
    timestamp: float
    qpos: List[float]
    err_norm: float
    steps: int
    success: bool
