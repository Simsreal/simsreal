from typing import List

from pydantic import BaseModel


class IKComputeRequest(BaseModel):
    timestamp: float
    site_name: str
    target_pos: List[float]
    joint_names: List[str]


class IKComputeResponse(BaseModel):
    timestamp: float
    qpos: List[float]
    err_norm: float
    steps: int
    success: bool
