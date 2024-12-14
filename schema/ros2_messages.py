from dataclasses import dataclass
from typing import List


@dataclass
class JointStateMessage:
    name: List[str] = [
        "pelvis",
        "right_lower_arm",
        "left_lower_arm",
        "right_shin",
        "left_shin",
    ]
    position: List[float]
    velocity: List[float]
    effort: List[float]
