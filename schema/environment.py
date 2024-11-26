from typing import Any, Dict, List

from pydantic import BaseModel


class Landmark(BaseModel):
    name: str
    configuration: Dict[str, Any]


class Landmarks(BaseModel):
    landmarks: List[Landmark]


class EnvironmentConfig(BaseModel):
    humans: Dict[str, Any]
    constraints: Dict[str, Any]
    environment: Dict[str, Any]
