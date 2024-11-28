from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ExecutionStatus(Enum):
    NOT_STARTED = auto()
    EMPTY_PLAN = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: Any
