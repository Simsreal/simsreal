from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict

from intelligence.context import Context


class ExecutionStatus(Enum):
    NOT_STARTED = auto()
    EMPTY_PLAN = auto()
    SUCCESS = auto()
    INVALID_PLAN = auto()
    FAILURE = auto()


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: Dict[str, Context]
