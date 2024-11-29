from dataclasses import dataclass
from typing import Deque, Dict, List

from neuro_symbol.neuro_symbol import NeuroSymbol
from neuro_symbol.symoblic_action import SymbolicAction
from schema.execution import ExecutionStatus


@dataclass
class Plan:
    ctx_name: str
    guide: str
    plan: Deque[SymbolicAction]
    plan_size: int
    cost: int
    consciousness: Dict[str, NeuroSymbol]
    completed: int
    results: List[ExecutionStatus]
    stats: Dict[ExecutionStatus, int]
