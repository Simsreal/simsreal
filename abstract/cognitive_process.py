from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2


class CognitiveProcess(ABC):
    def __init__(self):
        self.sampler = self.create_sampler()
        self.circuit = self.create_circuit()

    def create_sampler(self):
        return SamplerV2()
        # or you can implement your own sampler

    @abstractmethod
    def create_circuit(self) -> QuantumCircuit:
        """
        Create a quantum circuit for the cognitive process.
        """
        pass

    @abstractmethod
    def execute(self):
        """
        Execute the cognitive process.
        """
        pass
