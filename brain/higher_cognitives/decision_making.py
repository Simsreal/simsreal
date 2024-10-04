import qiskit

from abstract.cognitive_process import CognitiveProcess


class DecisionMaking(CognitiveProcess):
    def __init__(self):
        super().__init__()

    def create_circuit(self):
        circuit = qiskit.QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure_all()
        return circuit

    def execute(self):
        job = self.sampler.run([self.circuit], shots=128)
        result = job.result()
        counts = result[0].data.meas.get_counts()
        print(counts)
