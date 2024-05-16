# coding=utf-8

import pyqpanda as pq
from pyvqnet.nn.module import Module
from pyvqnet.qnn.measure import ProbsMeasure
from pyvqnet.qnn.quantumlayer import VQCLayer, VQC_wrapper
# from pyvqnet.qnn.quantumlayer import QuantumLayer as QLayer
from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess as QLayer
from pyvqnet.dtype import *


# Ansatz circuit
def get_circuit(num_class):

  def _circuit(input, param, num_qubits, _):
    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)

    prog = pq.QProg()
    circuit = pq.QCircuit()

    encoder = pq.Encode()
    encoder.angle_encode(qubits, input)
    circuit.insert(encoder.get_circuit())

    for idx in range(num_qubits + 1):
      c_qubit = qubits[idx%num_qubits]
      circuit.insert(pq.RY(c_qubit, param[idx]))
      t_qubit = qubits[(idx%num_qubits + num_class)%num_qubits]
      circuit.insert(pq.CNOT(c_qubit, t_qubit))

    prog.insert(circuit)

    output = []
    for pos in range(num_class):
      pauli_str = f'Z{pos}'
      po = pq.PauliOperator(pauli_str, 1)
      hamiltion = po.toHamiltonian(True)
      exp = machine.get_expectation(prog, hamiltion, qubits)
      output.append(exp)


    return output

  return _circuit

# Quantum FC layer
class QLinear(Module):

  def __init__(self, input_channels, output_channels):
    super(QLinear, self).__init__()
    self.linear = QLayer(
      get_circuit(output_channels), input_channels + 1, input_channels,dtype=kfloat64
    )


  def forward(self, x):
    o = self.linear(x)
    return o
