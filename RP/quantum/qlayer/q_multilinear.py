# coding=utf-8

import numpy as np
import pyqpanda as pq
from pyvqnet.nn.module import Module
from pyvqnet.qnn.measure import ProbsMeasure
# from pyvqnet.qnn.quantumlayer import QuantumLayer as QLayer
from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess as QLayer

def get_circuit(nlayer, num_class):

  # def _circuit(input, param, qubits, _, machine):
  #   num_qubits = len(qubits)

  # MultiProcess Part
  def _circuit(input, param, num_qubits, _):
    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)

    prog = pq.QProg()
    circuit = pq.QCircuit()

    input = np.pad(
      input, pad_width=[[0, 2**num_qubits - len(input)]], mode='constant'
    )
    circuit.insert(
      pq.amplitude_encode(qubits, input, b_need_check_normalization=False)
    )

    stride = 2
    for l in range(nlayer):
      for idx in range(num_qubits + 1):
        c_qubit = qubits[idx%num_qubits]
        circuit.insert(pq.RY(c_qubit, param[idx + l*(num_qubits + 1)]))
        t_qubit = qubits[(idx%num_qubits + stride)%num_qubits]
        circuit.insert(pq.CNOT(c_qubit, t_qubit))

    prog.insert(circuit)

    output = []
    for pos in range(num_class):
      pauli_str = f'Z{pos}'
      po = pq.PauliOperator(pauli_str, 1)
      hamiltion = po.toHamiltonian(True)
      exp = machine.get_expectation(prog, hamiltion, qubits)
      output.append(exp)
    # for pos in range(num_class):
    #   prob = ProbsMeasure([pos], prog, machine, qubits)
    #   output.append(prob[0])

    return output

  return _circuit

class QMultiLinear(Module):

  def __init__(self, nlayer, emb_size, num_class):
    super(QMultiLinear, self).__init__()
    self.multilinear = QLayer(
      get_circuit(nlayer, num_class), nlayer*(emb_size + 1), 'cpu', emb_size
    )

  def forward(self, x):
    x = self.multilinear(x)
    return x
