# coding=utf-8

import numpy as np
import pyqpanda as pq
from pyvqnet.nn.module import Module
from pyvqnet.qnn.measure import ProbsMeasure
# from pyvqnet.qnn.quantumlayer import QuantumLayer as QLayer
from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess as QLayer
from pyvqnet.qnn.template import BasicEntanglerTemplate, StronglyEntanglingTemplate
from pyvqnet.dtype import *

# Ansatz circuit
def get_circuit(enttype):


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
    for qubit in qubits:
      circuit.insert(pq.H(qubit))

    if enttype == 0:
      # Basic Ent Version
      weights = param.reshape([-1, num_qubits])
      # template = BasicEntanglerTemplate(
      #   weights, num_qubits=num_qubits, rotation=pq.RY
      # )
      for l in range(weights.shape[0]):
        for idx in range(num_qubits):
          circuit.insert(pq.RY(qubits[idx], weights[l, idx]))
        for idx in range(num_qubits):
          circuit.insert(pq.CNOT(qubits[idx], qubits[(idx + l+1)%num_qubits]))
    else:
      # Strong Ent Version
      weights = param.reshape([-1, num_qubits, 3])
      # template = StronglyEntanglingTemplate(weights, num_qubits=num_qubits)
      for l in range(weights.shape[0]):
        for idx in range(num_qubits):
          circuit.insert(pq.RY(qubits[idx], weights[l, idx, 0]))
          circuit.insert(pq.RZ(qubits[idx], weights[l, idx, 1]))
          #circuit.insert(pq.RZ(qubits[idx], weights[l, idx, 2]))
        for idx in range(num_qubits):
          circuit.insert(pq.CNOT(qubits[idx], qubits[(idx + l + 1)%num_qubits]))
    # circuit.insert(template.create_circuit(qubits))

    prog.insert(circuit)

    output = []
    for pos in range(num_qubits):
      pauli_str = f'Z{pos}'
      po = pq.PauliOperator(pauli_str, 1)
      hamiltion = po.toHamiltonian(True)
      exp = machine.get_expectation(prog, hamiltion, qubits)
      output.append(exp)


    return output

  return _circuit

# Quantum embedding
class QEmbedding(Module):

  def __init__(self, rep, emb_size, enttype=0):
    super(QEmbedding, self).__init__()
    num_rotate_gate = 1 if enttype == 0 else 3
    self.embedding = QLayer(
      get_circuit(enttype), rep*emb_size*num_rotate_gate,  emb_size,dtype=kfloat64
    )

  def forward(self, x):
    x = self.embedding(x)
    return x
