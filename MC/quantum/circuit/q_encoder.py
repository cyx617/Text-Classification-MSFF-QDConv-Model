# coding=utf-8

import numpy as np
import pyqpanda as pq

def encode(x, num_qubits):
  machine = pq.CPUQVM()
  machine.init_qvm()
  qubits = machine.qAlloc_many(num_qubits)

  x = np.pad(
    x, pad_width=[[0, 0], [0, 2**num_qubits - x.shape[1]]], mode='constant'
  )

  output = []
  for i in range(x.shape[0]):
    prog = pq.QProg()
    circuit = pq.QCircuit()

    circuit.insert(
      pq.amplitude_encode(qubits, x[i, :], b_need_check_normalization=False)
    )

    prog.insert(circuit)

    exps = []
    for pos in range(num_qubits):
      pauli_str = f'Z{pos}'
      po = pq.PauliOperator(pauli_str, 1)
      hamiltion = po.toHamiltonian(True)
      exp = machine.get_expectation(prog, hamiltion, qubits)
      exps.append(exp)

    output.append(exps)

  output = np.asarray(output)

  return output
