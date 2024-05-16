# coding=utf-8

import pyqpanda as pq
from pyvqnet import tensor
from pyvqnet.nn.module import Module, ModuleList
from pyvqnet.qnn.measure import ProbsMeasure
from pyvqnet.qnn.quantumlayer import VQCLayer, VQC_wrapper
# from pyvqnet.qnn.quantumlayer import QuantumLayer as QLayer
from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess as QLayer
from pyvqnet.dtype import *


# Ansatz circuit
def get_circuit(depth):


  def _circuit(input, param, num_qubits, _):
    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(num_qubits)

    prog = pq.QProg()
    circuit = pq.QCircuit()

    input = input.squeeze()
    for qubit in qubits:
      circuit.insert(pq.H(qubit))
    for angle, qubit in zip(input, qubits):
      circuit.insert(pq.RY(qubit, angle))

    param = param.reshape([depth, -1])

    for k in range(depth):
      for idx in range(num_qubits - 1):
        circuit.insert(pq.CNOT(qubits[idx], qubits[idx + 1]))
      for idx in range(num_qubits):
        circuit.insert(pq.RY(qubits[idx], param[k, idx]))

    prog.insert(circuit)

    output = []
    for pos in range(num_qubits):
      pauli_str = f'Z{pos}'
      po = pq.PauliOperator(pauli_str, 1)
      hamiltion = po.toHamiltonian(True)
      exp = machine.get_expectation(prog, hamiltion, qubits)
      output.append(exp)


    return sum(output)

  return _circuit

# Quantum depthwise convolution
class QDConv1D(Module):

  def __init__(
    self, ch_in, ch_out, kernel_size, depth, stride=1
  ):
    super(QDConv1D, self).__init__()
    self.ch_in, self.ch_out = ch_in, ch_out

    self.stride = stride
    self.kernel_size = kernel_size

    self.ch_in = self.ch_out = None
    self.kernels = ModuleList(
        [
          QLayer(get_circuit(depth), kernel_size*depth, kernel_size,dtype=kfloat64)

        ]
      )


  def forward(self, x):
    shape = x.shape
    feature_in = shape[-1]
    ch_in = self.ch_in
    ch_out = self.ch_out

    x = tensor.reshape(x, [-1, feature_in])

    feature_out = (feature_in - self.kernel_size + self.stride) // self.stride
    out = []

    for f in range(feature_out):
        o = self.kernels[0](
          x[:, f*self.stride:f*self.stride + self.kernel_size]
        )
        out.append(tensor.squeeze(o))

    out = tensor.stack(out,
                       1) if len(out) > 1 else tensor.reshape(out[0], [-1, 1])

    out = tensor.reshape(out, shape[:-1] + [feature_out])

    return out


# standard quantum convolution
class QConv1D(Module):

  def __init__(
    self, ch_in, ch_out, kernel_size, depth, stride=1, share_weight=False
  ):
    super(QConv1D, self).__init__()

    self.ch_in, self.ch_out = ch_in, ch_out
    self.share_weight = share_weight
    self.stride = stride
    self.kernel_size = kernel_size

    if share_weight:
      self.ch_in = self.ch_out = None



      self.kernels = ModuleList(
        [
          QLayer(get_circuit(depth), kernel_size*depth, kernel_size,dtype=kfloat64)

        ]
      )
    else:
      self.kernels = ModuleList(
        [
          QLayer(
            get_circuit(depth), kernel_size*depth,
            kernel_size,dtype=kfloat64
          )

           for _ in range(ch_in*ch_out)
        ]
      )

  def forward(self, x):

    shape = x.shape

    feature_in = shape[-1]
    ch_in = self.ch_in
    ch_out = self.ch_out


    x = tensor.reshape(x, [-1, ch_in, feature_in])
    feature_out = (feature_in - self.kernel_size + self.stride) // self.stride
    outputs = tensor.zeros([shape[0],ch_out,feature_out],dtype=kfloat64)


    for c_out in range(ch_out):

        for f in range(feature_out):
            output = 0
            for c_in in range(ch_in):
              o = self.kernels[c_in+c_out*ch_out](
                x[:, c_in, f*self.stride:f*self.stride + self.kernel_size]
              )

              output += o

            outputs[:,c_out,f] = output


    return outputs
