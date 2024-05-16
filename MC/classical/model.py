# coding=utf-8

from pyvqnet import tensor
from pyvqnet.nn import Module, Embedding, Linear, Softmax, Conv1D
from pyvqnet.nn import ReLu
from pyvqnet.dtype import *
from pyvqnet.tensor import QTensor



class DConvNet(Module):

  def __init__(self, vocab_size, emb_size, num_class, kernel_size, stride):
    super(DConvNet, self).__init__()

    self.we_mapping = Embedding(vocab_size, emb_size)
    self.se_mapping = Linear(vocab_size, emb_size,dtype=kfloat32)
    self.conv1d = Conv1D(1, 1, kernel_size, stride)

    self.fc = Linear(emb_size, num_class,dtype=kfloat32)

  def forward(self, we, se):



    # map one-hot vector to embedding vector
    we = self.we_mapping(we)


    # apply conv1d

    bs, _, emb_size = we.shape
    we = tensor.transpose(we, [0, 2, 1])

    we = tensor.reshape(we, [bs*emb_size, 1, -1])

    we = ReLu()(self.conv1d(we))


    we = tensor.reshape(we, [bs, emb_size, -1])
    we = tensor.mean(we, 2)



    se = QTensor(se,dtype=kfloat32)

    # map TF-IDF vector to embedding vector
    se = self.se_mapping(se)


    # combine embeding of two granularities
    e =   we + se

    # final classification layer
    o = self.fc(e)

    return o

class ConvNet(Module):

  def __init__(self, vocab_size, emb_size, num_class, kernel_size, stride):
    super(ConvNet, self).__init__()

    self.we_mapping = Embedding(vocab_size, emb_size)
    self.se_mapping = Linear(vocab_size, emb_size,dtype=kfloat32)
    self.conv1d = Conv1D(emb_size, emb_size, kernel_size, stride)

    self.fc = Linear(emb_size, num_class,dtype=kfloat32)

  def forward(self, we, se):



    # map one-hot vector to embedding vector
    we = self.we_mapping(we)



    # apply conv1d

    bs, _, emb_size = we.shape
    we = tensor.transpose(we, [0, 2, 1])

    we = ReLu()(self.conv1d(we))


    we = tensor.reshape(we, [bs, emb_size, -1])
    we = tensor.mean(we, 2)



    se = QTensor(se,dtype=kfloat32)



    # map TF-IDF vector to embedding vector
    se = self.se_mapping(se)


    # combine embeding of two granularities
    e =  we + se

    # final classification layer
    o = self.fc(e)

    return o
