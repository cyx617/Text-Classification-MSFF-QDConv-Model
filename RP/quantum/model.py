# coding=utf-8

from pyvqnet import tensor
from pyvqnet.nn import Module, Softmax

from .qlayer.q_linear import QLinear
from .qlayer.q_multilinear import QMultiLinear
from .qlayer.q_conv import QDConv1D,QConv1D
from .qlayer.q_embedding import QEmbedding

# MSFF-QDConv model
class QDConvNet(Module):

  def __init__(
    self,
    vocab_size,
    emb_size,
    num_class,
    kernel_size,
    depth,
    stride,
    emb_rep=1,
    enttype=0
  ):
    super(QDConvNet, self).__init__()
    self.emb_size = emb_size
    self.vocab_size = vocab_size
    self.we_mapping = QEmbedding(emb_rep, emb_size, enttype=enttype)
    self.se_mapping = QEmbedding(emb_rep, emb_size, enttype=enttype)
    self.conv1d_1 = QDConv1D(None, None, kernel_size, depth, stride)
    self.conv1d_2 = QDConv1D(None, None, kernel_size, depth, stride)

    self.fc = QLinear(emb_size, num_class)

  def forward(self, we, se):
    # map one-hot vector to embedding vector
    bs, seq_len = we.shape[:2]
    we = tensor.reshape(we, [bs*seq_len, -1])
    we = self.we_mapping(we)
    we = tensor.reshape(we, [bs, seq_len, self.emb_size])

    # apply conv1d
    we = tensor.transpose(we, [0, 2, 1])
    we = self.conv1d_1(we)
    we = self.conv1d_2(we)
    we = tensor.mean(we, 2)

    # map TF-IDF vector to embedding vector
    se = self.se_mapping(se)

    # combine embeding of two granularities
    e =  we + se

    # final classification layer
    o = self.fc(e)

    return o


# MSFF-QConv model
class QConvNet(Module):

  def __init__(
    self,
    vocab_size,
    emb_size,
    num_class,
    kernel_size,
    depth,
    stride,
    emb_rep=1,
    enttype=0
  ):
    super(QConvNet, self).__init__()
    self.emb_size = emb_size
    self.vocab_size = vocab_size
    self.we_mapping = QEmbedding(emb_rep, emb_size, enttype=enttype)
    self.se_mapping = QEmbedding(emb_rep, emb_size, enttype=enttype)

    self.conv1d_1 = QConv1D(emb_size, emb_size, kernel_size, depth, stride)
    self.conv1d_2 = QConv1D(emb_size, emb_size, kernel_size, depth, stride)
    self.fc = QLinear(emb_size, num_class)

  def forward(self, we, se):
    # map one-hot vector to embedding vector
    bs, seq_len = we.shape[:2]
    we = tensor.reshape(we, [bs*seq_len, -1])
    we = self.we_mapping(we)
    we = tensor.reshape(we, [bs, seq_len, self.emb_size])

    # apply conv1d
    we = tensor.transpose(we, [0, 2, 1])
    we = self.conv1d_1(we)
    we = self.conv1d_2(we)
    we = tensor.mean(we, 2)

    # map TF-IDF vector to embedding vector
    se = self.se_mapping(se)

    # combine embeding of two granularities
    e = we + se

    # final classification layer
    o = self.fc(e)

    return o
