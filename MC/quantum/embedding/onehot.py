# coding:utf-8

import numpy as np

ALL = None

def embedding(x, max_len):
  global ALL
  if ALL is None:
    ALL = np.diag(np.ones([max_len]))
  oe = ALL[x]
  return oe
