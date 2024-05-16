# coding=utf-8

import numpy as np

from .const import *

# Calculate the proportion of UNK tokens to the total number of tokens
def stats(data, exclude={PAD_ID}):
  unk_num = 0
  total_num = 0
  for sample in data:
    total_num += sum([token not in exclude for token in sample])
    unk_num += sum([token == UNK_ID for token in sample])
  print(f'UNK : {unk_num/total_num:.0%}')

# Count the model parameters
def how_many(model):
  num_params = 0
  params = model.state_dict()
  for k in params:
    w = params[k]
    num_params += np.prod(w.shape)
  print(f'Num Params: {num_params}')
