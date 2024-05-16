# coding=utf-8

import numpy as np
import pandas as pd
import pickle as pkl
import time
import argparse

from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CategoricalCrossEntropy as Loss
from pyvqnet.utils import metrics as vqnet_metrics
from pyvqnet.utils import storage

from quantum.const import *
from quantum.train import train_loop
from quantum.test import test_loop
from quantum.dataset import build_dataset
from quantum.tokenizer.ngram import get_tokenizer
from quantum.filter.stw import get_stw, stw_filter
from quantum.utils import stats
from quantum.embedding.onehot import embedding as onehot_embedding
from quantum.embedding.tf import embedding as tf_embedding
from quantum.circuit.q_encoder import encode

from classical.model import  DConvNet, ConvNet

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)

args = parser.parse_args()

dataset_name = 'MC'
train_path = './data/' + dataset_name + '/train.csv'
test_path = './data/' + dataset_name + '/test.csv'

# select model name (msff-dconv or msff-conv)
model_name = args.model_name

print('model name: ',model_name)

vocab_path = './data/' + dataset_name + '/vocab.classical.pkl'
idf_path = './data/' + dataset_name + '/idf.classical.pkl'
model_prefix = './model/' + dataset_name + '/' + model_name
log_prefix = './log/' + dataset_name + '/' +  model_name
batch_size = 8
epochs = 40
lr = 0.05
num_class = 2


seq_len = 6
kernel_size = 3
stride = 1



tokenizer = get_tokenizer(ngram=[1], token_filter=stw_filter, la='en')

vocab, idf, train_data, test_data = build_dataset(
  train_path, test_path, tokenizer, seq_len, need_pad=True
)

with open(vocab_path, 'wb') as f:
  pkl.dump(vocab, f)
with open(idf_path, 'wb') as f:
  pkl.dump(idf, f)

num_qubits = int(np.ceil(np.log2(len(vocab))))




x_test, y_test = np.asarray(test_data[0],
                            dtype=int), np.asarray(test_data[1], dtype=int)



def embedding(x, max_len, idf):
  # perfom word-level onehot embedding
  we = x

  # perfom sentence-level term vector embedding
  se = tf_embedding(x, max_len, idf)

  return we, se


x_test_we, x_test_se = embedding(x_test, len(vocab), idf)



if model_name == 'msff-conv':
    model = ConvNet(len(vocab), num_qubits, num_class, kernel_size,stride)
else:
    model = DConvNet(len(vocab), num_qubits, num_class, kernel_size,stride)


loss_func = Loss()
model_para = storage.load_parameters(f'{model_prefix}.best.model')
model.load_state_dict(model_para)


test_loss, test_acc = test_loop(
model,
loss_func, (x_test_we, x_test_se, y_test, batch_size),
metric_func= None
)

print(f'Test Acc is : {test_acc:.10f}')
