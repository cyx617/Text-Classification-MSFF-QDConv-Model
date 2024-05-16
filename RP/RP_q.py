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

from quantum.model import  QDConvNet, QConvNet


parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)

args = parser.parse_args()

dataset_name = 'RP'
train_path = './data/' + dataset_name + '/train.csv'
test_path = './data/' + dataset_name + '/test.csv'
# select model name (msff-qdconv or msff-qconv)
model_name = args.model_name


print('model name: ',model_name)

vocab_path = './data/' + dataset_name + '/vocab.quantum.pkl'
idf_path = './data/' + dataset_name + '/idf.quantum.pkl'
model_prefix = './model/' + dataset_name + '/' + model_name
log_prefix = './log/' + dataset_name + '/' +  model_name
batch_size = 6
epochs = 100
lr = 0.05
num_class = 2


seq_len = 5
emb_rep = 5
kernel_size = 3
depth = 2
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

print(f'Num of Qubits: {num_qubits}')

x_train, y_train = np.asarray(train_data[0],
                              dtype=int), np.asarray(train_data[1], dtype=int)
x_test, y_test = np.asarray(test_data[0],
                            dtype=int), np.asarray(test_data[1], dtype=int)



def embedding(x, max_len, idf):
  # perfom word-level onehot embedding
  we = onehot_embedding(x, max_len)
  # perfom sentence-level term vector embedding
  se = tf_embedding(x, max_len, idf)


  return we, se

x_train_we, x_train_se = embedding(x_train, len(vocab), idf)
x_test_we, x_test_se = embedding(x_test, len(vocab), idf)


if model_name == 'msff-qdconv':

    model = QDConvNet(len(vocab), num_qubits, num_class, kernel_size, depth, stride,emb_rep)
else:
    model = QConvNet(len(vocab), num_qubits, num_class, kernel_size, depth, stride,emb_rep)

def how_many(model):
  num_params = 0
  params = model.state_dict()
  #print(params)

  for k in params:
    w = params[k]
    num_params += np.prod(w.shape)
  print(f'Num Params: {num_params}')

how_many(model)
optimizer = Adam(model.parameters(), lr=lr)
loss_func = Loss()


best_for_now = -1e5
for epoch in range(1, epochs + 1):
  start = time.time()

  train_loss, train_acc = train_loop(
    model,
    optimizer,
    loss_func, (x_train_we, x_train_se, y_train, batch_size),
    metric_func= None
  )
  test_loss, test_acc = test_loop(
    model,
    loss_func, (x_test_we, x_test_se, y_test, batch_size),
    metric_func= None
  )

  end = time.time()

  print(
    f'[Epoch : {epoch}] Train Loss is : {train_loss:.10f}, Train Acc is : {train_acc:.10f}, Test Acc is : {test_acc:.10f}, Elapse Time: {end-start:.4f} s'
  )


  if test_acc > best_for_now:
    best_for_now = test_acc
    storage.save_parameters(model.state_dict(), f'{model_prefix}.best.model')


print('best test acc: ', best_for_now)
