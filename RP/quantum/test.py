# coding=utf-8

import numpy as np
from pyvqnet.tensor import tensor

from .dataset import dataloader

def test_loop(model, loss_func, data, metric_func=None):
  total_loss = []
  valid_true = []
  valid_pred = []

  model.eval()

  for x_we, x_se, y in dataloader(
    *data[:-1], batch_size=data[-1], shuffle=False
  ):
    # Forward pass
    output = model(x_we, x_se)
    # Calculating loss
    y_onehot = np.diag(np.ones([output.shape[1]]))[y]
    loss = loss_func(y_onehot.astype(np.int64), output)

    pred = output.to_numpy().argmax(axis=1).tolist()

    total_loss.append(loss.item())
    valid_true.extend(y)
    valid_pred.extend(pred)

  loss_mean = np.sum(total_loss)/len(total_loss)

  if metric_func is not None:
    y_true_Qtensor = tensor.QTensor(valid_true)
    y_pred_Qtensor = tensor.QTensor(valid_pred)
    metric = metric_func(y_true_Qtensor, y_pred_Qtensor)
  else:
    correct = sum(1 for i in range(len(valid_true)) if valid_true[i] == valid_pred[i])
    total = len(valid_true)
    accuracy = correct / total
    metric = accuracy

  return loss_mean, metric
