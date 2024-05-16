# coding=utf-8

import numpy as np

from .dataset import dataloader

from .test import test_loop

def train_loop(model, optimizer, loss_func, data, metric_func=None):
  
  total_loss = []

  model.train()

  for x_we, x_se, y in dataloader(
    *data[:-1], batch_size=data[-1], shuffle=True
  ):
    optimizer.zero_grad()
    # Forward pass
    output = model(x_we, x_se)
    # Calculating loss
    y = np.diag(np.ones([output.shape[1]]))[y]
    loss = loss_func(y.astype(np.int64), output)
    # Backward pass
    loss.backward()
    # Optimize the weights
    optimizer._step()

    total_loss.append(loss.item())

  loss_mean = np.sum(total_loss)/len(total_loss)

  if metric_func is not None:
    _, metric = test_loop(model, loss_func, data, metric_func=metric_func)
  else:
    _, metric = test_loop(model, loss_func, data, metric_func=None)

  return loss_mean, metric
