# coding=utf-8

import pandas as pd
import collections

from ..const import *

def embedding(x, max_len, idf):
  bs, _ = x.shape
  columns = list(range(max_len))
  template = dict.fromkeys(columns, 0)
  tf = []
  for bid in range(bs):
    _template = template.copy()
    _template.update(dict(collections.Counter(x[bid, :])))
    _template.update({UNK_ID: 1, PAD_ID: 0})
    tf.append(_template)
  tf = pd.DataFrame(tf).reindex(columns=columns)
  tf = tf.to_numpy()
  tf = tf/tf.sum(axis=1, keepdims=True)
  idf = pd.DataFrame([idf]).reindex(columns=columns)
  idf = idf.to_numpy()
  tfidf = tf*idf
  tfidf = tfidf/(abs(tfidf).max(axis=1, keepdims=True) + 1e-5)
  tfidf = tfidf/((tfidf*tfidf).sum(axis=1, keepdims=True)**0.5 + 1e-5)
  return tfidf
