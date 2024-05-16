# coding=utf-8

import numpy as np
import pandas as pd

from .const import *

def build_vocab(file_path, tokenizer, max_size, min_freq, max_freq):
  vocab = {}
  df = {}
  idf = {}
  sys_id = {UNK: UNK_ID, PAD: PAD_ID}
  data = pd.read_csv(file_path, encoding=FILE_ENCODING)
  N = data.shape[0]
  for line in data['text']:
    lin = line.strip()
    if not lin:
      continue
    content = lin.split('\t')[0]
    tokens = tokenizer(content)
    existed = set()
    for token in tokens:
      vocab[token] = vocab.get(token, 0) + 1
      existed.add(token)
    for token in existed:
      df[token] = df.get(token, 0) + 1
  for token in vocab:
    # idf[token] = np.log((N + 1)/(df.get(token, N) + 1)) + 1
    idf[token] = 1
  vocab_list = sorted(
    [(k, v) for k, v in vocab.items() if v >= min_freq and v <= max_freq],
    key=lambda x: x[1]*idf[x[0]],
    reverse=True
  )[:max_size]
  vocab = {
    token: idx + len(sys_id) for idx, (token, _) in enumerate(vocab_list)
  }
  vocab.update(sys_id)
  idf.clear()
  for token in vocab:
    idf[vocab[token]] = np.log((N + 1)/(df.get(token, N) + 1)) + 1
  return vocab, idf

def load_dataset(file_path, vocab, tokenizer, seq_len=32, need_pad=True):
  token_ids = []
  labels = []
  data = pd.read_csv(file_path, encoding=FILE_ENCODING)
  for _, row in data.iterrows():
    text, label = row['text'], row['label']
    _token_ids = []
    tokens = tokenizer(text)
    if need_pad:
      if seq_len > 0:
        if len(tokens) < seq_len:
          tokens.extend([PAD]*(seq_len - len(tokens)))
        else:
          tokens = tokens[:seq_len]
    # token to id
    for token in tokens:
      _token_ids.append(vocab.get(token, vocab.get(UNK)))
    token_ids.append(_token_ids)
    labels.append(label)
  return token_ids, labels

def build_dataset(train_ds, test_ds, tokenizer, seq_len,MAX_VOCAB_SIZE=20000,need_pad=True):
  vocab, idf = build_vocab(
    train_ds,
    tokenizer=tokenizer,
    max_size=MAX_VOCAB_SIZE,
    min_freq=MIN_FREQ,
    max_freq=MAX_FREQ
  )
  print(f'Vocab size: {len(vocab)}')

  train = load_dataset(
    train_ds, vocab, tokenizer, seq_len=seq_len, need_pad=need_pad
  )
  test = load_dataset(
    test_ds, vocab, tokenizer, seq_len=seq_len, need_pad=need_pad
  )

  return vocab, idf, train, test

def dataloader(*data, batch_size=16, shuffle=False):
  total_sample = len(data[0])
  index = np.arange(total_sample)
  if shuffle:
    index = np.random.permutation(total_sample)
  for i in range(0, total_sample - batch_size + 1, batch_size):
    yield [data_i[index[i:i + batch_size]] for data_i in data]
  if total_sample%batch_size != 0:
    yield [data_i[index[-(total_sample%batch_size):]] for data_i in data]
