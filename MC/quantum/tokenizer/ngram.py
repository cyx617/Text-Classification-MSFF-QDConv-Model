# coding=utf-8

import re
from functools import partial
from typing import Callable, List

def get_tokenizer(
  ngram: List[int] = [2],
  token_filter: Callable[[str], bool] = lambda _: True,
  la: str = 'zh'
):

  return partial(
    globals()[f'_tokenize_{la}'], ngram=ngram, token_filter=token_filter
  )

def _tokenize_zh(
  sent: str,
  *,
  ngram: List[int] = [2],
  token_filter: Callable[[str], bool] = lambda _: True,
  **_
):
  tokens = []
  for n in ngram:
    _tokens = list(
      filter(
        token_filter,
        map(
          lambda pos: f'{"".join(sent[pos:pos + n])}', range(len(sent) - n + 1)
        )
      )
    )
    tokens.extend(_tokens)
  return tokens

def _tokenize_en(
  sent: str,
  *,
  ngram: List[int] = [2],
  token_filter: Callable[[str], bool] = lambda _: True,
  **_
):
  sent = re.split(
    r',+|;+|\.+|!+|\:+|\'+|\?+|\-+|\s+|[()]+|[\[\]]+|[{}]+|[<>]+', sent
  )
  sent = list(filter(lambda w: w != '', sent))
  tokens = []
  for n in ngram:
    _tokens = list(
      filter(
        token_filter,
        map(
          lambda pos: f'{" ".join(sent[pos:pos + n])}',
          range(len(sent) - n + 1)
        )
      )
    )
    tokens.extend(_tokens)
  return tokens
