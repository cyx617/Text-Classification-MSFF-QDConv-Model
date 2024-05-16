# coding=utf-8

from ..const import *

STW = None

def get_stw(fpth):
  global STW
  with open(fpth, 'r', encoding=FILE_ENCODING) as f:
    STW = set(map(lambda x: x.strip('\n'), f.readlines()))

def stw_filter(token):
  if STW is None:
    return True
  return len(set(token) & STW) == 0
