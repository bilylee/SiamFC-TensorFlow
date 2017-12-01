#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Dataset Sampler"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Sampler(object):
  def __init__(self, data_source, shuffle=True):
    self.data_source = data_source
    self.shuffle = shuffle

  def __iter__(self):
    data_idxs = np.arange(len(self.data_source))
    if self.shuffle:
      np.random.shuffle(data_idxs)

    for idx in data_idxs:
      yield idx


if __name__ == '__main__':
  x = [1, 2, 3]
  sampler = Sampler(x, shuffle=True)
  p = 0
  for xx in sampler:
    print(x[xx])
    p += 1
    if p == 10: break
