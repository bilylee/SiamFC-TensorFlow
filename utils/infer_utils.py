#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""
Inference Utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
from cv2 import resize

from utils.misc_utils import get_center

"""Notice in Rectangle namedtuple object, x and y can be top-left or center"""
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def im2rgb(im):
  if len(im.shape) != 3 and len(im.shape) == 1:
    im = np.stack([im, im, im], -1)
  return im


def convert_bbox_format(bbox, to):
  x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
  if to == 'top-left-based':
    x -= get_center(target_width)
    y -= get_center(target_height)
  elif to == 'center-based':
    y += get_center(target_height)
    x += get_center(target_width)
  else:
    raise ValueError("Bbox format: {} was not recognized".format(to))
  return Rectangle(x, y, target_width, target_height)


def get_exemplar_images(images, exemplar_size, targets_pos=None):
  """Crop exemplar image from input images"""
  with tf.name_scope('get_exemplar_image'):
    batch_size, x_height, x_width = images.get_shape().as_list()[:3]
    z_height, z_width = exemplar_size

    if targets_pos is None:
      target_pos_single = [[get_center(x_height), get_center(x_width)]]
      targets_pos_ = tf.tile(target_pos_single, [batch_size, 1])
    else:
      targets_pos_ = targets_pos

    # convert to top-left corner based coordinates
    top = tf.to_int32(tf.round(targets_pos_[:, 0] - get_center(z_height)))
    bottom = tf.to_int32(top + z_height)
    left = tf.to_int32(tf.round(targets_pos_[:, 1] - get_center(z_width)))
    right = tf.to_int32(left + z_width)

    def _slice(x):
      f, t, l, b, r = x
      c = f[t:b, l:r]
      return c

    exemplar_img = tf.map_fn(_slice, (images, top, left, bottom, right), dtype=images.dtype)
    exemplar_img.set_shape([batch_size, z_height, z_width, 3])
    return exemplar_img

