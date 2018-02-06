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

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def im2rgb(im):
  if len(im.shape) != 3:
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


def get_crops(im, bbox, size_z, size_x, context_amount):
  """Obtain image sub-window, padding with avg channel if area goes outside of border

  Adapted from https://github.com/bertinetto/siamese-fc/blob/master/ILSVRC15-curation/save_crops.m#L46

  The mathematical formula is $s(w+2p) * s(h+2p) = A$, here you should be aware that
    s -> scale factor s is chosen such that the area of the scaled rectangle is equal to a constant 
    (w, h) -> tight bbox size
    p -> the amount of context to be half of the mean dimension p = (w+h)/4 in origin paper

  Args:
    im: Image ndarray
    bbox: Named tuple (x, y, width, height) x, y corresponds to the crops center
    size_z: Target + context size
    size_x: The resultant crop size
    context_amount: The amount of context

  Returns:
    image crop: Image ndarray
  """
  cy, cx, h, w = bbox.y, bbox.x, bbox.height, bbox.width
  wc_z = w + context_amount * (w + h)
  hc_z = h + context_amount * (w + h)
  s_z = np.sqrt(wc_z * hc_z)
  scale_z = size_z / s_z

  d_search = (size_x - size_z) / 2
  pad = d_search / scale_z
  s_x = s_z + 2 * pad
  scale_x = size_x / s_x

  image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx],
                                               [size_x, size_x],
                                               [np.round(s_x), np.round(s_x)])

  return image_crop_x, scale_x


def get_subwindow_avg(im, pos, model_sz, original_sz):
  # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
  avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
  if not original_sz:
    original_sz = model_sz
  sz = original_sz
  im_sz = im.shape
  # make sure the size is not too small
  assert im_sz[0] > 2 and im_sz[1] > 2
  c = [get_center(s) for s in sz]

  # check out-of-bounds coordinates, and set them to avg_chans
  context_xmin = np.int(np.round(pos[1] - c[1]))
  context_xmax = np.int(context_xmin + sz[1] - 1)
  context_ymin = np.int(np.round(pos[0] - c[0]))
  context_ymax = np.int(context_ymin + sz[0] - 1)
  left_pad = np.int(np.maximum(0, -context_xmin))
  top_pad = np.int(np.maximum(0, -context_ymin))
  right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
  bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

  context_xmin = context_xmin + left_pad
  context_xmax = context_xmax + left_pad
  context_ymin = context_ymin + top_pad
  context_ymax = context_ymax + top_pad
  if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
    R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[0]))
    G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[1]))
    B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)),
               'constant', constant_values=(avg_chans[2]))

    im = np.stack((R, G, B), axis=2)

  im_patch_original = im[context_ymin:context_ymax + 1,
                      context_xmin:context_xmax + 1, :]
  if not (model_sz[0] == original_sz[0] and model_sz[1] == original_sz[1]):
    im_patch = resize(im_patch_original, tuple(model_sz))
  else:
    im_patch = im_patch_original
  return im_patch, left_pad, top_pad, right_pad, bottom_pad
