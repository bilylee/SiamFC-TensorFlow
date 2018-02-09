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


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  # Get one 255x255 search demo image
  search_image = tf.image.decode_jpeg(tf.read_file("000000.00.crop.x.jpg"),
                                      channels=3, dct_method="INTEGER_ACCURATE")
  search_image = tf.to_float(search_image)
  search_image = tf.expand_dims(search_image, 0)

  search_image_ph = tf.placeholder(tf.float32, shape=(1, 255, 255, 3))
  exemplar_image = get_exemplar_images(search_image_ph, [127, 127])

  with tf.Session() as sess:
    simage = sess.run(search_image)
    exemplar_image = sess.run(exemplar_image, feed_dict={search_image_ph: simage})
    simage, exemplar_image = tf.squeeze(simage).eval() * 255, tf.squeeze(exemplar_image).eval() * 255

    print (exemplar_image.shape)
    print (simage.shape)

    # Draw our exemplar image
    plt.figure(1)
    plt.imshow(exemplar_image)
    plt.figure(2)
    plt.imshow(simage)
    plt.show()
