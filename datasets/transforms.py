#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.


"""Various transforms for video and image augmentation"""

import numbers

import tensorflow as tf


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, example):
    for t in self.transforms:
      example = t(example)
    return example


class RandomGray(object):
  def __init__(self, gray_ratio=0.25):
    self.gray_ratio = gray_ratio

  def __call__(self, img_sequence):
    def rgb_to_gray():
      gray_images = tf.image.rgb_to_grayscale(img_sequence)
      return tf.concat([gray_images] * 3, axis=3)

    def identity():
      return tf.identity(img_sequence)

    return tf.cond(tf.less(tf.random_uniform([], 0, 1), self.gray_ratio), rgb_to_gray, identity)


class RandomStretch(object):
  def __init__(self, max_stretch=0.05, interpolation='bilinear'):
    self.max_stretch = max_stretch
    self.interpolation = interpolation

  def __call__(self, img):
    scale = 1.0 + tf.random_uniform([], -self.max_stretch, self.max_stretch)
    img_shape = tf.shape(img)
    ts = tf.to_int32(tf.round(tf.to_float(img_shape[:2]) * scale))
    resize_method_map = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                         'bicubic': tf.image.ResizeMethod.BICUBIC}
    return tf.image.resize_images(img, ts, method=resize_method_map[self.interpolation])


class CenterCrop(object):
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img):
    th, tw = self.size
    return tf.image.resize_image_with_crop_or_pad(img, th, tw)


class RandomCrop(object):
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img):
    img_shape = tf.shape(img)
    th, tw = self.size

    y1 = tf.random_uniform([], 0, img_shape[0] - th, dtype=tf.int32)
    x1 = tf.random_uniform([], 0, img_shape[1] - tw, dtype=tf.int32)

    return tf.image.crop_to_bounding_box(img, y1, x1, th, tw)
