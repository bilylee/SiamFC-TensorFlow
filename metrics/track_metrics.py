#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.metrics_impl import _confusion_matrix_at_thresholds


def _auc(labels, predictions, weights=None, num_thresholds=200,
         metrics_collections=None, updates_collections=None,
         curve='ROC', name=None, summation_method='trapezoidal'):
  """Computes the approximate AUC via a Riemann sum.

  Modified version of tf.metrics.auc. Add support for AUC computation
  of the recall curve.
  """
  with tf.variable_scope(
      name, 'auc', (labels, predictions, weights)):
    if curve != 'ROC' and curve != 'PR' and curve != 'R':
      raise ValueError('curve must be either ROC, PR or R, %s unknown' %
                       (curve))
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _confusion_matrix_at_thresholds(
      labels, predictions, thresholds, weights)

    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6

    def compute_auc(tp, fn, tn, fp, name):
      """Computes the roc-auc or pr-auc based on confusion counts."""
      rec = tf.div(tp + epsilon, tp + fn + epsilon)
      if curve == 'ROC':
        fp_rate = tf.div(fp, fp + tn + epsilon)
        x = fp_rate
        y = rec
      elif curve == 'R':  # recall auc
        x = tf.linspace(1., 0., num_thresholds)
        y = rec
      else:  # curve == 'PR'.
        prec = tf.div(tp + epsilon, tp + fp + epsilon)
        x = rec
        y = prec
      if summation_method == 'trapezoidal':
        return tf.reduce_sum(
          tf.multiply(x[:num_thresholds - 1] - x[1:],
                      (y[:num_thresholds - 1] + y[1:]) / 2.),
          name=name)
      elif summation_method == 'minoring':
        return tf.reduce_sum(
          tf.multiply(x[:num_thresholds - 1] - x[1:],
                      tf.minimum(y[:num_thresholds - 1], y[1:])),
          name=name)
      elif summation_method == 'majoring':
        return tf.reduce_sum(
          tf.multiply(x[:num_thresholds - 1] - x[1:],
                      tf.maximum(y[:num_thresholds - 1], y[1:])),
          name=name)
      else:
        raise ValueError('Invalid summation_method: %s' % summation_method)

    # sum up the areas of all the trapeziums
    auc_value = compute_auc(
      values['tp'], values['fn'], values['tn'], values['fp'], 'value')
    update_op = compute_auc(
      update_ops['tp'], update_ops['fn'], update_ops['tn'], update_ops['fp'],
      'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, auc_value)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return auc_value, update_op


def get_center_index(response):
  """Get the index of the center in the response map"""
  shape = tf.shape(response)
  c1 = tf.to_int32((shape[1] - 1) / 2)
  c2 = tf.to_int32((shape[2] - 1) / 2)
  return c1, c2


def center_score_error(response):
  """Center score error.

  The error is low when the center of the response map is classified as target.
  """
  with tf.name_scope('CS-err'):
    r, c = get_center_index(response)
    center_score = response[:, r, c]
    mean, update_op = tf.metrics.mean(tf.to_float(center_score < 0))
    with tf.control_dependencies([update_op]):
      mean = tf.identity(mean)
    return mean


def get_maximum_index(response):
  """Get the index of the maximum value in the response map"""
  response_shape = response.get_shape().as_list()
  response_spatial_size = response_shape[-2:]  # e.g. [29, 29]
  length = response_spatial_size[0] * response_spatial_size[1]

  # Get maximum response index (note index starts from zero)
  ind_max = tf.argmax(tf.reshape(response, [-1, length]), 1)
  ind_row = tf.div(ind_max, response_spatial_size[1])
  ind_col = tf.mod(ind_max, response_spatial_size[1])
  return ind_row, ind_col


def center_dist_error(response):
  """Center distance error.

  The error is low when the maximum response is at the center of the response map.
  """
  with tf.name_scope('CD-err'):
    radius_in_pixel = 50.
    total_stride = 8.
    num_thresholds = 100
    radius_in_response = radius_in_pixel / total_stride

    gt_r, gt_c = get_center_index(response)
    max_r, max_c = get_maximum_index(response)
    gt_r = tf.to_float(gt_r)
    gt_c = tf.to_float(gt_c)
    max_r = tf.to_float(max_r)
    max_c = tf.to_float(max_c)
    distances = tf.sqrt((gt_r - max_r) ** 2 + (gt_c - max_c) ** 2)

    # We cast distances as prediction accuracies in the range [0, 1] where 0 means fail and
    # 1 means success. In this way, we can readily use streaming_auc to compute area
    # under curve.
    dist_norm = distances / radius_in_response
    dist_norm = tf.minimum(dist_norm, 1.)
    predictions = 1. - dist_norm
    labels = tf.ones_like(predictions)

    auc, update_op = _auc(labels, predictions, num_thresholds=num_thresholds, curve='R')
    with tf.control_dependencies([update_op]):
      err = 1. - auc
    return err
