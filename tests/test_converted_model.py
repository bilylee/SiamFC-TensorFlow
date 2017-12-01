#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Tests for track model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

import numpy as np
import scipy.io as sio
import tensorflow as tf
from scipy.misc import imread  # Only pillow 2.x is compatible with matlab 2016R

CURRENT_DIR = osp.dirname(__file__)
PARENT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(PARENT_DIR)

import siamese_model
import configuration
from utils.misc_utils import load_cfgs


def test_load_embedding_from_mat():
  """Test if the embedding model loaded from .mat
     produces the same features as the original MATLAB implementation"""
  matpath = osp.join(PARENT_DIR, 'assets/2016-08-17.net.mat')
  test_im = osp.join(CURRENT_DIR, '01.jpg')
  gt_feat = osp.join(CURRENT_DIR, 'result.mat')

  model_config = configuration.MODEL_CONFIG
  model_config['embed_config']['embedding_name'] = 'convolutional_alexnet'
  model_config['embed_config']['embedding_checkpoint_file'] = matpath  # For SiameseFC
  model_config['embed_config']['train_embedding'] = False

  g = tf.Graph()
  with g.as_default():
    model = siamese_model.SiameseModel(model_config, configuration.TRAIN_CONFIG, mode='inference')
    model.build()

    with tf.Session() as sess:
      # Initialize models
      init = tf.global_variables_initializer()
      sess.run(init)

      # Load model here
      model.init_fn(sess)

      # Load image
      im = imread(test_im)
      im_batch = np.expand_dims(im, 0)

      # Feed image
      feature = sess.run([model.exemplar_embeds], feed_dict={model.examplar_feed: im_batch})

      # Compare with features computed from original source code
      ideal_feature = sio.loadmat(gt_feat)['r']['z_features'][0][0]
      diff = feature - ideal_feature
      diff = np.sqrt(np.mean(np.square(diff)))
      print('Feature computation difference: {}'.format(diff))
      print('You should get something like: 0.00892720464617')


def test_load_embedding_from_converted_TF_model():
  """Test if the embedding model loaded from converted TensorFlow checkpoint
     produces the same features as the original implementation"""
  checkpoint = osp.join(PARENT_DIR, 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained')
  test_im = osp.join(CURRENT_DIR, '01.jpg')
  gt_feat = osp.join(CURRENT_DIR, 'result.mat')

  if not osp.exists(checkpoint):
    raise Exception('SiamFC-3s-color-pretrained is not generated yet.')
  model_config, train_config, track_config = load_cfgs(checkpoint)

  # Build the model
  g = tf.Graph()
  with g.as_default():
    model = siamese_model.SiameseModel(model_config, train_config, mode='inference')
    model.build()

    with tf.Session() as sess:
      # Load model here
      saver = tf.train.Saver(tf.global_variables())
      if osp.isdir(checkpoint):
        model_path = tf.train.latest_checkpoint(checkpoint)
      else:
        model_path = checkpoint

      saver.restore(sess, model_path)

      # Load image
      im = imread(test_im)
      im_batch = np.expand_dims(im, 0)

      # Feed image
      feature = sess.run([model.exemplar_embeds], feed_dict={model.examplar_feed: im_batch})

      # Compare with features computed from original source code
      ideal_feature = sio.loadmat(gt_feat)['r']['z_features'][0][0]
      diff = feature - ideal_feature
      diff = np.sqrt(np.mean(np.square(diff)))
      print('Feature computation difference: {}'.format(diff))
      print('You should get something like: 0.00892720464617')


def test():
  test_load_embedding_from_mat()
  test_load_embedding_from_converted_TF_model()


if __name__ == '__main__':
  test()
