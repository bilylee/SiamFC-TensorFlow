#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Convert the matlab-pretrained model into TensorFlow format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import sys

import numpy as np
import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

import configuration
import siamese_model
from utils.misc_utils import auto_select_gpu, save_cfgs

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
tf.logging.set_verbosity(tf.logging.DEBUG)

from sacred import Experiment

ex = Experiment(configuration.RUN_NAME)


@ex.config
def configurations():
  # Add configurations for current script, for more details please see the documentation of `sacred`.
  model_config = configuration.MODEL_CONFIG
  train_config = configuration.TRAIN_CONFIG
  track_config = configuration.TRACK_CONFIG


@ex.automain
def main(model_config, train_config, track_config):
  # Create training directory
  train_dir = train_config['train_dir']
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info('Creating training directory: %s', train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the Tensorflow graph
  g = tf.Graph()
  with g.as_default():
    # Set fixed seed
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    # Build the model
    model = siamese_model.SiameseModel(model_config, train_config, mode='inference')
    model.build()

    # Save configurations for future reference
    save_cfgs(train_dir, model_config, train_config, track_config)

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])

    if not model_path:
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      start_step = 0

      # Load pretrained embedding model if needed
      if model_config['embed_config']['embedding_checkpoint_file']:
        model.init_fn(sess)

    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(tf.local_variables_initializer())
      saver.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1

    checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=start_step)
