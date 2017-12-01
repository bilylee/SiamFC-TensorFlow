#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Support integration with OTB benchmark"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time

import tensorflow as tf

# Code root absolute path
CODE_ROOT = '/path/to/SiamFC-TensorFlow'

# Checkpoint for evaluation
CHECKPOINT = '/path/to/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import auto_select_gpu, load_cfgs
from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
tf.logging.set_verbosity(tf.logging.DEBUG)


def run_SiamFC(seq, rp, bSaveImage):
  checkpoint_path = CHECKPOINT
  logging.info('Evaluating {}...'.format(checkpoint_path))

  # Read configurations from json
  model_config, _, track_config = load_cfgs(checkpoint_path)

  track_config['log_level'] = 0  # Skip verbose logging for speed

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint_path)
  g.finalize()

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(graph=g, config=sess_config) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    tracker = Tracker(model, model_config, track_config)

    tic = time.clock()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)

    trajectory_py = tracker.track(sess, init_bb, frames)
    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.clock() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result
