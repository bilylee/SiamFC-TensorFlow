#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Class for tracking using a track model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path as osp

import numpy as np
import cv2
from cv2 import imwrite

from utils.infer_utils import convert_bbox_format, Rectangle
from utils.misc_utils import get_center, get


class TargetState(object):
  """Represent the target state."""

  def __init__(self, bbox, search_pos, scale_idx):
    self.bbox = bbox  # (cx, cy, w, h) in the original image
    self.search_pos = search_pos  # target center position in the search image
    self.scale_idx = scale_idx  # scale index in the searched scales


class Tracker(object):
  """Tracker based on the siamese model."""

  def __init__(self, siamese_model, model_config, track_config):
    self.siamese_model = siamese_model
    self.model_config = model_config
    self.track_config = track_config

    self.num_scales = track_config['num_scales']
    logging.info('track num scales -- {}'.format(self.num_scales))
    scales = np.arange(self.num_scales) - get_center(self.num_scales)
    self.search_factors = [self.track_config['scale_step'] ** x for x in scales]

    self.x_image_size = track_config['x_image_size']  # Search image size
    self.window = None  # Cosine window
    self.log_level = track_config['log_level']

  def track(self, sess, first_bbox, frames, logdir='/tmp'):
    """Runs tracking on a single image sequence."""
    # Get initial target bounding box and convert to center based
    bbox = convert_bbox_format(first_bbox, 'center-based')

    # Feed in the first frame image to set initial state.
    bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
    input_feed = [frames[0], bbox_feed]
    frame2crop_scale = self.siamese_model.initialize(sess, input_feed)

    # Storing target state
    original_target_height = bbox.height
    original_target_width = bbox.width
    search_center = np.array([get_center(self.x_image_size),
                              get_center(self.x_image_size)])
    current_target_state = TargetState(bbox=bbox,
                                       search_pos=search_center,
                                       scale_idx=int(get_center(self.num_scales)))

    include_first = get(self.track_config, 'include_first', False)
    logging.info('Tracking include first -- {}'.format(include_first))

    # Run tracking loop
    reported_bboxs = []
    for i, filename in enumerate(frames):
      if i > 0 or include_first:  # We don't really want to process the first image unless intended to do so.
        bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                     current_target_state.bbox.height, current_target_state.bbox.width]
        input_feed = [filename, bbox_feed]

        outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
        search_scale_list = outputs['scale_xs']
        response = outputs['response']
        response_size = response.shape[1]

        # Choose the scale whole response map has the highest peak
        if self.num_scales > 1:
          response_max = np.max(response, axis=(1, 2))
          penalties = self.track_config['scale_penalty'] * np.ones((self.num_scales))
          current_scale_idx = int(get_center(self.num_scales))
          penalties[current_scale_idx] = 1.0
          response_penalized = response_max * penalties
          best_scale = np.argmax(response_penalized)
        else:
          best_scale = 0

        response = response[best_scale]

        with np.errstate(all='raise'):  # Raise error if something goes wrong
          response = response - np.min(response)
          response = response / np.sum(response)

        if self.window is None:
          window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                          np.expand_dims(np.hanning(response_size), 0))
          self.window = window / np.sum(window)  # normalize window
        window_influence = self.track_config['window_influence']
        response = (1 - window_influence) * response + window_influence * self.window

        # Find maximum response
        r_max, c_max = np.unravel_index(response.argmax(),
                                        response.shape)

        # Convert from crop-relative coordinates to frame coordinates
        p_coor = np.array([r_max, c_max])
        # displacement from the center in instance final representation ...
        disp_instance_final = p_coor - get_center(response_size)
        # ... in instance feature space ...
        upsample_factor = self.track_config['upsample_factor']
        disp_instance_feat = disp_instance_final / upsample_factor
        # ... Avoid empty position ...
        r_radius = int(response_size / upsample_factor / 2)
        disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
        # ... in instance input ...
        disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
        # ... in instance original crop (in frame coordinates)
        disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
        # Position within frame in frame coordinates
        y = current_target_state.bbox.y
        x = current_target_state.bbox.x
        y += disp_instance_frame[0]
        x += disp_instance_frame[1]

        # Target scale damping and saturation
        target_scale = current_target_state.bbox.height / original_target_height
        search_factor = self.search_factors[best_scale]
        scale_damp = self.track_config['scale_damp']  # damping factor for scale update
        target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
        target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))

        # Some book keeping
        height = original_target_height * target_scale
        width = original_target_width * target_scale
        current_target_state.bbox = Rectangle(x, y, width, height)
        current_target_state.scale_idx = best_scale
        current_target_state.search_pos = search_center + disp_instance_input

        assert 0 <= current_target_state.search_pos[0] < self.x_image_size, \
          'target position in feature space should be no larger than input image size'
        assert 0 <= current_target_state.search_pos[1] < self.x_image_size, \
          'target position in feature space should be no larger than input image size'

        if self.log_level > 0:
          np.save(osp.join(logdir, 'num_frames.npy'), [i + 1])

          # Select the image with the highest score scale and convert it to uint8
          image_cropped = outputs['image_cropped'][best_scale].astype(np.uint8)
          # Note that imwrite in cv2 assumes the image is in BGR format.
          # However, the cropped image returned by TensorFlow is RGB.
          # Therefore, we convert color format using cv2.cvtColor
          imwrite(osp.join(logdir, 'image_cropped{}.jpg'.format(i)),
                  cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))

          np.save(osp.join(logdir, 'best_scale{}.npy'.format(i)), [best_scale])
          np.save(osp.join(logdir, 'response{}.npy'.format(i)), response)

          y_search, x_search = current_target_state.search_pos
          search_scale = search_scale_list[best_scale]
          target_height_search = height * search_scale
          target_width_search = width * search_scale
          bbox_search = Rectangle(x_search, y_search, target_width_search, target_height_search)
          bbox_search = convert_bbox_format(bbox_search, 'top-left-based')
          np.save(osp.join(logdir, 'bbox{}.npy'.format(i)),
                  [bbox_search.x, bbox_search.y, bbox_search.width, bbox_search.height])

      reported_bbox = convert_bbox_format(current_target_state.bbox, 'top-left-based')
      reported_bboxs.append(reported_bbox)
    return reported_bboxs
