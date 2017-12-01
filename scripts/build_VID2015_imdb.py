#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Save the paths of crops from the ImageNet VID 2015 dataset in pickle format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import pickle
import sys

import numpy as np
import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from utils.misc_utils import sort_nicely


class Config:
  ### Dataset
  # directory where curated dataset is stored
  dataset_dir = 'data/ILSVRC2015-VID-Curation'
  save_dir = 'data/'

  # percentage of all videos for validation
  validation_ratio = 0.1


class DataIter:
  """Container for dataset of one iteration"""
  pass


class Dataset:
  def __init__(self, config):
    self.config = config

  def _get_unique_trackids(self, video_dir):
    """Get unique trackids within video_dir"""
    x_image_paths = glob.glob(video_dir + '/*.crop.x.jpg')
    trackids = [os.path.basename(path).split('.')[1] for path in x_image_paths]
    unique_trackids = set(trackids)
    return unique_trackids

  def dataset_iterator(self, video_dirs):
    video_num = len(video_dirs)
    iter_size = 150
    iter_num = int(np.ceil(video_num / float(iter_size)))
    for iter_ in range(iter_num):
      iter_start = iter_ * iter_size
      iter_videos = video_dirs[iter_start: iter_start + iter_size]

      data_iter = DataIter()
      num_videos = len(iter_videos)
      instance_videos = []
      for index in range(num_videos):
        print('Processing {}/{}...'.format(iter_start + index, video_num))
        video_dir = iter_videos[index]
        trackids = self._get_unique_trackids(video_dir)

        for trackid in trackids:
          instance_image_paths = glob.glob(video_dir + '/*' + trackid + '.crop.x.jpg')

          # sort image paths by frame number
          instance_image_paths = sort_nicely(instance_image_paths)

          # get image absolute path
          instance_image_paths = [os.path.abspath(p) for p in instance_image_paths]
          instance_videos.append(instance_image_paths)
      data_iter.num_videos = len(instance_videos)
      data_iter.instance_videos = instance_videos
      yield data_iter

  def get_all_video_dirs(self):
    ann_dir = os.path.join(self.config.dataset_dir, 'Data', 'VID')
    all_video_dirs = []

    # We have already combined all training and validation videos in ILSVRC2015 and put them in the `train` directory.
    # The file structure is like:
    # train
    #    |- a
    #    |- b
    #    |_ c
    #       |- ILSVRC2015_train_00024001
    #       |- ILSVRC2015_train_00024002
    #       |_ ILSVRC2015_train_00024003
    #               |- 000045.00.crop.x.jpg
    #               |- 000046.00.crop.x.jpg
    #               |- ...
    train_dirs = os.listdir(os.path.join(ann_dir, 'train'))
    for dir_ in train_dirs:
      train_sub_dir = os.path.join(ann_dir, 'train', dir_)
      video_names = os.listdir(train_sub_dir)
      train_video_dirs = [os.path.join(train_sub_dir, name) for name in video_names]
      all_video_dirs = all_video_dirs + train_video_dirs

    return all_video_dirs


def main():
  # Get the data.
  config = Config()
  dataset = Dataset(config)
  all_video_dirs = dataset.get_all_video_dirs()
  num_validation = int(len(all_video_dirs) * config.validation_ratio)

  ### validation
  validation_dirs = all_video_dirs[:num_validation]
  validation_imdb = dict()
  validation_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(validation_dirs)):
    validation_imdb['videos'] += data_iter.instance_videos
  validation_imdb['n_videos'] = len(validation_imdb['videos'])
  validation_imdb['image_shape'] = (255, 255, 3)

  ### train
  train_dirs = all_video_dirs[num_validation:]
  train_imdb = dict()
  train_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(train_dirs)):
    train_imdb['videos'] += data_iter.instance_videos
  train_imdb['n_videos'] = len(train_imdb['videos'])
  train_imdb['image_shape'] = (255, 255, 3)

  if not tf.gfile.IsDirectory(config.save_dir):
    tf.logging.info('Creating training directory: %s', config.save_dir)
    tf.gfile.MakeDirs(config.save_dir)

  with open(os.path.join(config.save_dir, 'validation_imdb.pickle'), 'wb') as f:
    pickle.dump(validation_imdb, f)
  with open(os.path.join(config.save_dir, 'train_imdb.pickle'), 'wb') as f:
    pickle.dump(train_imdb, f)


if __name__ == '__main__':
  main()
