#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

import os.path as osp
import sys
import zipfile

import six.moves.urllib as urllib

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.misc_utils import mkdir_p


def download_or_skip(download_url, save_path):
  if not osp.exists(save_path):
    print('Downloading: {}'.format(download_url))
    opener = urllib.request.URLopener()
    opener.retrieve(download_url, save_path)
  else:
    print('File {} exists, skip downloading.'.format(save_path))


if __name__ == '__main__':
  assets_dir = osp.join(ROOT_DIR, 'assets')

  # Make assets directory
  mkdir_p(assets_dir)

  # Download the pretrained color model
  download_base = 'https://www.robots.ox.ac.uk/~luca/stuff/siam-fc_nets/'
  model_name = '2016-08-17.net.mat'
  download_or_skip(download_base + model_name, osp.join(assets_dir, model_name))

  # Download the pretrained gray model
  download_base = 'https://www.robots.ox.ac.uk/~luca/stuff/siam-fc_nets/'
  model_name = '2016-08-17_gray025.net.mat'
  download_or_skip(download_base + model_name, osp.join(assets_dir, model_name))

  # Download one test sequence
  download_base = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq_new/"
  seq_name = 'KiteSurf.zip'
  download_or_skip(download_base + seq_name, osp.join(assets_dir, seq_name))

  # Unzip the test sequence
  with zipfile.ZipFile(osp.join(assets_dir, seq_name), 'r') as zip_ref:
    zip_ref.extractall(assets_dir)
