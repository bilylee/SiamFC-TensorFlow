#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import xml.etree.ElementTree as ET
from glob import glob
from multiprocessing.pool import ThreadPool

import cv2
from cv2 import imread, imwrite

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format
from utils.misc_utils import mkdir_p


def get_track_save_directory(save_dir, split, subdir, video):
  subdir_map = {'ILSVRC2015_VID_train_0000': 'a',
                'ILSVRC2015_VID_train_0001': 'b',
                'ILSVRC2015_VID_train_0002': 'c',
                'ILSVRC2015_VID_train_0003': 'd',
                '': 'e'}
  return osp.join(save_dir, 'Data', 'VID', split, subdir_map[subdir], video)


def process_split(root_dir, save_dir, split, subdir='', ):
  data_dir = osp.join(root_dir, 'Data', 'VID', split)
  anno_dir = osp.join(root_dir, 'Annotations', 'VID', split, subdir)
  video_names = os.listdir(anno_dir)

  for idx, video in enumerate(video_names):
    print('{split}-{subdir} ({idx}/{total}): Processing {video}...'.format(split=split, subdir=subdir,
                                                                           idx=idx, total=len(video_names),
                                                                           video=video))
    video_path = osp.join(anno_dir, video)
    xml_files = glob(osp.join(video_path, '*.xml'))

    for xml in xml_files:
      tree = ET.parse(xml)
      root = tree.getroot()

      folder = root.find('folder').text
      filename = root.find('filename').text

      # Read image
      img_file = osp.join(data_dir, folder, filename + '.JPEG')
      img = None

      # Get all object bounding boxes
      bboxs = []
      for object in root.iter('object'):
        bbox = object.find('bndbox')
        xmax = float(bbox.find('xmax').text)
        xmin = float(bbox.find('xmin').text)
        ymax = float(bbox.find('ymax').text)
        ymin = float(bbox.find('ymin').text)
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        bboxs.append([xmin, ymin, width, height])

      for idx, object in enumerate(root.iter('object')):
        id = object.find('trackid').text
        class_name = object.find('name').text

        track_save_dir = get_track_save_directory(save_dir, 'train', subdir, video)
        mkdir_p(track_save_dir)
        savename = osp.join(track_save_dir, '{}.{:02d}.crop.x.jpg'.format(filename, int(id)))
        if osp.isfile(savename): continue  # skip existing images

        if img is None:
          img = imread(img_file)

        # Get crop
        target_box = convert_bbox_format(Rectangle(*bboxs[idx]), 'center-based')
        crop, _ = get_crops(img, target_box,
                            size_z=127, size_x=255,
                            context_amount=0.5, )

        imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == '__main__':
  vid_dir = osp.join(ROOT_DIR, 'data/ILSVRC2015')

  # Or, you could save the actual curated data to a disk with sufficient space
  # then create a soft link in `data/ILSVRC2015-VID-Curation`
  save_dir = 'data/ILSVRC2015-VID-Curation'

  pool = ThreadPool(processes=5)

  one_work = lambda a, b: process_split(vid_dir, save_dir, a, b)

  results = []
  results.append(pool.apply_async(one_work, ['val', '']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0000']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0001']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0002']))
  results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0003']))
  ans = [res.get() for res in results]
