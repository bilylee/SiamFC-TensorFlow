#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Load pretrained color model in the SiamFC paper and save it in the TensorFlow format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from configuration import LOG_DIR
from scripts.convert_pretrained_model import ex

if __name__ == '__main__':
  RUN_NAME = 'SiamFC-3s-gray-pretrained'
  ex.run(
    config_updates={'model_config': {'embed_config': {'embedding_checkpoint_file': 'assets/2016-08-17_gray025.net.mat',
                                                      'train_embedding': False, },
                                     },
                    'train_config': {'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME), },
                    'track_config': {'log_dir': osp.join(LOG_DIR, 'track_model_inference', RUN_NAME), }
                    },
    options={'--name': RUN_NAME,
             '--force': True,
             '--enforce_clean': False,
             })
