# -*- coding: utf-8 -*-

import os
import json
import unittest
import requests

from config import * 
from data_loader import *

class LoadVocTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')


    def tearDown(self):
        print('tearDown...')


    def test_app(self):
        input_param = {'queue_size': 6000,
                       'trainfile': 'train',
                       'validfile': 'valid',
                       'testfile': os.path.join(RESOURCE_PATH, 'dataset/voc/voc_2007_test'),
                       }
        import pdb; pdb.set_trace() 
        tr = LoadVoc(input_param['testfile'], input_param)
        for i, (x, box_reg, box_reg_tag, box_cls, box_cls_tag, idx) in enumerate(tr):
            print(i, x.shape, box_reg.shape, box_reg_tag.shape, box_cls.shape, box_cls_tag.shape)
#           print(idx)
#           print(y)
#       import pdb; pdb.set_trace() 
#       print('#################')
#       for x, y, idx in tr:
#           print(i, x.shape, y.shape)
#           print(idx)
#           print(y)


