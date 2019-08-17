# -*- coding: utf-8 -*-

import os
import json
import unittest
import requests

from config import * 
from data_loader import *

class LoadData2Test(unittest.TestCase):
    def setUp(self):
        print('setUp...')


    def tearDown(self):
        print('tearDown...')


    def test_app(self):
        input_param = {'imgpath': '/home/commonrec/llh/cbir/framework_v0.1/images_sql/',
                       'queue_size': 10,
                       'batch_size': 3,
                       'cache_size': 15,
                       'w' : 192,
                       'h': 320,
                       'labelfile': os.path.join(RESOURCE_PATH, 'dataset/cont_pool_picture/labels.txt'),
                       'trainfile': os.path.join(RESOURCE_PATH, 'dataset/cont_pool_picture/train.txt'),
                       'validfile': None,
                       'testfile': None,
                       'mode': 'train',

                       }
#       import pdb; pdb.set_trace() 
        tr = Input2(input_param['trainfile'], input_param)
        for i, (x, y, idx) in enumerate(tr):
            print(i, x.shape, y.shape)
            print(idx)
            print(y)
#       import pdb; pdb.set_trace() 
        print('#################')
        for x, y, idx in tr:
            print(x.shape, y.shape)
            print(idx)
            print(y)
