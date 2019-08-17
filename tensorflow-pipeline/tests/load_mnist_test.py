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
        input_param = {'imgpath': os.path.join(RESOURCE_PATH, 'dataset/mnist'),
                       'queue_size': 1000,
                       'batch_size': 32,
                       'trainfile': 'train',
                       'validfile': 'valid',
                       'testfile': 'test',
                       'mode': 'test',
                       }
        import pdb; pdb.set_trace() 
        tr = LoadCifar(input_param)
        for i, (x, y, idx) in enumerate(tr):
            print(i, x.shape, y.shape)
            print(idx)
#           print(y)
#       import pdb; pdb.set_trace() 
        print('#################')
        for x, y, idx in tr:
            print(i, x.shape, y.shape)
            print(idx)
#           print(y)
