# -*- coding: utf-8 -*-

import os
import json
import unittest
import requests

from config import * 
from tools import *

class ImageUtilsTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')


    def tearDown(self):
        print('tearDown...')


    def test_app(self):
#       import pdb; pdb.set_trace()
        with open('../resource/dataset/quanquan/screen_shot/screen_shot_urls.txt', 'r') as f:
            urls = [line.rstrip('\n') for line in f.readlines()]
        
        multidownloader = MultiDownload()
        multidownloader.download_multithread(100, urls, '../resource/dataset/quanquan/screen_shot')



        pass
