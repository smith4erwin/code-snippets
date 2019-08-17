# -*- coding: utf-8 -*-

import unittest

import sys
import os
print(os.getcwd())
print(sys.path)

import extractor

class AppTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        pass


    def tearDown(self):
        print('tearDown...')
        pass


    def test_p1(self):
        print('test [1')
