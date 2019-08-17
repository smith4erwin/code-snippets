# -*- coding: utf-8 -*-

import argparse
import unittest


from service import sql


class OfflineTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        parser = argparse.ArgumentParser(description='Extract feature and Build index')
        parser.add_argument('-f', action='store', dest='flag', type=int)
        parser.add_argument('-a', '--algo', action='store', dest='algo')
        parser.add_argument('-c', '--cfg',  action='store', dest='cfg')
        parser.add_argument('-m', action='store', dest='n_thread', type=int, default=5)
        
        args = parser.parse_args()
        print(args)


    def tearDown(self):
        print('tearDown...')
        pass


    def test_idx(self):
#       dwld = sql.Dwld()
#       dwld.run()
        pass





