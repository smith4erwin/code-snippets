# -*- coding: utf-8 -*-

import os
import math
import queue

import cv2
import numpy as np

from config import *


class LoadDataExample(object):
    def __init__(self, kargs):
        self.mode = kargs.get('mode', None)
        self.n_data = 0
        self.readover = False
        self.i_batch = 0
        self.get_indexes()
    

    def __iter__(self):
        return self


    def __next__(self):
        if self.i_batch < self.n_batch:
            self.i_batch += 1
        else:
            self.readover = False
            self.i_batch = 0
            self.get_indexes()
            raise StopIteration


    def get_indexes(self):
        self.current_idx = np.arange(self.n_data)
#       if self.filename in ['train']:
        if 'train' in self.filename:
            np.random.shuffle(self.current_idx)
        self.current_idx = self.current_idx.tolist()
