# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from config import *

class PretrainedVGG16(object):
    def __init__(self, path):
        vgg16 = np.load(path, encoding='latin1').item()
#       print(type(vgg16))
        self.initializer = {}
        for k, v in vgg16.items():
            self.initializer[k] = [tf.constant_initializer(v[0]),\
                                   tf.constant_initializer(v[1])]

        self.VGG_MEAN = [103.939, 116.779, 123.68] # BGR

