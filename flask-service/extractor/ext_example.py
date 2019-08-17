#!/home/commonrec/llh/software/condallh/bin/python
# coding: utf-8

import os
import math
import threading
import time
import yaml
import cv2
import numpy as np

from .utils import * 


class Ext1(object):
    def __init__(self, **kargs):
        pass


    def readcfg(self, cfg):
        with open(cfg, 'r') as f:
            self.args = yaml.load(f)
        
        # general property
        # assert some keyword in self.args
        # eg. ext_algo, vecdir, dbfile, dbpath, params
        
        # own property
        # assert some keyword in self.args['params']
        # eg. batch_size

        # init something
        # add n_dim, n_feature, vecf

        # check the number of images, extracted and not extracted 
        pass

    def writecfg(self, vecdir, cfgname, cfg):
        fullpath = os.path.join(vecdir, cfgname)
        with open(fullpath, 'w') as f:
            yaml.dump(cfg, f)


    def save(self, vecdir, vecname, vec):
        fullpath = os.path.join(vecdir, vecname)
        np.save(fullpath, vec)


    def load(self):
        assert self.args['vecf']
        return Vec(self.args['vecdir'], self.args['vecf'])


    def extract(self):
        # extract online
        pass


    def extract_offline(self):
        # extract offline
        pass


    def read_imgs(self):
        # read offline
        # must resize before preprocess 
        pass


    def preprocess(self):
        pass


    def resize(self):
        # just for online
        pass
