# -*- coding: utf-8 -*-

import os

import yaml
import numpy as np

from config import *
import indexer
from .utils import *


class Tag2(object):
    def __init__(self, idxer=None):
        self.idxer = idxer

    def readcfg(self, cfg):
        with open('cfg', 'r') as f:
            self.args = yaml.load(f)
        logger.info('')
        logger.info('\n'+yaml.dump(self.args))

        # general property
        assert 'tag_algo' in self.args
        assert 'tagdir'   in self.args
        assert 'params'   in self.args

        # own property


        pass


    def writecfg(self):
        pass


    def get_tags(self, info_ids):
        res_tags = []
        for info_id in info_ids:
            res_tags.append(self.data['info_id']['tag'])
        return res_tags


    def tagged_offline(self):
        pass
        



