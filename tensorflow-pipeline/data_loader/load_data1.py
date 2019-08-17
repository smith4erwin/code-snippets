# -*- coding: utf-8 -*-

import os
import math
import queue

import cv2
import numpy as np

from config import *



class Input1(object):
    def __init__(self, filename, kargs):
        self.imgpath    = kargs.get('imgpath', None)
        self.queue_size = kargs.get('queue_size', 1000)
        self.batch_size = kargs.get('batch_size', 32)
        self.cache_size = kargs.get('cache_size', 10000)
        self.tagfile    = kargs.get('tagfile', None)
        self.mode       = kargs.get('mode', None)
        self.filename   = filename

        with open(self.tagfile, 'r') as f:
            tags = f.readlines()
            tags = [tag.strip() for tag in tags]
            self.tags_size = len(tags)
        self.tags = dict(zip(tags, range(self.tags_size)))
        self.tags_reverse = dict(zip(range(self.tags_size), tags))

        self.data = self.readfile(filename)
        self.n_batch = math.ceil(len(self.data)/self.batch_size)
        self.q = queue.Queue(maxsize=self.queue_size)

        self.cache = []
        self.cache_dict = {}
        self.cache_dict_reverse = {}

        self.readover = False 
        self.i_batch = 0
        self.get_indexes() 


    def readfile(self, filename):
        with open(filename, 'r') as f:
           lines = f.readlines()
           lines = [line.strip().split('\t') for line in lines]
        x = [os.path.join(self.imgpath, line[0]+'.jpg') for line in lines]
        y = [line[2].split(' ') for line in lines]
        return list(zip(x, y))


    def __iter__(self):
        return self


    def __next__(self):
        if self.i_batch < self.n_batch:
            self.i_batch += 1
            if self.q.qsize() < self.batch_size:
                self.put2q()

            x, y, idx = [], [], []
            i = 0
            while i < self.batch_size and not self.q.empty():
                try:
                    next_x, next_y, next_idx = self.q.get(block=True, timeout=1)
                except:
                    break
                x.append(next_x)
                y.append(next_y)
                idx.append(next_idx)
#               print(self.q.empty())
                i += 1
            x = np.stack(x)
            y = np.stack(y)
            return x, y, idx
        else:
            self.readover = False
            self.i_batch = 0
            self.get_indexes()
            raise StopIteration


    def put2q(self):
        while not self.q.full():
            if self.readover:
                return
            try:
                idx = self.current_idx.pop(0)
            except:
                self.readover = True
                return

            raw_x, raw_y = self.data[idx]
            if raw_x in self.cache_dict:
                x, y = self.cache[self.cache_dict[raw_x]]
            else:
                x, y = self.proc_x(raw_x), self.proc_y(raw_y)

                if len(self.cache) == self.cache_size:
                    rd = np.random.randint(0, len(self.cache))
                    rdx = self.cache_dict_reverse[rd]
                    del self.cache_dict[rdx]
                    self.cache_dict[raw_x] = rd
                    self.cache_dict_reverse[rd] = raw_x
                    self.cache[rd] = (x,y)
                else:
                    self.cache.append((x,y))
                    self.cache_dict[raw_x] = len(self.cache)-1
                    self.cache_dict_reverse[len(self.cache)-1] = raw_x

            if x is not None and y is not None:
                self.q.put((x,y, idx))
            else:
                logger.info('Error read {0}'.format(self.data[idx]))


    def get_indexes(self):
        self.current_idx = np.arange(len(self.data))
        if self.mode in ['train']:
            np.random.shuffle(self.current_idx)
        self.current_idx = self.current_idx.tolist()


    def proc_x(self, x):
        try:
            img = cv2.imread(x) 
            if img.shape[0] < img.shape[1]:
                img = np.transpose(img, axes=(1,0,2))
            img = cv2.resize(img, (192, 320)).astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            return img
        except:
            return None


    def proc_y(self, y):
        try:
            label = np.zeros(self.tags_size, dtype=np.float32)
            for yy in y:
                label[self.tags[yy]] = 1
            return label
        except:
            return None


