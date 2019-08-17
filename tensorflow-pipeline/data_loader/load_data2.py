# -*- coding: utf-8 -*-

import os
import math
import copy
import queue
from operator import itemgetter

import cv2
import numpy as np

from config import *
from tools import MultiWork


class Input2(object):
    def __init__(self, filename, kargs):
        self.imgpath  = kargs.get('imgpath', None)
        # in general, batch_size << queue_size < cache_size < data_size
        self.batch_size = kargs.get('batch_size', 32)
        self.queue_size = kargs.get('queue_size', 1000)
        self.cache_size = kargs.get('cache_size', 4000)
        self.mode       = kargs.get('mode', None)
        self.labelfile  = kargs.get('labelfile', None)
        self.w          = kargs.get('w', 10)
        self.h          = kargs.get('h', 10)
        self.datatxt = filename

        with open(self.labelfile, 'r') as f:
            self.labels = [line.rstrip('\n') for line in f.readlines()]
            self.n_label = len(self.labels)
            self.labels = dict(zip(self.labels, range(self.n_label)))
        logger.info('number of label: {0}'.format(self.n_label))
        logger.info(self.labels)

        self.data = self.read_datatxt(filename)
        self.n_batch = math.ceil(len(self.data)/self.batch_size)
        self.q = queue.Queue(maxsize=self.queue_size)

        self.cache = {}

        self.readover = False
        self.i_batch = 0
        self.get_indexes()

    
    def read_datatxt(self, filename):
        with open(filename, 'r') as f:
            xys = [line.rstrip('\n').split(' ') for line in f.readlines()]
            return xys 


    def get_indexes(self):
        self.current_idx = np.arange(len(self.data))
        if self.mode in ['train']:
            np.random.shuffle(self.current_idx)
        self.current_idx = self.current_idx.tolist()


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
                i += 1
            if len(x) == 0: #当数据集有问题时，有效的datasize小于实际的datasize，最后无法读取数据
                self.readover = False
                self.i_batch = 0
                self.get_indexes()
                raise StopIteration
            x = np.stack(x)
            y = np.stack(y)

            return x, y, idx
        else:
            self.readover = False
            self.i_batch = 0
            self.get_indexes()
            raise StopIteration


    def put2q(self):
        if self.readover: return
        hole_size = self.q.maxsize - self.q.qsize()
#       try:
#           idxs = []
#           while hole_size > 0:
#               idxs.append(self.current_idx.pop(0))
#               hole_size -= 1
#       except:
#           self.readover = True

        idxs = self.current_idx[:hole_size]
        self.current_idx = self.current_idx[hole_size:]
        if len(self.current_idx) == 0:
            self.readover = True

        if len(idxs) == 0: return

#       raw_xys = itemgetter(*idxs)(self.data)
        raw_xys = [self.data[idx] for idx in idxs]
        raw_xs, raw_ys = list(zip(*raw_xys))
        xs = self.proc_batch_x(raw_xs)
        ys = self.proc_batch_y(raw_xys)
        
        for raw_x, raw_y, (x, xincache), (y, yincache), idx in zip(raw_xs, raw_ys, xs, ys, idxs):
            try:
                assert xincache == yincache
            except Exception as e:
                logger.error('cache error {0}'.format(self.data[idx]))
#               print(xincache, yincache, self.cache.keys(), raw_x in self.cache, raw_x)
                continue
            if x is None or y is None:
                logger.error('Error read {0}'.format(self.data[idx]))
                continue

            if not self.q.full():
                self.q.put((x,y,idx))

            if not xincache:
                self.cache_inout((raw_x, raw_y), (x, y))


    def cache_inout(self, raw_xy, xy):
        # not thread safe
        raw_x, raw_y = raw_xy
        if len(self.cache) == self.cache_size:
            rd_keys = list(self.cache.keys())[np.random.randint(0, len(self.cache))]
            del self.cache[rd_keys]
            self.cache[raw_x] = xy
        else:
            self.cache[raw_x] = xy


    def proc_batch_x(self, raw_xs):
        results = {}
        multiworker = MultiWork()
        results = multiworker(30, self.proc_x, raw_xs, results)
        proc_xs = [results[i] for i, _ in enumerate(raw_xs)]

#       results = {}
#       for i, raw_x in enumerate(raw_xs):
#           results = self.proc_x(raw_x, results, i)
#       proc_xs = [results[i] for i, _ in enumerate(raw_xs)]
#       for xm, x in zip(proc_xs_multi, proc_xs):
#           print(xm[0].shape, x[0].shape)
#           print(xm[1], x[1])
        return proc_xs


    def proc_batch_y(self, raw_xys):
        results = {}
        for i, (raw_x, raw_y) in enumerate(raw_xys):
            results = self.proc_y((raw_x, raw_y), results, i)
        proc_ys = [results[i] for i, _ in enumerate(raw_xys)]
        return proc_ys


    def proc_x(self, raw_x, results={}, idx=-1):
        try:
            if raw_x in self.cache:
                img, incache = self.cache[raw_x][0], True
            else:
                img = cv2.imread(os.path.join(self.imgpath, raw_x))
                if img.shape[0] < img.shape[1]:
                    img = np.transpose(img, axes=(1,0,2))
                img = cv2.resize(img, (self.w, self.h))
                img = img.astype(np.float32)
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
                incache = False

            results[idx] = (img, incache)
            return results
        except Exception as e:
            logger.error('proc_x error {0}'.format(raw_x))
            logger.error(e)
            results[idx] = (None, None)
            return results


    def proc_y(self, raw_xy, results={}, idx=-1):
        raw_x, raw_y = raw_xy 
        try:
            if raw_x in self.cache:
                label, incache = self.cache[raw_x][1], True
            else:
                label = np.zeros(self.n_label, dtype=np.float32)
                label[self.labels[raw_y]] = 1
                incache = False
    
            results[idx] = (label, incache)
            return results
        except Exception as e:
            logger.error('proc_y error {0}'.format(raw_x))
            logger.error(e)
            results[idx] = (None, None)
            return results
