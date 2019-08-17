# -*- coding: utf-8 -*-

import os
import math
import queue
import struct

import numpy as np

from .load_data_example import LoadDataExample


class LoadMnist(LoadDataExample):
    def __init__(self, filename, kargs):
        self.imgpath = kargs.get('imgpath', None)

        self.batch_size = kargs.get('batch_size', 32)
        self.queue_size = kargs.get('queue_size', 1000)
        self.cache_size = kargs.get('cache_size', 4000)
        self.mode       = kargs.get('mode', None)
        if filename == 'train':
            label_path = os.path.join(self.imgpath, 'train-labels.idx1-ubyte')
            image_path = os.path.join(self.imgpath, 'train-images.idx3-ubyte')
        else:
            label_path = os.path.join(self.imgpath, 't10k-labels.idx1-ubyte')
            image_path = os.path.join(self.imgpath, 't10k-images.idx3-ubyte')
        self.load_data(label_path, image_path)
        self.n_data = len(self.xs)

        self.n_batch = math.ceil(self.n_data/self.batch_size)
        self.q = queue.Queue(maxsize=self.queue_size)

        self.readover = False
        self.i_batch = 0
        self.get_indexes()


    def __iter__(self):
        return self
    
    
    def load_data(self, label_path, image_path):
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
            a = np.zeros((len(labels), 10), dtype=np.float32)
            a[range(len(labels)), labels] = 1
            labels = a

        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28, 1).astype(np.float32)

        self.xs, self.ys = images, labels


    def __next__(self):
        if self.i_batch < self.n_batch:
            self.i_batch += 1
            if self.q.qsize() < self.batch_size:
                self.put2q()
            x, y, idx = [], [], []
            i = 0
            while i < self.batch_size and not self.q.empty():
                try:
                    next_idx = self.q.get(block=True, timeout=1)
                except:
                    break
                idx.append(next_idx)
                i += 1
            return self.xs[idx], self.ys[idx], idx
        else:
            self.readover = False
            self.i_batch = 0
            self.get_indexes()
            raise StopIteration


    def put2q(self):
        if self.readover: return
        hole_size = self.q.maxsize - self.q.qsize()
        idxs = self.current_idx[:hole_size]
        self.current_idx = self.current_idx[hole_size:]
        if len(self.current_idx) == 0:
            self.readover = True

        for idx in idxs:
            self.q.put(idx)
