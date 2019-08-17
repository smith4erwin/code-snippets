# -*- coding: utf-8 -*-

import os
import math
import copy
import queue
import pickle as pkl
from operator import itemgetter

import cv2
import numpy as np

from config import *
from tools import MultiWork
from .load_data_example import LoadDataExample


class LoadCifar(LoadDataExample):
    def __init__(self, filename, kargs):
        self.imgpath = kargs.get('imgpath', None)
        self.batch_size = kargs.get('batch_size', 32)
        self.queue_size = kargs.get('queue_size', 5000)
        self.mode       = kargs.get('mode', None)
        self.cifar      = kargs.get('cifar', 0)
        self.filename   = filename

        if filename == 'train':
            if self.cifar == 100:
                filepath = os.path.join(self.imgpath, 'train')
            elif self.cifar == 10:
                filepath = [os.path.join(self.imgpath, 'data_batch_'+str(i)) for i in range(1,6)]
        else:
            if self.cifar == 100:
                filepath = os.path.join(self.imgpath, 'test')
            elif self.cifar == 10:
                filepath = [os.path.join(self.imgpath, 'test_batch')]

        self.load_data(filepath)
        self.n_batch = math.ceil(self.n_data/self.batch_size)
        self.q = queue.Queue(maxsize=self.queue_size)

        self.readover = False
        self.i_batch = 0
        self.get_indexes()
            

    def load_data(self, filepath):
        if self.cifar == 100:
            with open(filepath, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                n_label = 100
    
            xs = data['data'].reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32)
            ys = data['fine_labels']
            xs[:,:,:,0] = (xs[:,:,:,0]-129.3)/68.2
            xs[:,:,:,1] = (xs[:,:,:,1]-124.1)/65.4
            xs[:,:,:,2] = (xs[:,:,:,2]-112.4)/70.4
            self.xs = xs
            self.n_data = self.xs.shape[0]
            self.ys = np.zeros((self.n_data, n_label), dtype=np.float32)
            self.ys[range(self.n_data), data['fine_labels']] = 1. #[:100]

        elif self.cifar == 10:
            raw_data = []
            for path in filepath:
                with open(path, 'rb') as f:
                    raw_data.append(pkl.load(f, encoding='latin1'))
            xs = np.concatenate([data['data'] for data in raw_data],axis=0).reshape((-1,3,32,32)).transpose(0,2,3,1).astype(np.float32)
            ys = np.concatenate([data['labels'] for data in raw_data],axis=0).flatten()
            xs[:,:,:,0]  = (xs[:,:,:,0]-125.3)/63.0
            xs[:,:,:,1]  = (xs[:,:,:,1]-123.0)/62.1
            xs[:,:,:,2]  = (xs[:,:,:,2]-113.9)/66.7
            self.xs = xs
            self.n_data = self.xs.shape[0]
            n_label = 10
            self.ys = np.zeros((self.n_data, n_label), dtype=np.float32)
            self.ys[range(self.n_data), ys] = 1.


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
                    next_idx = self.q.get(block=True, timeout=1)
                except:
                    break
                idx.append(next_idx)
                i += 1
            xs = self.preprocess(self.xs[idx])
            return xs, self.ys[idx], idx
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


    def preprocess(self, raw_xs):
        # data augmentation and preprocessing
        return raw_xs
        if self.filename != 'train':
            return raw_xs
        
        n, h, w, c = raw_xs.shape
        for i in range(n):
            if np.random.rand() < 0.5:
                raw_xs[i] = raw_xs[i][:,::-1,:]
                
            pad = np.zeros((h+8, w+8,c), dtype=np.float32)
            pad[4:-4, 4:-4, :] = raw_xs[i]
            y, x = np.random.rand(0, 8, size=2, dtype='int32')
            raw_xs[i] = pad[y:y+32, x:x+32]

        return raw_xs



