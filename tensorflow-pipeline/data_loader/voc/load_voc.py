# -*- coding: utf-8 -*-

import os
import math
import queue

import cv2
import numpy as np

from config import *
from tools.voc import *
from ..load_data_example import LoadDataExample


class LoadVoc(LoadDataExample):
    def __init__(self, filepath, kargs):
        self.queue_size = kargs.get('queue_size', 5000)
        self.mini_batch = kargs.get('mini_batch', 256)
        self.strides    = kargs.get('strides', 16)
        self.filename   = filepath
        self.batch_size = 1

        self.load_data(filepath)
        self.n_batch = math.ceil(self.n_data/self.batch_size)
        self.q = queue.Queue(maxsize=self.queue_size)

        self.readover = False
        self.i_batch  = 0
        self.get_indexes()


    def load_data(self, filepath):
        self.data = []
        for jpgpath in os.listdir(os.path.join(filepath, 'JPEGImages')):
            xmlpath = jpgpath.split('.')[0]+'.xml'
            bboxes, w, h = get_bbox(os.path.join(filepath, 'Annotations', xmlpath))

            img = cv2.imread(os.path.join(filepath, 'JPEGImages', jpgpath))
            assert w == img.shape[1] and h == img.shape[0]

            img, bboxes, pixels_anchors, ious, \
            box_reg_tag, box_reg, box_cls_tag, box_cls, pos = self.preprocess(bboxes, img)
            self.data.append([img, bboxes, pixels_anchors, ious, \
                              box_reg_tag, box_reg, box_cls_tag, box_cls, pos])
        self.n_data = len(self.data)


    def __next__(self):
        if self.i_batch < self.n_batch:
            self.i_batch += 1
            if self.q.qsize() < self.batch_size:
                self.put2q()
            x, box_reg, box_cls, box_reg_tag, box_cls_tag, idx = [], [], [], [], [], []
            i = 0
            while i < self.batch_size and not self.q.empty():
                try:
                    next_x, \
                    next_box_reg, next_box_cls, \
                    next_box_reg_tag, next_box_cls_tag, \
                    next_idx = self.q.get(block=True, timeout=1)
                    next_x = next_x.astype(np.float32)
                    next_x[:,:,0] -= 103.939
                    next_x[:,:,1] -= 116.779
                    next_x[:,:,2] -= 123.68
                    next_box_cls =  next_box_cls.astype(np.float32)
                except:
                    break
                x.append(next_x)
                box_reg.append(next_box_reg)
                box_reg_tag.append(next_box_reg_tag)
                box_cls.append(next_box_cls)
                box_cls_tag.append(next_box_cls_tag)
                idx.append(next_idx)
                i += 1
            x = np.stack(x)
            box_reg = np.stack(box_reg)
            box_cls = np.stack(box_cls)
            box_reg_tag = np.stack(box_reg_tag)
            box_cls_tag = np.stack(box_cls_tag)
            return x, box_reg, box_reg_tag, box_cls, box_cls_tag, idx
        else:
            self.readover = False
            self.i_batch = 0
            self.get_indexes()
            raise StopIteration


    def put2q(self):
        while not self.q.full():
            if self.readover: return

            try:
                idx = self.current_idx.pop(0)
            except:
                self.readover = True
                return

            allinfo = self.data[idx]
            x, box_reg, box_cls, y_box_reg_tag, y_box_cls_tag = self.get_xy(allinfo)

            self.q.put((x, box_reg, box_cls, y_box_reg_tag, y_box_cls_tag, idx))


    def preprocess(self, bboxes, img):
        h, w, _ = img.shape
        minedge = h if h < w else w
        ratio = 600 / minedge
        new_h, new_w = int(h*ratio), int(w*ratio)
        img = cv2.resize(img, (new_w, new_h))

        pixels_anchors = get_pixels_anchors(new_w, new_h, strides=self.strides)
        bboxes = (np.array(bboxes)*ratio).astype(np.int32)

        ious = get_iou(pixels_anchors, bboxes)

#       import pdb; pdb.set_trace()
        box_reg_tag, box_reg, box_cls_tag, box_cls, pos = relate(pixels_anchors, bboxes, ious, new_w, new_h)
        return img, bboxes, pixels_anchors, ious, \
               box_reg_tag, box_reg, box_cls_tag, box_cls, pos


    def get_xy(self, allinfo):
        img, bboxes, pixels_anchors, ious,\
        box_reg_tag, box_reg, box_cls_tag, box_cls, pos = allinfo
        x = img
        # 关于label，主要随机选取正负样本
        # 那么主要改动box_reg_tag, box_cls_tag 
        ii, jj = np.where(pos)
        if len(ii) > self.mini_batch / 2:
            rd = np.arange(len(ii))
            np.random.shuffle(rd)
            ii = ii[rd[:int(self.mini_batch/2)]]
            jj = jj[rd[:int(self.mini_batch/2)]]
        y_pos = pos.copy()
        y_pos[:,:] = False
        y_pos[ii, jj] = True
        y_pos_row_sum = y_pos.sum(axis=1)

        y_box_reg_tag = np.zeros((pixels_anchors.shape[0], 4), dtype=np.float32)
        y_box_reg_tag[y_pos_row_sum != 0] = 1

        y_box_cls_tag = np.zeros(ious.shape[0], dtype=np.float32)
        neg = ious < 0.3
        neg_row_tag = (neg.sum(axis=1) == ious.shape[1])
        neg_row_tag[y_pos_row_sum != 0] = False
        n_neg = self.mini_batch - y_pos_row_sum.sum()
        if neg_row_tag.sum() > n_neg:
            rd, = np.where(neg_row_tag)
            np.random.shuffle(rd)
            rd = rd[:n_neg]
            neg_row_tag[:] = False
            neg_row_tag[rd] = True
        y_box_cls_tag[neg_row_tag] = 1
        y_box_cls_tag[y_pos_row_sum != 0] = 1
#       if y_box_cls_tag.sum() != self.mini_batch:
#           import pdb; pdb.set_trace()

        h, w, _ = img.shape
        W, H = math.ceil(w/self.strides), math.ceil(h/self.strides)
        y_box_cls_tag = y_box_cls_tag.reshape((H, W, -1))
        y_box_reg_tag = y_box_reg_tag.reshape((H, W, -1))

        return x, box_reg, box_cls, y_box_reg_tag, y_box_cls_tag
    
