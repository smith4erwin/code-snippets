# -*- coding: utf-8 -*-

import os
import math
import xml.etree.ElementTree as ET

import numpy as np

anchors = np.array([[ -45,  -90,   45,   90],
                    [ -64,  -64,   64,   64],
                    [ -90,  -45,   90,   45],
                    [ -90, -181,   90,  181],
                    [-128, -128,  128,  128],
                    [-181,  -90,  181,   90],
                    [-181, -362,  181,  362],
                    [-256, -256,  256,  256],
                    [-362, -181,  362,  181]], dtype=np.int32)

def get_bbox(xmlpath):
    f = open(xmlpath, 'r')
    tree = ET.parse(f)
    root = tree.getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    bboxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        xmlbox = obj.find('bndbox')
        bbox = int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)
        bboxes.append(bbox)
    f.close()
    return bboxes, w, h


def get_pixels_anchors(w, h, strides=16):
    y_range, x_range = np.arange(h)[::strides], np.arange(w)[::strides]
    pos = np.meshgrid(x_range, y_range)
    pos = np.stack(pos).transpose((1,2,0)).reshape((-1, 2))
    pixels_anchors = []
    for anchor in anchors:
        lu_x, lu_y = anchor[0] + pos[:,0], anchor[1] + pos[:,1]
        rd_x, rd_y = anchor[2] + pos[:,0], anchor[3] + pos[:,1]
        pixels_anchors.append(np.stack([lu_x, lu_y, rd_x, rd_y]))
    pixels_anchors = np.stack(pixels_anchors).transpose((2, 0, 1)).reshape((-1, 4))
    return pixels_anchors.astype(np.int32)


def get_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def convert_box(box):
    x1, y1, x2, y2 = box
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    w, h = x2-x1+1, y2-y1+1
    return cx, cy, w, h


def get_t(anchor, bbox):
    anchor = convert_box(anchor)
    bbox   = convert_box(bbox)
    tx = (bbox[0] - anchor[0]) / anchor[2]
    ty = (bbox[1] - anchor[1]) / anchor[3]
    tw = np.log(bbox[2]/anchor[2])
    th = np.log(bbox[3]/anchor[3])
    return [tx, ty, tw, th]


def relate(pixels_anchors, bboxes, ious, w, h, strides=16, high_thresh=0.7, low_thresh=0.3):
    pos = (ious >= high_thresh)    # 与某个bbox的iou大于阈值的anchor
    for i in range(ious.shape[1]):
        miou  = np.max(ious[:, i])
        idxes = np.where(ious[:, i] == miou)
        pos[idxes, i] = True      # 与某个bbox的iou最大的anchor
    cross_boundary = (pixels_anchors[:,0] < 0)  | \
                     (pixels_anchors[:,1] < 0)  | \
                     (pixels_anchors[:,2] >= w) | \
                     (pixels_anchors[:,3] >= h)
    pos[cross_boundary] = False   # 排除越过边界的anchor
    pos_row_sum = pos.sum(axis=1)
    invalid = (pos_row_sum > 1)
    row_max = np.argmax(ious, axis=1)
    pos[invalid] = False
    pos[invalid, row_max[invalid]] = True
    
    box_reg_tag = np.zeros((pixels_anchors.shape[0], 4), dtype=np.float32)
    box_reg_tag[pos_row_sum != 0] = 1
    
    box_reg = np.zeros((pixels_anchors.shape[0], 4), dtype=np.float32)
    ii, jj = np.where(pos)
    for i, j in zip(ii, jj):
        box_reg[i] = get_t(pixels_anchors[i], bboxes[j])
    
    box_cls_tag = np.array([0]*ious.shape[0], dtype=np.int8)
    neg = ious < low_thresh
    neg_row_sum = neg.sum(axis=1)
    box_cls_tag[neg_row_sum == ious.shape[1]] = 1    #与所有bbox的iou均小于阈值的anchor
    box_cls_tag[pos_row_sum != 0] = 1                #关联到了某个bbox的anchor
    
    box_cls = np.array([0]*ious.shape[0], dtype=np.int8)
    box_cls[pos_row_sum != 0] = 1
    
    W, H = math.ceil(w/strides), math.ceil(h/strides)
    box_cls_tag = box_cls_tag.reshape((H, W, -1))
    box_cls     = box_cls.reshape((H, W, -1))
    box_reg_tag = box_reg_tag.reshape((H, W, -1))
    box_reg     = box_reg.reshape((H, W, -1))
    return box_reg_tag, box_reg, box_cls_tag, box_cls, pos


