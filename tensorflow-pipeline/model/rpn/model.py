# -*- coding: utf-8 -*-

import os
import math

import numpy as np
import tensorflow as tf

from config import *
from tools import BaseModel

from ..load_vgg import PretrainedVGG16

vgg16 = PretrainedVGG16(os.path.join(PRETRAINED_MODEL_PATH, 'vgg16_machrisaa.npy'))

class Model(BaseModel):

    def __init__(self, kargs):
        self.lmbda = kargs.get('lmbda', 10)
        super(Model, self).__init__()
        self.build()


    def build(self):
        self.add_placeholder()
        self.pred = self.add_predition_op()
        self.loss = self.add_loss_op(self.pred)


    def add_placeholder(self):
        batch_size = None
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3], name='input')
        self.box_reg_tag       = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 36], name='box_reg_tag')
        self.box_reg           = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 36], name='box_reg')
        self.box_cls_tag       = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 9], name='box_cls_tag')
        self.box_cls           = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 9], name='box_cls')


    def add_predition_op(self):
        w_shape = [3,3,3,64]
        w_init, b_init = vgg16.initializer['conv1_1']
        conv1_1 = self._conv_layer('conv1_1', self.input_placeholder, w_shape, w_init, b_init, trainable=False)

        w_shape = [3,3,64,64]
        w_init, b_init = vgg16.initializer['conv1_2']
        conv1_2 = self._conv_layer('conv1_2', conv1_1, w_shape, w_init, b_init, trainable=False)

        maxpool_1 = self._maxpool_layer('maxpool_1', conv1_2)

        w_shape = [3,3,64,128]
        w_init, b_init = vgg16.initializer['conv2_1']
        conv2_1 = self._conv_layer('conv2_1', maxpool_1, w_shape, w_init, b_init, trainable=False) 

        w_shape = [3,3,128,128]
        w_init, b_ini = vgg16.initializer['conv2_2']
        conv2_2 = self._conv_layer('conv2_2', conv2_1, w_shape, w_init, b_init, trainable=False)

        maxpool_2 = self._maxpool_layer('maxpool_2', conv2_2)

        w_shape = [3,3,128,256]
        w_init, b_init = vgg16.initializer['conv3_1']
        conv3_1 = self._conv_layer('conv3_1', maxpool_2, w_shape, w_init, b_init, trainable=True)

        w_shape = [3,3,256,256]
        w_init, b_init = vgg16.initializer['conv3_2']
        conv3_2 = self._conv_layer('conv3_2', conv3_1, w_shape, w_init, b_init, trainable=True)

        w_shape= [3,3,256,256]
        w_init, b_init = vgg16.initializer['conv3_3']
        conv3_3 = self._conv_layer('conv3_3', conv3_2, w_shape, w_init, b_init, trainable=True)

        maxpool_3 = self._maxpool_layer('maxpool_3', conv3_3)

        w_shape = [3,3,256,512]
        w_init, b_init = vgg16.initializer['conv4_1']
        conv4_1 = self._conv_layer('conv4_1', maxpool_3, w_shape, w_init, b_init, trainable=True)

        w_shape = [3,3,512,512]
        w_init, b_init = vgg16.initializer['conv4_2']
        conv4_2 = self._conv_layer('conv4_2', conv4_1, w_shape, w_init, b_init, trainable=True)

        w_shape= [3,3,512,512]
        w_init, b_init = vgg16.initializer['conv4_3']
        conv4_3 = self._conv_layer('conv4_3', conv4_2, w_shape, w_init, b_init, trainable=True)

        maxpool_4 = self._maxpool_layer('maxpool_4', conv4_3)

        w_shape = [3,3,512,512]
        w_init, b_init = vgg16.initializer['conv5_1']
        conv5_1 = self._conv_layer('conv5_1', maxpool_4, w_shape, w_init, b_init, trainable=True)

        w_shape = [3,3,512,512]
        w_init, b_init = vgg16.initializer['conv5_2']
        conv5_2 = self._conv_layer('conv5_2', conv5_1, w_shape, w_init, b_init, trainable=True)

        w_shape= [3,3,512,512]
        w_init, b_init = vgg16.initializer['conv5_3']
        conv5_3 = self._conv_layer('conv5_3', conv5_2, w_shape, w_init, b_init, trainable=True)
        
        w_shape=[3,3,512,512]
        w_init, b_init = tf.random_normal_initializer(stddev=1e-2), tf.zeros_initializer()
        sliding = self._conv_layer('sliding', conv5_3, w_shape, w_init, b_init, trainable=True)

        w_shape = [1,1,512,36]
        w_init, b_init = tf.random_normal_initializer(stddev=1e-2), tf.zeros_initializer()
        reg = self._conv_layer('reg', sliding, w_shape, w_init, b_init, relu=False)

        w_shape = [1,1,512,9]
        w_init, b_init = tf.random_normal_initializer(stddev=1e-2), tf.zeros_initializer()
        cls = self._conv_layer('cls', sliding, w_shape, w_init, b_init, relu=False)

        self.summary['conv3_1'] = conv3_1
        self.summary['conv3_2'] = conv3_2
        self.summary['conv3_3'] = conv3_3
        self.summary['conv4_1'] = conv4_1
        self.summary['conv4_2'] = conv4_2
        self.summary['conv4_3'] = conv4_3
        self.summary['conv5_1'] = conv5_1
        self.summary['conv5_2'] = conv5_2
        self.summary['conv5_3'] = conv5_3
        self.summary['sliding'] = sliding 
        self.summary['reg']     = reg
        self.summary['cls']     = cls

        return reg, cls


    def add_loss_op(self, pred):
        box_reg_pred, box_cls_pred = pred
        with tf.variable_scope('loss'):
            reg_loss = tf.losses.huber_loss(labels=self.box_reg, predictions=box_reg_pred, reduction='none')
            reg_loss = reg_loss * self.box_reg_tag 
            reg_loss = tf.reduce_sum(reg_loss) / tf.reduce_sum(self.box_reg_tag) * 4

            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.box_cls, logits=box_cls_pred)
            cls_loss = cls_loss * self.box_cls_tag
            cls_loss = tf.reduce_sum(cls_loss) / tf.reduce_sum(self.box_cls_tag)

            loss = reg_loss * self.lmbda + cls_loss
            self.summary['reg_loss'] = reg_loss
            self.summary['cls_loss'] = cls_loss
            return loss



