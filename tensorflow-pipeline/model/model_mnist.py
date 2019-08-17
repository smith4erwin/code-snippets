# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

from config import *
from tools import BaseModel

class ModelMnist(BaseModel):
    def __init__(self, kargs):
        super(ModelMnist, self).__init__()
        self.build()

    
    def build(self):
        self.add_placeholder()
        self.pred = self.add_predictions_op()
        self.loss = self.add_loss_op(self.pred)
        self.add_eval_batch_op(self.pred)
        self.train_op = None

    
    def add_placeholder(self):
        batch_size = None
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1], name='input')
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name='label')


    def add_predictions_op(self):
        w_shape = [3,3,1,8]
        w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        conv1_1 = self._conv_layer('conv1_1', self.input_placeholder, w_shape, w_init, b_init, trainable=True)
        
        maxpool_1 = self._maxpool_layer('maxpool_1', conv1_1) 
        
        w_shape = [3,3,8,8]
        w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        conv2_1 = self._conv_layer('conv2_1', maxpool_1, w_shape, w_init, b_init, trainable=True)

        maxpool_2 = self._maxpool_layer('maxpool_2', conv2_1) 

        fc_in = np.prod(maxpool_2.shape.as_list()[1:])
        flattend = tf.reshape(maxpool_2, [-1, fc_in])
        
        w_shape = [fc_in, 32]
        w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        fc_3 = self._fc('fc_3', flattend, w_shape, w_init, b_init)

        w_shape = [32, 10]
        w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        fc_4 = self._fc('fc_4', fc_3, w_shape, w_init, b_init, relu=False)
        
        return fc_4


    def add_loss_op(self, pred):
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_placeholder, logits=pred)
            self.summary['loss'] = loss
            loss = tf.reduce_mean(loss)
            return loss


    def add_eval_batch_op(self, pred):
        with tf.variable_scope('eval'):
            gt_label = tf.arg_max(self.label_placeholder, dimension=1)
            pd_label = tf.arg_max(pred, dimension=1)
            self.corr = tf.equal(gt_label, pd_label)
