# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

from config import *
from tools import BaseModel

from .load_vgg import PretrainedVGG16

vgg16 = PretrainedVGG16(os.path.join(PRETRAINED_MODEL_PATH, 'vgg16_machrisaa.npy'))

class Model1(BaseModel):

    def __init__(self, kargs):
        self.height    = kargs['height']
        self.width     = kargs['width']
        self.n_label   = kargs['n_label']
        super(Model1, self).__init__()
        self.build()


    def build(self):
        self.add_placeholder()
        self.pred = self.add_predictions_op()
        self.loss = self.add_loss_op(self.pred)
        self.infer = self.add_inference_op(self.pred)
        self.add_eval_batch_op(self.pred)
        self.train_op = None


    def add_placeholder(self):
        batch_size = None
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.height, self.width, 3], name='input')
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_label], name='label')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')


    def add_predictions_op(self):
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
        conv3_1 = self._conv_layer('conv3_1', maxpool_2, w_shape, w_init, b_init, trainable=False)

        w_shape = [3,3,256,256]
        w_init, b_init = vgg16.initializer['conv3_2']
        conv3_2 = self._conv_layer('conv3_2', conv3_1, w_shape, w_init, b_init, trainable=False)

        w_shape= [3,3,256,256]
        w_init, b_init = vgg16.initializer['conv3_3']
        conv3_3 = self._conv_layer('conv3_3', conv3_2, w_shape, w_init, b_init, trainable=False)

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

        maxpool_5 = self._maxpool_layer('maxpool_5', conv5_3)  #[None, 10, 6, 512]

        fc_in = np.prod(maxpool_5.shape.as_list()[1:])
        flattened = tf.reshape(maxpool_5, [-1, fc_in])

        w_shape = [fc_in, 4096]
#       w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        w_init, b_init = vgg16.initializer['fc6']
        fc_6 = self._fc('fc_6', flattened, w_shape, w_init, b_init)
        dp_6 = self._dropout('dp_6', fc_6, self.dropout_placeholder)

        w_shape = [4096, 4096]
#       w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        w_init, b_init = vgg16.initializer['fc7']
        fc_7 = self._fc('fc_7', fc_6, w_shape, w_init, b_init)
        dp_7 = self._dropout('dp_7', fc_7, self.dropout_placeholder)

        w_shape = [4096, self.n_label]
        w_init, b_init = [tf.contrib.layers.xavier_initializer(seed=0)] * 2
        fc_8 = self._fc('fc_8', fc_7, w_shape, w_init, b_init, relu=False)

        
        self.summary['layer1'] = maxpool_1
        self.summary['layer2'] = maxpool_2
        self.summary['layer3'] = maxpool_3
        self.summary['layer4'] = maxpool_4
        self.summary['layer5'] = maxpool_5
        self.summary['layer6'] = fc_6
        self.summary['layer7'] = fc_7
        self.summary['layer8'] = fc_8

        return fc_8


    def add_inference_op(self, pred_score):
        return tf.nn.softmax(pred_score, name='infer')
#       if self.loss_type == 'ce':
#           infer = tf.sigmoid(pred_score, 'infer')
#       elif self.loss_type == 'hg':
#           infer = pred_score
#       elif self.loss_type == 'sf':
#           infer = tf.nn.softmax(pred_score, name='infer') 
#       return infer


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


#        if self.loss_type == 'ce':
#            with tf.variable_scope('loss'):
#                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_placeholder, logits=pred)
#                self.summary['loss'] = loss
#                loss = tf.reduce_mean(loss, name='loss')
#                return loss
#        elif self.loss_type == 'hg':
#            with tf.variable_scope('loss'):
##               y2_sub_1 = tf.subtract(tf.scalar_mul(2., self.label_placeholder), 1.)
##               myloss = tf.maximum(0., tf.subtract(1.,tf.multiply(y2_sub_1, pred)))
##               print(myloss.shape)
##               myloss_mean = tf.reduce_mean(myloss, name='myloss')
#
#                loss = tf.reduce_mean(tf.losses.hinge_loss(labels=self.label_placeholder, logits=pred))
#                self.summary['loss'] = loss
##               self.summary['loss'] = (myloss_mean, loss)
#                return loss
#        elif self.loss_type == 'sf':
#            with tf.variable_scope('loss'):
#                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_placeholder, logits=pred)
#                loss = tf.reduce_mean(loss)
#                self.summary['loss'] = loss
#                return loss


