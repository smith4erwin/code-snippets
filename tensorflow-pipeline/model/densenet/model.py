# -*- coding: utf-8 -*-

import os
import math

import numpy as np
import tensorflow as tf

from config import *
from tools import BaseModel


class Model(BaseModel):

    def __init__(self, kargs):
        self.height  = kargs['height']
        self.width   = kargs['width']
        self.n_label = kargs['n_label']

        self.bn_momentum = kargs['bn_momentum']
        self.keep_prob   = kargs['keep_prob']

        self.n_filter_conv0 = kargs['n_filter_conv0']
        self.n_layers    = kargs['n_layers']
        self.growth_rate = kargs['growth_rate']
        self.bottleneck  = kargs['bottleneck']
        self.compression_rate = kargs['compression_rate']

        super(Model, self).__init__()
        self.build()


    def build(self):
        self.add_placeholder()
        self.pred = self.add_predition_op()
        self.loss = self.add_loss_op(self.pred)
        self.add_eval_batch_op(self.pred)
        self.train_op = None


    def add_placeholder(self):
        batch_size = None
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.height, self.width, 3], name='input')
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, self.n_label], name='label')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')


    def add_predition_op(self):
#       in_channel, out_channel = 3, self.growth_rate*2
        in_channel, out_channel = 3, self.n_filter_conv0 
        out = self.init_conv('conv0', self.input_placeholder, in_channel, out_channel)
        print(out)
        self.summary['conv0'] = out

        tr_channel = out_channel
        for i, n_layer in enumerate(self.n_layers):
            in_channel, n_layer = tr_channel, n_layer
            out, in_channel = self.dense_block('db'+str(i+1), out, in_channel, n_layer)
            print(out)
            self.summary['db'+str(i+1)] = out

            tr_channel = math.floor(self.compression_rate*in_channel)
            out = self.transition('tr'+str(i+1), out, in_channel, tr_channel, \
                                  last=(i==len(self.n_layers)-1))
            print(out)
            self.summary['tr'+str(i+1)] = out

#       in_channel, n_layer = out_channel, 12
#       out, in_channel = self.dense_block('db1', out, in_channel, n_layer)
#       print(out)
#       self.summary['db1'] = out
#       tr_channel = math.floor(self.compression_rate*in_channel)
#       out = self.transition('tr1', out, in_channel, tr_channel)
#       print(out)
#       self.summary['tr1'] = out

#       in_channel, n_layer = tr_channel, 12
#       out, in_channel = self.dense_block('db2', out, in_channel, n_layer)
#       print(out)
#       self.summary['db2'] = out
#       tr_channel = math.floor(self.compression_rate*in_channel)
#       out = self.transition('tr2', out, in_channel, tr_channel)
#       print(out)
#       self.summary['tr2'] = out

#       in_channel, n_layer = tr_channel, 12
#       out, in_channel = self.dense_block('db3', out, in_channel, n_layer)
#       print(out)
#       self.summary['db3'] = out
#       tr_channel = 0
#       out = self.transition('tr3', out, in_channel, tr_channel, last=True)
#       print(out)
#       self.summary['tr3'] = out
        
        w_shape = [tr_channel, self.n_label]
#       w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), tf.zeros_initializer()]
#       w_init, b_init = [tf.random_uniform_initializer(minval=-tf.sqrt(6/in_channel), maxval=tf.sqrt(6/in_channel)), tf.zeros_initializer()]
#       w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/in_channel)), tf.zeros_initializer()]
        w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/self.n_label)), tf.zeros_initializer()]
        out = self._fc('gb_fc', out, w_shape, w_init, b_init, relu=False)
        return out


    def add_loss_op(self, pred):
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_placeholder, logits=pred)
            self.summary['loss'] = loss
            loss = tf.reduce_mean(loss)
            return loss


    def init_conv(self, name, in_, in_channel, out_channel):
        w_shape = [3, 3, in_channel, out_channel]
#       w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), tf.zeros_initializer()]
#       w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), None]
#       w_init, b_init = [tf.random_uniform_initializer(minval=-tf.sqrt(6/(9*in_channel)), maxval=tf.sqrt(6/(9*in_channel))), None]
#       w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/(9*in_channel))), None]
        w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/(9*out_channel))), None]
        out = self._conv_layer(name, in_, w_shape, w_init, b_init, relu=False)
        return out


    def dense_block(self, name, in_, in_channel, n_layer):
        inpt = in_
        out_channel = in_channel
        for i in range(1, n_layer+1):
            bn  = self._bn('_'.join([name, str(i), 'bn']), inpt, self.is_train, self.bn_momentum)
            rel = tf.nn.relu(bn)
            if self.bottleneck:
                w_shape = [1, 1, out_channel, self.growth_rate*4]
#               w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), tf.zeros_initializer()]
#               w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), None]
#               w_init, b_init = [tf.random_uniform_initializer(minval=-tf.sqrt(6/out_channel), maxval=tf.sqrt(6/out_channel)), None]
#               w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/out_channel)), None]
                w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/(self.growth_rate*4))), None]
                B_conv = self._conv_layer('_'.join([name, str(i), 'B_conv']), rel, w_shape, w_init, b_init, relu=False)
                B_dp   = self._dropout('_'.join([name, str(i), 'B_dp']), B_conv, self.dropout_placeholder)
                B_bn   = self._bn('_'.join([name, str(i), 'B_bn']), B_dp, self.is_train, self.bn_momentum)
                B_rel  = tf.nn.relu(B_bn)
                rel    = B_rel

            ic = self.growth_rate*4 if self.bottleneck else out_channel
                
            w_shape = [3, 3, ic, self.growth_rate]
#           w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), tf.zeros_initializer()]
#           w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), None]
#           w_init, b_init = [tf.random_uniform_initializer(minval=-tf.sqrt(6/(9*ic)), maxval=tf.sqrt(6/(9*ic))), None]
#           w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/(9*ic))), None]
            w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/(9*self.growth_rate))), None]
            conv = self._conv_layer(name+'_'+str(i), rel, w_shape, w_init, b_init, relu=False)
            dp   = self._dropout(name+'_'+str(i), conv, self.dropout_placeholder)

            inpt = tf.concat([inpt, dp], axis=3)
            out_channel += self.growth_rate
        out = inpt
        return out, out_channel


    def transition(self, name, in_, in_channel, tr_channel, last=False):
        bn  = self._bn(name+'_bn', in_, self.is_train, self.bn_momentum)
        rel = tf.nn.relu(bn)
        if not last:
            w_shape = [1, 1, in_channel, tr_channel]
#           w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), tf.zeros_initializer()]
#           w_init, b_init = [tf.contrib.layers.variance_scaling_initializer(), None]
#           w_init, b_init = [tf.random_uniform_initializer(minval=-tf.sqrt(6/in_channel), maxval=tf.sqrt(6/in_channel)), None]
#           w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/in_channel)), None]
            w_init, b_init = [tf.random_normal_initializer(stddev=tf.sqrt(2/tr_channel)), None]
            conv = self._conv_layer(name+'_conv', rel, w_shape, w_init, b_init, relu=False)
            dp   = self._dropout(name+'_dp', conv, self.dropout_placeholder)
            out  = self._averagepool_layer(name+'_pool', dp)
        else:
            out = tf.reduce_mean(rel, axis=[1,2], name=name+'_gp')
#           out = tf.nn.avg_pool(rel, ksize=[1,8,8,1], strides=[1,1,1,1], padding='VALID', name=name+'_gp')
#           out = tf.reshape(out, shape=[-1, tr_channel])
        return out


    def add_eval_batch_op(self, pred):
        with tf.variable_scope('eval'):
            gt_label = tf.arg_max(self.label_placeholder, dimension=1)
            pd_label = tf.arg_max(pred, dimension=1)
            self.corr = tf.equal(gt_label, pd_label)
