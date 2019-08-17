# -*- coding: utf-8 -*-

import tensorflow as tf

class BaseModel(object):
    def __init__(self):
        self.summary = {}


    def _conv_layer(self, name, in_, w_shape, w_init, b_init=None, relu=True, trainable=True):
        with tf.variable_scope(name):
            self.summary[name] = {}
            w = self._get_variable('w', w_shape, w_init, trainable)
            self.summary[name]['w'] = w
            out = tf.nn.conv2d(in_, w, strides=[1,1,1,1], padding='SAME')
            if b_init is not None:
                b = self._get_variable('b', w_shape[-1], b_init, trainable)
                self.summary[name]['b'] = b
                out = out + b
            if relu:
                out = tf.nn.relu(out)
            return out


    def _maxpool_layer(self, name, in_, **kargs):
        ksize = kargs.get('ksize', [1,2,2,1])
        strides = kargs.get('strides', [1,2,2,1])
        with tf.variable_scope(name):
            return tf.nn.max_pool(in_, ksize=ksize, strides=strides, padding='SAME')


    def _averagepool_layer(self, name, in_, **kargs):
        ksize = kargs.get('ksize', [1,2,2,1])
        strides = kargs.get('strides', [1,2,2,1])
        with tf.variable_scope(name):
            return tf.nn.avg_pool(in_, ksize=ksize, strides=strides, padding='SAME')


    def _fc(self, name, in_, w_shape, w_init, b_init, relu=True, trainable=True):
        with tf.variable_scope(name):
            w = self._get_variable('w', w_shape, w_init, trainable) 
            b = self._get_variable('b', w_shape[-1], b_init, trainable)
            self.summary[name] = {}
            self.summary[name]['w'] = w
            self.summary[name]['b'] = b
            out = tf.nn.xw_plus_b(in_, w, b)
            if relu:
                out = tf.nn.relu(out)
            return out


    def _dropout(self, name, in_, keep_prob):
        with tf.variable_scope(name):
            return tf.nn.dropout(in_, keep_prob)


    def _bn(self, name, in_, is_train, momentum):
#       with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name):
#           import pdb; pdb.set_trace()
#           params_shape = tf.shape(in_)[-1:]
            params_shape = [1,1,1]+in_.shape.as_list()[-1:]
            moving_mean = tf.get_variable('moving_mean', params_shape, dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
            moving_var  = tf.get_variable('moving_var',  params_shape, dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)

            def update_mean_var():
#               moving_mean = tf.get_variable('moving_mean', params_shape, dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
#               moving_var  = tf.get_variable('moving_var',  params_shape, dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
                from tensorflow.python.training.moving_averages import assign_moving_average

                batch_mean, batch_var = tf.nn.moments(in_, [0,1,2], name='moments')
#               moving_mean = tf.assign(moving_mean, momentum * moving_mean + (1 - momentum) * batch_mean)
#               moving_var  = tf.assign(moving_var,  momentum * moving_var  + (1 - momentum) * batch_var)
#               with tf.control_dependencies([moving_mean, moving_var]):
#                   return tf.identity(batch_mean), tf.identity(batch_var)
                with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, momentum, False),\
                                              assign_moving_average(moving_var,  batch_var,  momentum, False)]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
#                   return batch_mean, batch_var

#           batch_mena, batch_var = tf.cond(is_train, , la)
#           moving_mean = tf.assign()
#           moving_var  = tf.assign()
#           mean = tf.cond(is_train, lambda:(), lambda:(moving_mean, moving_var))

            mean, var = tf.cond(is_train, update_mean_var, lambda:(moving_mean, moving_var))
#           print(mean, var)
            gamma = tf.get_variable('gamma', shape=params_shape, dtype=tf.float32, initializer=tf.ones_initializer)
            beta  = tf.get_variable('beta',  shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
            out   = tf.nn.batch_normalization(in_, mean, var, beta, gamma, 1e-5)
            return out


    def _get_variable(self, name, shape, initializer, trainable):
        v = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable)
        return v


