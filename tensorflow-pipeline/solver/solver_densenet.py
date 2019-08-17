# -*- coding: utf-8 -*-

import os
import argparse
import time

import numpy as np
import tensorflow as tf

from config import *
from model.densenet.model import Model
from data_loader import *
from .solver_example import SolverExample


class SolverDensenet(SolverExample):
    def __init__(self, kargs):
        super(SolverDensenet, self).__init__(kargs)

    
    def extend_train_fetchs(self, model):
#       return [model.global_norm_before_clip, model.global_norm_after_clip]
#       return [model.is_train]
        return [model.global_norm_before_clip, model.global_norm_after_clip, model.corr]
#       return [self.l2_loss, model.global_norm_before_clip, model.global_norm_after_clip, model.corr]
    

    def logger_extend_train_info(self, extend_fetchs):
        grads_norm_bef, grads_norm_aft, corr = extend_fetchs
        logger.info('acc: {2}, grads: {0}/{1}'\
                .format(grads_norm_bef, grads_norm_aft, corr.sum()))


    def step(self, sess, model, inpt, fetchs, is_train=True):
        feed_dict = {model.input_placeholder: inpt[0],
                     model.label_placeholder: inpt[1],
                     model.is_train: is_train,
                     model.dropout_placeholder: model.keep_prob  if is_train else 1.,
                     self.lr: self.optim_config['learning_rate'],
                     self.wd: self.weight_decay}
        run_ret = sess.run(fetchs, feed_dict=feed_dict)
        return run_ret


    def validate(self, sess, model, va):
        logger.info('start validate')
        validate_start = time.time()
        total_loss = 0.
        total_corr = 0
        batch = 1
        n_batch = va.n_batch
        for inpt in va:
            fetchs = [model.loss, model.corr]
            batch_loss, batch_corr = self.step(sess, model, inpt, fetchs, False)
            total_loss += (batch_loss*inpt[0].shape[0])
            total_corr += batch_corr.sum()
            logger.info('validation batch {0}/{1} loss {2}'.format(batch, n_batch, batch_loss))
            batch += 1
        val_loss = total_loss / va.n_data  #can't average by batch_loss
        val_acc  = total_corr / va.n_data
        logger.info('validation loss and acc: {0} and {1}'.format(val_loss, val_acc))
        return val_loss


    def infer(self, sess, model, te):
        pass


    def add_train_op(self, model):
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])
        if self.update_rule == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.update_rule == "sgd_momentum":
            momentum = self.optim_config.get('momentum', 0.9)
            use_nesterov = self.optim_config.get('use_nesterov', False)
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum, use_nesterov=use_nesterov)
        elif self.update_rule == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.update_rule == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            raise Exception("no optimizer you want")

#       self.l2_loss = self.get_weights_loss()
#       grads_and_vars = optimizer.compute_gradients(model.loss+self.l2_loss)
        grads_and_vars = optimizer.compute_gradients(model.loss)

        grads, varss = zip(*grads_and_vars)
        model.global_norm_before_clip = tf.global_norm(grads) # self.my_calculate_global_norm = tf.sqrt( tf.add_n( [tf.norm(grad, 2)**2 for grad in grads] ) )
        if self.clip_grad:
            tmp_grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_grad)
            grads = tmp_grads
        model.global_norm_after_clip  = tf.global_norm(grads)

        grads = self.weight_decay_op(grads, varss)
        grads_and_vars = zip(grads, varss)

        train_op = optimizer.apply_gradients(grads_and_vars, name="train_op")
                                                                            
        ## add some var to instance for debugging conveniently
        model.summary['grads'] = grads
        return train_op


    def get_weights_loss(self):
        current_g = tf.get_default_graph()
        l2_loss_weights = []
        trainable_variables = current_g.get_collection('trainable_variables')
        for w in trainable_variables:
#           if 'b:0' not in w.name:
            l2_loss_weights.append(w)
        l2_loss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in l2_loss_weights]))
        self.wd = tf.placeholder(dtype=tf.float32, shape=[])
        return l2_loss * self.wd


    def weight_decay_op(self, grads, weights):
        self.wd = tf.placeholder(dtype=tf.float32, shape=[])
        add_decay_grads = []
        for w, g in zip(weights, grads):
            g = g + self.wd * w
            add_decay_grads.append(g)
        return add_decay_grads


