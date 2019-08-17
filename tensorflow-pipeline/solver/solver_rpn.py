# -*- coding: utf-8 -*-

import os
import argparse
import time

import numpy as np
import tensorflow as tf

from config import *
from model.rpn.model import Model
from data_loader import *
from .solver_example import SolverExample


class SolverRPN(SolverExample):
    def __init__(self, kargs):
        self.update_rule  = kargs.get('update_rule', None)
        self.optim_config = kargs.get('optim_config', {})
        self.lr_decay     = kargs.get('lr_decay', 1.0)
        self.decay_epoch  = kargs.get('decay_epoch', 5)
        self.clip_grad    = kargs.get("clip_grad", None)
        self.weight_decay = kargs.get('weight_decay', 0)

        self.num_epochs  = kargs.get('num_epochs', 100)
        self.max_epochs_no_best = kargs.get('max_epochs_no_best', 5)
        
        self.save_interval = kargs.get("save_interval", 1)
        self.save_mode     = kargs.get("save_mode", ["interval", "final", "best"])
        self.save_paths    = kargs.get('save_paths', None) 
        self.is_validate   = kargs.get('is_validate', False)
        
        self.result_path = kargs.get('result_path', None)

        self.verbose     = kargs.get('verbose', True)
        self.print_every = kargs.get('print_every', 10)


    def train(self, sess, model, dataset, saver):
        tr, va = dataset
        self.best_val_loss, self.curr_val_loss = np.inf, np.inf
        no_better_val_step = 0

        start = time.time()
        for epoch in range(1, self.num_epochs+1):
            epoch_start = time.time()
            n_batch = tr.n_batch

            for batch, inpt in enumerate(tr):
                fetchs = self.get_train_fetchs(model)
                fetchs.extend(self.extend_train_fetchs(model))
#               import pdb; pdb.set_trace()

                _, batch_loss, *extend_fetchs = self.step(sess, model, inpt, fetchs)
                
                if self.verbose and (batch) % self.print_every == 0:
                    logger.info('Epoch {0}/{1}, batch {2}/{3}, batch_loss {4}'.format\
                            (epoch, self.num_epochs, batch+1, n_batch, batch_loss))
                    self.logger_extend_train_info(extend_fetchs)

            epoch_end = time.time()
            logger.info('Epoch {0}/{1}, batch {2}/{3}, batch_loss {4}'.format\
                    (epoch, self.num_epochs, n_batch, n_batch, batch_loss))
            logger.info('duration for training Epoch {0}/{1}: {2}'.format(epoch, self.num_epochs, epoch_end - epoch_start))

            has_decay = False
            if epoch % self.decay_epoch == 0:
                self.optim_config['learning_rate'] *= self.lr_decay
                logger.info("learning rate: {0}".format(self.optim_config['learning_rate']))
                has_decay = True
            
            if self.is_validate:
                self.curr_val_loss = self.validate(sess, model, va)

            self.save(sess, saver, epoch)
            
            if self.is_validate:
                no_better_val_step += 1
                if self.curr_val_loss < self.best_val_loss:
                    self.best_val_loss = self.curr_val_loss
                    no_better_val_step = 0
                else:
                    if not has_decay:
                        pass
#                       self.optim_config['learning_rate'] *= self.lr_decay
#                       logger.info("learning rate: {0}".format(self.optim_config['learning_rate']))
                if no_better_val_step >= self.max_epochs_no_best:
                    logger.info('trained {0} epochs'.format(epoch))
                    break
        end = time.time()
        logger.info('total duration for training: {0}'.format(end-start))

    
    def get_train_fetchs(self, model):
        return [self.train_op, model.loss]


    def extend_train_fetchs(self, model):
        return [model.summary['reg_loss'], model.summary['cls_loss'],\
                model.global_norm_before_clip, model.global_norm_after_clip]
        

    def logger_extend_train_info(self, extend_fetchs):
        logger.info('reg_loss:{0}, cls_loss:{1}, before:{2}, after:{3}'\
                     .format(*extend_fetchs))


    def validate(self, sess, model, va):
        # can validate on va or tr 
        # to compute loss and acc
        pass


    def infer(self, sess, model, te):
        # just get results
        # not compute loss and acc
        # if want to compute acc, 
        # run evaluator, not solver
        pass


    def step(self, sess, model, inpt, fetchs):
        feed_dict = {model.input_placeholder: inpt[0],
                     model.box_reg:           inpt[1],
                     model.box_reg_tag:       inpt[2],
                     model.box_cls:           inpt[3],
                     model.box_cls_tag:       inpt[4],
                     self.lr: self.optim_config['learning_rate'],
                     self.wd: self.weight_decay}
        run_ret = sess.run(fetchs, feed_dict=feed_dict)
        return run_ret


    def save(self, sess, saver, epoch=0):
        if 'interval' in self.save_mode and epoch % self.save_interval == 0:
            save_path = saver.save(sess, self.save_paths[1], global_step=sess.graph.get_collection('global_step')[0])
            logger.info('model saved in file {0}'.format(save_path))
        if 'best' in self.save_mode and self.is_validate:
            if self.curr_val_loss < self.best_val_loss:
                save_path = saver.save(sess, self.save_paths[2])
                logger.info('model saved in file {0}'.format(save_path))


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


