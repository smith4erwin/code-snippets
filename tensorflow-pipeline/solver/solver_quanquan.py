# -*- coding: utf-8 -*-

import os
import time
import math
import argparse

import numpy as np
import tensorflow as tf

from config import *
from model import Model1
from data_loader import Input2


class SolverQuanQuan(object):

    def __init__(self, kargs):
        self.update_rule  = kargs.get('update_rule', None)
        self.optim_config = kargs.get('optim_config', {})
        self.lr_decay     = kargs.get('lr_decay', 1.0)
        self.decay_epoch  = kargs.get('decay_epoch', 5)
        self.clip_grad    = kargs.get("clip_grad", None)
        
        self.num_epochs  = kargs.get('num_epochs', 100)
        self.batch_size  = kargs.get('batch_size', 32)
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

        self.best_val_loss = np.inf
        self.curr_val_loss = np.inf
        no_better_val_step = 0

        start = time.time()
        for epoch in range(1, self.num_epochs+1):
            epoch_start = time.time()
            batch = 1
            n_batch = math.ceil(len(tr.data) / self.batch_size)

            for inpt in tr:
                fetchs = [model.train_op, model.loss]
                fetchs.extend([model.global_norm_before_clip, model.global_norm_after_clip])
                add_feed_dict = {model.dropout_placeholder: 0.5}

                _, batch_loss, \
                grads_norm_bef, grads_norm_aft \
                        = self.step(sess, model, inpt, fetchs)
                
                if self.verbose and (batch) % self.print_every == 0:
                    logger.info('Epoch {0}/{1}, batch {2}/{3}, batch_loss {4}'.format(epoch, self.num_epochs, batch, n_batch, batch_loss))
                    logger.info('grads: {0}/{1}'.format(grads_norm_bef, grads_norm_aft))
                batch += 1

            epoch_end = time.time()
            logger.info('Epoch {0}/{1}, batch {2}/{3}, batch_loss {4}'.format(epoch, self.num_epochs, batch-1, n_batch, batch_loss))
            logger.info('duration for training Epoch {0}/{1}: {2}'.format(epoch, self.num_epochs, epoch_end - epoch_start))

            b = False
            if epoch % self.decay_epoch == 0:
                self.optim_config['learning_rate'] *= self.lr_decay
                logger.info("learning rate: {0}".format(self.optim_config['learning_rate']))
                b = True
            
            if self.is_validate:
                self.curr_val_loss = self.validate(sess, model, va)

            self.save(sess, saver, epoch)
            
            if self.is_validate:
                no_better_val_step += 1
                if self.curr_val_loss < self.best_val_loss:
                    self.best_val_loss = self.curr_val_loss
                    no_better_val_step = 0
                else:
                    if not b:
                        self.optim_config['learning_rate'] *= self.lr_decay
                        logger.info("learning rate: {0}".format(self.optim_config['learning_rate']))

                if no_better_val_step >= self.max_epochs_no_best:
                    logger.info('trained {0} epochs'.format(epoch))
                    break

        end = time.time()
        logger.info('total duration for training: {0}'.format(end-start))


    def validate(self, sess, model, va):
        logger.info('start validate')
        validate_start = time.time()
        total_loss = 0.
        total_corr = 0
        batch = 1
        n_batch = math.ceil(len(va.data)/self.batch_size)
        for inpt in va:
            fetchs = [model.loss, model.corr]
            add_feed_dict = {model.dropout_placeholder: 1.0}
            batch_loss, batch_corr = self.step(sess, model, inpt, fetchs, add_feed_dict)
            total_loss += (batch_loss*inpt[0].shape[0])
            total_corr += batch_corr.sum()
            logger.info('validation batch {0}/{1} loss {2}'.format(batch, n_batch, batch_loss))
            batch += 1
        val_loss = total_loss / len(va.data)  #can't average by batch_loss
        val_acc  = total_corr / len(va.data)
        logger.info('validation loss and acc: {0} and {1}'.format(val_loss, val_acc))

        validate_end = time.time()
        logger.info('duration for validation: {0}\n'.format(validate_end - validate_start))
        return val_loss


    def infer(self, sess, model, te):
        logger.info('start test')
        test_start = time.time()
        batch = 1
        n_batch = math.ceil(len(te.data)/self.batch_size)
        results = []
        for inpt in te:
            fetchs = model.pred#model.infer
            add_feed_dict = {model.dropout_placeholder: 1.0}
            probs  = self.step(sess, model, inpt, fetchs, add_feed_dict)
            results.append(probs)
            logger.info('test batch {0}/{1}'.format(batch, n_batch))
            batch += 1
        results = np.concatenate(results, axis=0)
        np.save(os.path.join(self.result_path, 'infer_score.npy'), results)

        test_end = time.time()
        logger.info('duration for test: {0}'.format(test_end - test_start))


    def step(self, sess, model, inpt, fetchs, add_feed_dict={}):
        feed_dict = {model.input_placeholder: inpt[0],
                     model.label_placeholder: inpt[1],
                     self.lr: self.optim_config['learning_rate']}
        feed_dict.update(add_feed_dict)
        run_ret = sess.run(fetchs, feed_dict=feed_dict)
        return run_ret


    def save(self, sess, saver, epoch=0):
        if 'interval' in self.save_mode and epoch % self.save_interval == 0:
            save_path = saver.save(sess, self.save_paths[1], global_step=sess.graph.get_collection('global_step')[0])
            logger.info('model saved in file {0}'.format(save_path))
        if 'best' in self.save_mode and self.is_validate:
            if self.curr_val_loss < self.best_val_loss:
                save_path = saver.save(sess, self.save_paths[2], write_state=False)
                logger.info('model saved in file {0}'.format(save_path))


    def add_train_op(self, model):
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])
        if self.update_rule == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.update_rule == "sgd_momentum":
            momentum = self.optim_config.get('momentum', 0.9)
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum)
        elif self.update_rule == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.update_rule == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            raise Exception("no optimizer you want")

        grads_and_vars = optimizer.compute_gradients(model.loss)
        grads, varss = zip(*grads_and_vars)
        model.global_norm_before_clip = tf.global_norm(grads) # self.my_calculate_global_norm = tf.sqrt( tf.add_n( [tf.norm(grad, 2)**2 for grad in grads] ) )
        if self.clip_grad:
            tmp_grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_grad)
            grads = tmp_grads
        model.global_norm_after_clip  = tf.global_norm(grads)
                                                    
        grads_and_vars = zip(grads, varss)
        train_op = optimizer.apply_gradients(grads_and_vars, name="train_op")
                                                                            
        ## add some var to instance for debugging conveniently
        model.summary['grads'] = grads
        return train_op




