# -*- coding: utf-8 -*-

import os
import argparse
import time

import numpy as np
import tensorflow as tf

from config import *
from model import ModelMnist
from data_loader import LoadMnist
from .solver_example import SolverExample


class SolverMnist(SolverExample):
    def __init__(self, kargs):
        super(SolverMnist, self).__init__(kargs)


    def extend_train_fetchs(self, model):
        return [model.global_norm_before_clip, model.global_norm_after_clip]
    

    def logger_extend_train_info(self, extend_fetchs):
        grads_norm_bef, grads_norm_aft = extend_fetchs
        logger.info('grads: {0}/{1}'.format(grads_norm_bef, grads_norm_aft))


    def validate(self, sess, model, va):
        logger.info('start validate')
        validate_start = time.time()
        total_loss = 0.
        total_corr = 0
        batch = 1
        n_batch = va.n_data
        for inpt in va:
            fetchs = [model.loss, model.corr]
            batch_loss, batch_corr = self.step(sess, model, inpt, fetchs)
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

