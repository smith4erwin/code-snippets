# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from config import *
from model.rpn.model import Model
from data_loader import LoadVoc 
from .solver_rpn import SolverRPN

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf_config = tf.ConfigProto(log_device_placement=False)
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9



def train(mode, training_param, input_param, model_param):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session(config=tf_config) as sess:
            solver = SolverRPN(training_param)

            model = Model(model_param)
            solver.train_op = solver.add_train_op(model) # 在这里把weight_decay加进去

            global_step = tf.get_variable("global_step", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer, trainable=False)
            global_step_add = global_step.assign_add(1)
            g1.add_to_collection("global_step", global_step_add)

            if mode == 'init':
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                
                saver = tf.train.Saver(max_to_keep=10)
                save_path = saver.save(sess, training_param['save_paths'][0])
                logger.info('Model saved in file: {0}'.format(save_path))

            elif mode == 'train':
                saver = tf.train.Saver(max_to_keep=10)
                saver.restore(sess, training_param['restore_path'][0])

                tr = LoadVoc(input_param['trainfile'], input_param)
#               va = LoadVoc(input_param['validfile'], input_param)
                solver.train(sess, model, (tr, tr), saver)

            elif mode == 'test':
                saver = tf.train.Saver()
                saver.restore(sess, training_param['restore_path'][0])

#               va = Input2(input_param['trainfile'], input_param)
#               solver.validate(sess, model, va)
                te = LoadMnist(input_param['validfile'], input_param)
                solver.infer(sess, model, te)


def main(mode):
    training_param = {'update_rule': 'sgd_momentum',
                      'optim_config': {
                          'learning_rate': 3e-4,
                          'momentum': 0.9,
                          'use_nesterov': False,
                          },
                      'weight_decay': 5e-4,
                      'clip_grad': 100,
                      'lr_decay':  1.,
                      'decay_epoch': 300,

                      'num_epochs': 100,
                      'max_epochs_no_best': 200,

                      'save_interval': 1,
                      'save_mode': ['interval', 'best'],
                      'save_paths':   [os.path.join(SAVED_MODEL_PATH, '190308-1500/model_init.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190308-1500/model.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190308-1500/model_best.ckpt')],
                      'restore_path': [os.path.join(SAVED_MODEL_PATH, '190308-1500/model_init.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190308-1500/model_best.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190308-1500/model.ckpt-207')],

                      'is_validate': False,

                      'result_path': os.path.join(SAVED_MODEL_PATH, '181219-1740/results/181219-1740'),

                      'print_every': 1,
                      'verbose': True,
                      }
    input_param = {'queue_size': 600,
                   'trainfile': os.path.join(RESOURCE_PATH, 'dataset/voc', 'voc_2007_train'),
                   'validfile': os.path.join(RESOURCE_PATH, 'dataset/voc', 'voc_2007_test'),
                   'testfile': 'test',
                   'mode': mode,
                   }
    model_param = {'lmbda': 10,
                  }
                   
    train(mode, training_param, input_param, model_param)
