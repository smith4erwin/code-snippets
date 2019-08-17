# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from config import *
from model import ModelMnist
from data_loader import LoadMnist
from .solver_mnist import SolverMnist

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto(log_device_placement=False)
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8


def train(mode, training_param, input_param, model_param):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session(config=tf_config) as sess:
            solver = SolverMnist(training_param)

            model = ModelMnist(model_param)
            model.train_op = solver.add_train_op(model)

            global_step = tf.get_variable("global_step", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer, trainable=False)
            global_step_add = global_step.assign_add(1)
            g1.add_to_collection("global_step", global_step_add)

            if mode == 'init':
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                
                saver = tf.train.Saver()
                save_path = saver.save(sess, training_param['save_paths'][0])
                logger.info('Model saved in file: {0}'.format(save_path))

            elif mode == 'train':
                saver = tf.train.Saver()
                saver.restore(sess, training_param['restore_path'][0])

                tr = LoadMnist(input_param['trainfile'], input_param)
                va = LoadMnist(input_param['validfile'], input_param)
                solver.train(sess, model, (tr, va), saver)

            elif mode == 'test':
                saver = tf.train.Saver()
                saver.restore(sess, training_param['restore_path'][0])

#               va = Input2(input_param['trainfile'], input_param)
#               solver.validate(sess, model, va)
                te = LoadMnist(input_param['validfile'], input_param)
                solver.infer(sess, model, te)



def main(mode):
    training_param = {'update_rule': 'sgd',
                      'optim_config': {
                          'learning_rate': 1e-3,
                          'momentum': 0.9,
                          },
                      'clip_grad': 100,
                      'lr_decay':  0.6,
                      'decay_epoch': 3,

                      'num_epochs': 200,
                      'batch_size': 32,
                      'max_epochs_no_best': 10,

                      'save_interval': 1,
                      'save_mode': ['interval', 'best'],
                      'save_paths':   [os.path.join(SAVED_MODEL_PATH, '181219-1740/model_init.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '181219-1740/model.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '181219-1740/model_best.ckpt')],
                      'restore_path': [os.path.join(SAVED_MODEL_PATH, '181219-1740/model_init.ckpt')],
                      'is_validate': True,

                      'result_path': os.path.join(SAVED_MODEL_PATH, '181219-1740/results/181219-1740'),

                      'print_every': 10,
                      'verbose': True,
                      }
    input_param = {'imgpath': os.path.join(RESOURCE_PATH, 'dataset/mnist'),
                   'queue_size': 1000,
                   'batch_size': 256,
                   'trainfile': 'train',
                   'validfile': 'valid',
                   'testfile': 'test',
                   'mode': mode,
                   }
    model_param = {}
                   
    train(mode, training_param, input_param, model_param)

