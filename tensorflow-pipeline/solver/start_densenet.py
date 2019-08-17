# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from config import *
from model.densenet.model import Model
from data_loader import LoadCifar
from .solver_densenet import SolverDensenet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto(log_device_placement=False)
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9


def train(mode, training_param, input_param, model_param):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session(config=tf_config) as sess:
            solver = SolverDensenet(training_param)

            model = Model(model_param)
            model.train_op = solver.add_train_op(model) # 在这里把weight_decay加进去

            global_step = tf.get_variable("global_step", dtype=tf.int32, shape=[], initializer=tf.zeros_initializer, trainable=False)
            global_step_add = global_step.assign_add(1)
            g1.add_to_collection("global_step", global_step_add)

            if mode == 'init':
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                
                saver = tf.train.Saver(max_to_keep=100)
                save_path = saver.save(sess, training_param['save_paths'][0])
                logger.info('Model saved in file: {0}'.format(save_path))

            elif mode == 'train':
                saver = tf.train.Saver(max_to_keep=100)
                saver.restore(sess, training_param['restore_path'][0])

                tr = LoadCifar(input_param['trainfile'], input_param)
                va = LoadCifar(input_param['validfile'], input_param)
                solver.train(sess, model, (tr, va), saver)

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
                          'learning_rate': 1e-3,
                          'momentum': 0.9,
                          'use_nesterov': True,
                          },
                      'weight_decay': 1e-4,
                      'clip_grad': 0,
                      'lr_decay':  0.6,
                      'decay_epoch': 300,

                      'num_epochs': 150,
                      'batch_size': 64,
                      'max_epochs_no_best': 200,

                      'save_interval': 1,
                      'save_mode': ['interval', 'best'],
                      'save_paths':   [os.path.join(SAVED_MODEL_PATH, '190218-2000/model_init.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190218-2000/model.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190218-2000/model_best.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190218-2000/model_init.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190218-2000/model_best.ckpt')],
                      'restore_path': [os.path.join(SAVED_MODEL_PATH, '190218-2000/model.ckpt-207')],

#                     'save_paths':   [os.path.join(SAVED_MODEL_PATH, '190221-1030/model_init.ckpt'),
#                                      os.path.join(SAVED_MODEL_PATH, '190221-1030/model.ckpt'),
#                                      os.path.join(SAVED_MODEL_PATH, '190221-1030/model_best.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190221-1030/model_init.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190221-1030/model_best.ckpt')],
#                     'restore_path': [os.path.join(SAVED_MODEL_PATH, '190221-1030/model.ckpt-225')],

                      'is_validate': True,

                      'result_path': os.path.join(SAVED_MODEL_PATH, '181219-1740/results/181219-1740'),

                      'print_every': 10,
                      'verbose': True,
                      }
    cifar = 100
    imgpath = os.path.join(RESOURCE_PATH, 'dataset/cifar/cifar-100-python') if cifar == 100 \
              else os.path.join(RESOURCE_PATH, 'dataset/cifar/cifar-10-batches-py')
    input_param = {'imgpath': imgpath,
                   'queue_size': 50000,
                   'batch_size': 64,
                   'trainfile': 'train',
                   'validfile': 'valid',
                   'testfile': 'test',
                   'mode': mode,
                   'cifar': cifar,
                   }
    model_param = {'height': 32,
                   'width':  32,
                   'n_label': cifar,

                   'n_filter_conv0': 16,
                   'n_layers': [12, 12, 12],
                   'growth_rate': 12,
                   'bottleneck': False,
                   'compression_rate': 1.,

                   'bn_momentum': 0.9,
                   'keep_prob': 0.8}
                   
    train(mode, training_param, input_param, model_param)

