# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from config import *
from .solver_quanquan import SolverQuanQuan
from model import Model1
from data_loader import Input2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto(log_device_placement=False)
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8


def train(mode, training_param, input_param, model_param):
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session(config=tf_config) as sess:
            solver = SolverQuanQuan(training_param)

            model = Model1(model_param)
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
#               print(tf.get_variable('conv2_1/w:0').trainable)
#               print(tf.get_variable('conv3_1/w:0').trainable)
#               print(tf.get_variable('conv4_1/w:0').trainable)
                saver = tf.train.Saver()
                saver.restore(sess, training_param['restore_path'][0])

#               training_param['update_rule'] = 'adam'
#               model.train_op = solver.add_train_op(model)
#               print(tf.get_variable('conv2_1/w:0').trainable)
#               print(tf.get_variable('conv3_1/w:0').trainable)
#               print(tf.get_variable('conv4_1/w:0').trainable)

                tr = Input2(input_param['trainfile'], input_param)
                va = Input2(input_param['validfile'], input_param)
                solver.train(sess, model, (tr, va), saver)

            elif mode == 'test':
                saver = tf.train.Saver()
                saver.restore(sess, training_param['restore_path'][0])

                va = Input2(input_param['trainfile'], input_param)
                solver.validate(sess, model, va)
#               te = Input2(input_param['validfile'], input_param)
#               solver.infer(sess, model, te)



def main(mode):
    training_param = {'update_rule': 'sgd',
                      'optim_config': {
                          'learning_rate': 1e-3,
                          'momentum': 0.9,
                          },
                      'clip_grad': 500,
                      'lr_decay':  0.6,
                      'decay_epoch': 100,

                      'num_epochs': 200,
                      'batch_size': 128,
                      'max_epochs_no_best': 10,

                      'save_interval': 1,
                      'save_mode': ['interval', 'best'],
                      'save_paths':   [os.path.join(SAVED_MODEL_PATH, '190109-1500/model_init.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190109-1500/model.ckpt'),
                                       os.path.join(SAVED_MODEL_PATH, '190109-1500/model_best.ckpt')],
                      'restore_path': [os.path.join(SAVED_MODEL_PATH, '190109-1500/model_init.ckpt')],
                      'is_validate': True,

                      'result_path': os.path.join(SAVED_MODEL_PATH, '190109-1500/results/190109-1500'),

                      'print_every': 10,
                      'verbose': True,
                      }
    input_param = {'imgpath': '',
                   'queue_size': 2000,
                   'batch_size': training_param['batch_size'],
                   'cache_size': 4000,
                   'w' : 224,
                   'h' : 224,
                   'labelfile': os.path.join(RESOURCE_PATH, 'dataset/quanquan/manual_category/labels_190107-2000.txt'),
                   'trainfile': os.path.join(RESOURCE_PATH, 'dataset/quanquan/manual_category/train_190107-2000.txt'),
                   'validfile': os.path.join(RESOURCE_PATH, 'dataset/quanquan/manual_category/validate_190107-2000.txt'),
                   'testfile': None,
                   'mode': mode
                  }
    model_param = {'height': input_param['w'],
                   'width':  input_param['h'],
                   'n_label': 34,
                   }
    train(mode, training_param, input_param, model_param)



