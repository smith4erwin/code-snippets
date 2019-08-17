# -*- coding: utf-8 -*-

import os
import sys
import time

import cv2
import yaml
import numpy as np
import tensorflow as tf

from config import *


class Classifier1(object):
    def __init__(self, **kargs):
        self.auto_load = kargs.get('auto_load', True)
        pass


    def readcfg(self, cfg):
        with open(cfg, 'r') as f:
            self.args = yaml.load(f)
        logger.info('======= classifier config ========')
        logger.info('\n'+yaml.dump(self.args))
        
        assert 'labelfile' in self.args
        assert 'model'     in self.args
        
        with open(self.args['labelfile'], 'r') as f:
            labels = [line.rstrip('\n') for line in f.readlines()]
            self.id2label = dict(enumerate(labels))
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
        
        if self.auto_load:
            self.load_model()


    def writecfg(self, cfg):
        pass


    def __load_model(self):
        g = tf.Graph()
        self.sess = tf.Session(config=gpu_config, graph=g)
        with open(self.args['model'], 'rb') as f:
            g_def = tf.GraphDef()
            g_def.ParseFromString(f.read())
        with g.as_default():
            tf.import_graph_def(g_def)
            self.model_input  = g.get_tensor_by_name('import/input:0')
            self.model_output = g.get_tensor_by_name('import/fc_8/xw_plus_b:0')


    def load_model(self):
        try:
            self.__load_model()
            logger.info('===== The first load model success! =====\n')
        except Exception as e:
            logger.error(e)
            logger.error('===== The first load model failed!!! =====\n')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                self.__load_model()
                logger.info('===== The second load model success! =====\n')
            except Exception as e:
                logger.error(e)
                logger.error('===== The second load model failed!!! =====\n')
                sys.exit(1)


    def online_categorize(self, img):
        img = self.resize(img)
        img = self.preprocess(img)
        label = self.categorize(img)
        return label


    def categorize(self, img):
        score = self.sess.run(self.model_output, feed_dict={self.model_input: img})
#       argmax_score = np.argmax(score)
#       return self.id2label[argmax_score]
        argsort_score = np.argsort(score, axis=1)
        return [[(self.id2label[ranking], float(score[0][ranking])) for ranking in argsort_score[i][::-1]] for i in range(len(argsort_score))]
#       return [(self.id2label[ranking], float(score[0][ranking])) for ranking in argsort_score[0][::-1]] 


    def read_imgs(self, imgnames):
        pass


    def preprocess(self, img):
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = np.expand_dims(img, axis=0)
        return img


    def resize(self, img):
        if img.shape[0] < img.shape[1]:
            img = np.transpose(img, axes=(1,0,2))
        return cv2.resize(img, (224, 224))
#       return cv2.resize(img, (192, 320))







