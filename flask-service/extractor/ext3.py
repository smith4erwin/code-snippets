# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import pickle
import threading

import cv2
import yaml
import keras
import scipy.linalg
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.preprocessing import image
from keras import backend as K
from keras.backend.tensorflow_backend import set_session, get_session
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

from sklearn.decomposition import PCA
from sklearn.externals import joblib

from .utils import *
from config import * 


class Ext3(object):
    def __init__(self, **kargs):
        self.auto_load = kargs.get('auto_load', True)

    
    def readcfg(self, cfg):
        with open(cfg, 'r') as f:
            self.args = yaml.load(f)
        logger.info("======= extractor config =======")
        logger.info('\n'+yaml.dump(self.args))

        # general property
        assert 'ext_algo' in self.args
        assert 'vecdir'   in self.args
        assert 'dbfile'   in self.args
        assert 'dbpath'   in self.args
        assert 'params'   in self.args
        assert 'vecf'     in self.args
        assert 'dim'      in self.args
        assert 'n_feature' in self.args
        assert os.path.isdir(self.args['vecdir'])

        # own property
        assert 'batch_size'   in self.args['params']
        assert 'pca'          in self.args['params']
        assert 'pca_ndim'     in self.args['params']
        assert 'model'        in self.args['params']

        # init something
        imglst_file = os.path.join(self.args['vecdir'], 'img.lst')
        if not os.path.exists(imglst_file):
            open(imglst_file, 'w').close()

        with open(imglst_file, 'r') as f:
            lines = f.readlines()
            self.imglst = [line.rstrip('\n').split('\t') for line in lines]

        # check len(self.imglst) == number of feature
        if len(self.args['vecf']) != 0:
            vecfs, vecf_n_feature = list(zip(*self.args['vecf']))
            assert len(vecfs) == len(set(vecfs))
            assert len(self.imglst) == sum(vecf_n_feature)
            assert len(self.imglst) == self.args['n_feature']
        else:
            assert len(self.imglst) == 0
            assert self.args['n_feature'] == 0

        raw_infoid = set([infoid for infoid, url, _ in self.imglst])
        logger.info('{0} images have been extracted'.format(len(raw_infoid)))

        with open(self.args['dbfile'], 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split('\t') for line in lines]
        self.imgnames = [[line[0], line[1], line[2]] for line in lines if line[0] not in raw_infoid]
        logger.info("{0} images haven't been extracted".format(len(self.imgnames)))
    
        self.imgpath = self.args['dbpath']
        self.has_load_model = False
        if not self.has_load_model and self.auto_load:
            self.load_model()


    def writecfg(self):
        fullpath = os.path.join(self.args['vecdir'], 'vec.yaml')
        with open(fullpath, 'w') as f:
            yaml.dump(self.args, f)

        fullpath = os.path.join(self.args['vecdir'], 'img.lst')
        with open(fullpath, 'w') as f:
            for item in self.imglst:
                f.write('\t'.join(item)+'\n')


    def __save(self, end_vec_id, end_vec, end_lst):
        vecname = str(end_vec_id)+'.npy'

        self.imglst.extend(end_lst)
        self.args['n_feature'] = len(self.imglst)

        if len(self.args['vecf']) == 0 or vecname != self.args['vecf'][-1][0]:
            self.args['vecf'].append([vecname, end_vec.shape[0]])
        elif vecname == self.args['vecf'][-1][0]:
            self.args['vecf'][-1] = [vecname, end_vec.shape[0]]

        vecfs, vecf_n_feature = list(zip(*self.args['vecf']))
        assert len(vecfs) == len(set(vecfs))
        assert len(self.imglst) == sum(vecf_n_feature)
        assert len(self.imglst) == self.args['n_feature']

        np.save(os.path.join(self.args['vecdir'], vecname), end_vec)
        self.writecfg()


    def save(self, end_vec_id, end_vec, end_lst, vec, imgnames):
        save_size = 10000
        remain_size = save_size - end_vec.shape[0]
        # assert vec.shape[0] <= save_size
        v1, v2 = np.vsplit(vec, [remain_size])
        if end_vec.shape[1] == 0:
            end_vec = np.zeros((0, v1.shape[1]), dtype=np.float32)
        end_vec = np.concatenate([end_vec, v1], axis=0)

        l1, l2 = imgnames[:remain_size], imgnames[remain_size:]
        end_lst.extend(l1)

        if end_vec.shape[0] >= save_size:
            self.__save(end_vec_id, end_vec, end_lst) 
            end_vec_id += 1
            end_vec = v2
            end_lst = l2
        return end_vec, end_vec_id, end_lst


    def load(self):
        vecfs = [vecf for vecf, n_feature in self.args['vecf']]
        return Vec(self.args['vecdir'], vecfs)


    def __load_model(self):
        if K.tensorflow_backend._SESSION is None:
            logger.info('############# set session #############')
            set_session(tf.Session(config=gpu_config))
        s = get_session()
#       print(s)
#       print(s.graph)        
        with s.graph.as_default():
#           print(tf.get_default_graph())
#           tf.reset_default_graph() 

            self.model = load_model(self.args['params']['model'])
#           self.model = keras.applications.VGG16(weights=self.args['params']['model'], include_top=True)
            self.feat_extractor = Model(inputs=self.model.input, outputs=self.model.get_layer('fc2').output)

        # keras bug when using with flask/django
        a = [1]
        a.extend(self.model.input.shape[1:])
        b = np.zeros(a, dtype=np.float32)
        self.feat_extractor.predict(b)
        
#       print(self.model.input.graph)
#       print(self.feat_extractor.input.graph)
#       print(self.feat_extractor.output.graph)

        self.pca = joblib.load(self.args['params']['pca'])
        self.pca_ndim = self.args['params']['pca_ndim']
        self.pca_components = self.pca.components_.T[:, :self.pca_ndim]

        np.random.seed(2)
#       self.orth, _ = scipy.linalg.qr(np.random.randn(self.pca_ndim, self.pca_ndim))
        self.orth, _ = scipy.linalg.qr(np.load(self.args['params']['orth']))
        self.orth = self.orth.astype(np.float32)
        # assert np.allclose(self.orth.dot(self.orth.T), np.eye(self.pca_ndim))
        self.trans_mat = self.pca_components.dot(self.orth)

        self.has_load_model = True


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


    def extract(self, img):
        img = self.resize(img)
        img = self.preprocess(img)
        vec = self.extract_small_batch([img])
        return vec


    def extract_offline(self, n_thread=50):
        logger.info('===== extract offline =====')
        bs = self.args['params']['batch_size']
        n_imgs  = len(self.imgnames)
        n_batch = math.ceil(n_imgs/bs)
        n_loop  = math.ceil(n_batch/n_thread)

        if len(self.args['vecf']) != 0:
            end_vecf = self.args['vecf'][-1]
            end_vec_id = len(self.args['vecf']) - 1 #int(end_vecf[0].split('.')[0])
            end_vec  = np.load(os.path.join(self.args['vecdir'], end_vecf[0]))
        else :
            end_vecf = ['', 0]
            end_vec_id = 0
            end_vec  = np.zeros((0,self.args['dim']), dtype=np.float32)
        end_lst = []

        logger.info('n_imgs, n_batch, n_loop, n_thread, end_vecf: {0} {1} {2} {3} {4}'.format(n_imgs, n_batch, n_loop, n_thread, end_vecf))
        logger.info("start extracting {0} images' features".format(n_imgs))
        for i in range(n_loop):
            s_batch = i*n_thread
            e_batch = i*n_thread+n_thread
            if e_batch > n_batch:
                e_batch = n_batch

            task_threads = []
            imgdict = {}
            for j in range(s_batch, e_batch):
                batch_names = [item[0]+'.jpg' for item in self.imgnames[j*bs:j*bs+bs]]
                task = threading.Thread(target=self.task_read, args=(batch_names, imgdict, j))
                task_threads.append(task)
            for task in task_threads:
                task.start()
            for task in task_threads:
                task.join()

            vecdict = {}
            for j in range(s_batch, e_batch):
                vecdict[j] = self.extract_small_batch(imgdict[j], j) 

            for j in range(s_batch, e_batch):
                self.args['dim'] = vecdict[j].shape[-1]
                end_vec, end_vec_id, end_lst = self.save(end_vec_id, end_vec, end_lst, vecdict[j], self.imgnames[j*bs:j*bs+bs])
                        
        if end_vec.shape[0] != 0:
            self.__save(end_vec_id, end_vec, end_lst)


    def task_read(self, batch_names, imgdict, j_batch):
        st = time.time()
        batch_imgs = self.read_imgs(batch_names)
        et = time.time()
        logger.info('    ====== read and preprocess {0}th batch {1} images costs: {2}'.format(j_batch, len(batch_names), et-st))
        
        imgdict[j_batch] = batch_imgs


    def extract_small_batch(self, imgs, j_batch=None):
        st = time.time()
        imgs = np.stack(imgs, axis=0)
        vec = self.feat_extractor.predict_on_batch(imgs)
        et1 = time.time()
        logger.debug('predict cost: {0}'.format(et1-st))

#       vec = (vec - self.pca.mean_).dot(self.pca_components)
#       vec = vec.dot(self.orth)
        vec = (vec-self.pca.mean_).dot(self.trans_mat) 
        et2 = time.time()
        logger.debug('transform cost:{0}'.format(et2-et1))

        if j_batch is not None:
            logger.debug('    ====== extract {0}th batch {1} images costs: {2}'.format(j_batch, vec.shape[0], et2-st))
        
        return vec


    def read_imgs(self, imgnames):
        # before preprocess, must resize
        imgs = []
        for name in imgnames:
            img = image.load_img(os.path.join(self.imgpath, name), target_size=self.model.input_shape[1:3])
            img = image.img_to_array(img)
#            img = self.resize(img)
            img = self.preprocess(img)
            imgs.append(img)
        return imgs


    def preprocess(self, img):
        return preprocess_input(img)


    def resize(self, img):
        return cv2.resize(img, self.model.input_shape[1:3])



