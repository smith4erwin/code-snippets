# -*- coding: utf-8 -*-

import os
import yaml
import faiss
import numpy as np


from config import *
import extractor


class Idx1(object):
    def __init__(self, **kargs):
        pass


    def readcfg(self, cfg):
        with open(cfg, 'r') as f:
            self.args = yaml.load(f)
        logger.info("======= index config =======")
        logger.info('\n'+yaml.dump(self.args))

        # general property
        assert 'idx_algo'  in self.args
        assert 'idxdir'    in self.args
        assert 'ext_algo'  in self.args
        assert 'vecdir'    in self.args
        assert 'params'    in self.args
        assert 'dim'       in self.args
        assert 'n_feature' in self.args
        assert 'idxf'      in self.args

        # own property
        assert 'nlist' in self.args['params']
        assert 'm'     in self.args['params']
        assert 'k'     in self.args['params']

        # init something

        self.extor = getattr(extractor, self.args['ext_algo'])(auto_load=False)
        self.extor.readcfg(os.path.join(self.args['vecdir'], 'vec.yaml'))
        assert self.extor.args['ext_algo'] == self.args['ext_algo']
        assert self.extor.args['vecdir']   == self.args['vecdir']

        self.args['dim'] = self.extor.args['dim']
        # index algorithm
        self.quantizer = faiss.IndexFlatL2(self.args['dim'])

        # kmeans + ivf
        #self.index = faiss.IndexIVFFlat(self.quantizer, self.args['dim'], self.args['params']['nlist'], faiss.METRIC_L2)
        # kmeans + ivf + pq
        m = self.args['params']['m']
        k = self.args['params']['k']
        nlist = self.args['params']['nlist']
        self.index = faiss.IndexIVFPQ(self.quantizer, self.args['dim'], nlist, m, k)
        # 
        #self.index = self.quantizer


    def writecfg(self, idxdir, cfgname, cfg):
        fullpath = os.path.join(idxdir, cfgname)
        with open(fullpath, 'w') as f:
            yaml.dump(cfg, f)


    def save(self, idxdir, idxname, idx):
        fullpath = os.path.join(idxdir, idxname)
        faiss.write_index(idx, fullpath)


    def load(self):
        self.index = faiss.read_index(os.path.join(self.args['idxdir'], self.args['idxf']))


    def train(self):
        # whether train from scratch
        from_scratch = True
        if self.args['n_feature'] - self.args['last_trained_features'] < 1000:
#       if self.extor.args['n_feature'] - self.args['n_feature'] < 1000:
            from_scratch = False

        logger.info("===== train {0}'s index =====".format(type(self).__name__))

        vec_iter = self.extor.load()
        vecs = [vec for vec in vec_iter]
        vecs = np.concatenate(vecs, axis=0)

        if not from_scratch:
            logger.info('=== just add features')
            add_feature = vecs[self.args['n_feature']:]
            self.args['n_feature'] += add_feature.shape[0]  
            logger.info('=== added features shape: {0}*{1}'.format(*add_feature.shape))
            self.load()
            self.index.add(add_feature)
        else:
            logger.info('=== train from scratch') 
            self.xb = vecs
            self.args['n_feature'] = self.xb.shape[0]
            logger.info("=== features: {0}*{1}, {2}===".format(*self.xb.shape, self.xb.dtype))
            self.index.train(self.xb)
            self.index.add(self.xb)
            self.args['last_trained_features'] = self.args['n_feature']
        
        self.save    (self.args['idxdir'], 'idx.st',   self.index)
        self.writecfg(self.args['idxdir'], 'idx.yaml', self.args)


    def search(self, xq, k):
        self.index.nprobe = int(self.args['params']['nlist']*0.2)
        if xq.ndim == 1:
            xq = xq.reshape(1, -1)
        D, I = self.index.search(xq, k)

        I = np.split(I, len(I), axis=0)
        D = np.split(D, len(D), axis=0)
        for i in range(len(I)):
            D[i] = D[i][I[i] != -1]
            I[i] = I[i][I[i] != -1]
        return D, I


    def get_imginfo(self, I):
        I = I[0]
        infoids = [self.extor.imglst[idx][0]  for idx in I]
        urls    = [self.extor.imglst[idx][1]  for idx in I]
        return infoids, urls







