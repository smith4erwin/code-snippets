# -*- coding: utf-8 -*-

import os
import math
import time
import threading

import cv2
import yaml
import numpy as np

from .utils import *
from config import *
from tools import resize 

class Ext2(object):
    def __init__(self, **kargs):
        pass


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
        assert 'binss'        in self.args['params']
        assert 'colorspace'   in self.args['params']
        assert 'norm'         in self.args['params']
        assert 'sep_or_joint' in self.args['params']
        assert 'grid'         in self.args['params']
#       assert 'rmside'       in self.args['params']
        assert 'algo'         in self.args['params']
        if 'ccv_thres' in self.args['params']:
            pass

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
        self.ext_func = getattr(self, self.args['params']['algo'])


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


    def extract(self, img):
        img = self.resize(img)
        img = self.preprocess(img)
        vec = self.ext_func([img])
        return vec


    def extract_offline(self, n_thread=3):
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
            vecdict = {}
            for j in range(s_batch, e_batch):
                batch_names = [item[0]+'.jpg' for item in self.imgnames[j*bs:j*bs+bs]]
                task = threading.Thread(target=self.task_extract, args=(batch_names, vecdict, j))
                task_threads.append(task)
            for task in task_threads:
                task.start()
            for task in task_threads:
                task.join()

            for j in range(s_batch, e_batch):
                self.args['dim'] = vecdict[j].shape[-1]
                end_vec, end_vec_id, end_lst = self.save(end_vec_id, end_vec, end_lst, vecdict[j], self.imgnames[j*bs:j*bs+bs])

        if end_vec.shape[0] != 0:
            self.__save(end_vec_id, end_vec, end_lst)
#           vecname = str(end_vec_id) + '.npy'

#           self.imglst.extend(end_lst)
#           self.args['n_feature'] = len(self.imglst)

#           if len(self.args['vecf']) == 0 or vecname != self.args['vecf'][-1][0]:
#               self.args['vecf'].append([vecname, end_vec.shape[0]])
#           elif vecname == self.args['vecf'][-1][0]:
#               self.args['vecf'][-1] = [vecname, end_vec.shape[0]]

#           vecfs, vecf_n_feature = list(zip(*self.args['vecf']))
#           assert len(vecfs) == len(set(vecfs))
#           assert len(self.imglst) == sum(vecf_n_feature)
#           assert len(self.imglst) == self.args['n_feature']

#           np.save(os.path.join(self.args['vecdir'], vecname), end_vec)
#           self.writecfg()


    def task_extract(self, batch_names, vecdict, j_batch):
        #time.sleep(1)
        t1 = time.time()

        batch_imgs = self.read_imgs(batch_names)
        t2 = time.time()
        logger.info('    ====== read and preprocess {0}th batch {1} images costs: {2}'.format(j_batch, len(batch_names), t2-t1))
       
#       vec = self.gcf(batch_imgs)
#       vec = self.lcf(batch_imgs)
        vec = self.ext_func(batch_imgs)
        t3 = time.time()
        logger.info('    ====== extract {0}th batch {1} images costs: {2}'.format(j_batch, len(batch_names), t3-t2))

        vecdict[j_batch] = vec
        # is list or dict thread-safe?
        # vecname = str(j_batch)+'.npy'
        # self.save(self.args['vecdir'], vecname, vec)
        # self.args['vecf'].append(vecname)
        # self.args['dim'] = vec.shape[-1]

    def gcf(self, imgs):
        ext_func = getattr(self, self.args['params']['sep_or_joint']+'_hist_grid')
        all_hist = []
        for i in range(len(imgs)):
            hist = ext_func(imgs[i])
            all_hist.append(hist)
        all_hist = np.concatenate(all_hist, axis=0)
        return all_hist


    def lcf(self, imgs):
        # local seperate or joint hist
        # when grid=[1,1], equals to global hist
        grid = self.args['params']['grid']
        n_grid = grid[0] * grid[1]
        ext_func = getattr(self, self.args['params']['sep_or_joint']+'_hist_grid')
        all_hist = []
        for i in range(len(imgs)):
            h, w = imgs[i].shape[:2]
            hh = np.linspace(0, h, grid[0]+1, dtype=np.int)
            ww = np.linspace(0, w, grid[1]+1, dtype=np.int)
            hs, ws = np.meshgrid(hh[:-1], ww[:-1])
            he, we = np.meshgrid(hh[1 :], ww[1: ])
            a = list(zip(hs.T.ravel(), ws.T.ravel()))
            b = list(zip(he.T.ravel(), we.T.ravel()))
            xys = list(zip(a,b))
            blocks = []
            for xy in xys:
                blocks.append(imgs[i][xy[0][0]:xy[1][0], xy[0][1]:xy[1][1], :])
            
            if len(blocks) == 1:
                hist = ext_func(blocks[0])
            else:
                if n_grid < 30:
#               st = time.time()
                    task_threads = []
                    vecdict = {}
                    for j, block in enumerate(blocks):
                        task = threading.Thread(target=self.extract_block, args=(ext_func, block, vecdict, j))
                        task_threads.append(task)
                    for task in task_threads:
                        task.start()
                    for task in task_threads:
                        task.join()
                    
                    hist = []
                    for j_block in range(len(blocks)):
                        hist.append(vecdict[j_block])
                    hist = np.concatenate(hist, axis=1)
#               et1 = time.time()
                
                else:
                    hist = []
                    for block in blocks:
                        vec = ext_func(block)
                        hist.append(vec)
                    hist = np.concatenate(hist, axis=1)
#                   assert np.allclose(hist, hist1)
#               et2 = time.time()
#               print("blocks multithread: {0}".format(et1-st))
#               print("blocks singlethread: {0}".format(et2-et1))
            
            all_hist.append(hist)
        all_hist = np.concatenate(all_hist, axis=0)
        return all_hist


    def extract_block(self, ext_func, block, vecdict, j_block):
        vec = ext_func(block)
        vecdict[j_block] = vec


    def sep_hist_grid(self, img):
        # img ndim must be 3, if not, transform it
        assert img.ndim == 3
        nc = img.shape[-1]
        hist = []
        for c in range(nc):
            fea = np.histogram(img[:,:,c], self.args['params']['binss'][c])[0]
            if self.args['params']['norm'] == 1:
                fea = fea * 1./sum(fea)
            elif self.args['params']['norm'] == 2:
                fea = fea * 1./np.sqrt(sum(fea**2))

            if self.args['params']['rmside'] == 1:
                fea = fea[1:-1]
            hist.append(fea)

        hist = np.concatenate(hist, axis=0)
        hist.shape = (1, hist.shape[0])
        hist = hist.astype(np.float32)
        return hist
    

    def joint_hist_grid(self, img):
        assert img.ndim == 3
        nc = img.shape[-1]
        c = []
        for j in range(nc):
            c.append(img[:,:,j].ravel().reshape(-1,1))
        hist = np.histogramdd(np.concatenate(c, axis=1), self.args['params']['binss'])[0].ravel()
        
        if self.args['params']['norm'] == 1:
            hist = hist*1./sum(hist)
        elif self.args['params']['norm'] == 2:
            hist = hist*1./np.sqrt(sum(fea**2))

        if self.args['params']['rmside'] == 1:
            shp = [len(self.args['params']['binss'][0]), \
                   len(self.args['params']['binss'][1]), \
                   len(self.args['params']['binss'][2]), ]
            hist = hist.reshape(*shp)
            hist = hist[1:-1,1:-1,1:-1].ravel()
        
        hist.shape = (1, hist.shape[0])
        hist = hist.astype(np.float32)
        return hist


    def ccv(self, imgs):
        ccv_thres = self.args['params']['ccv_thres']
        binss = self.args['params']['binss']
        norm  = self.args['params']['norm']
        sep_or_joint = self.args['params']['sep_or_joint']
        all_feats = []
        for i in range(len(imgs)):
#           t1 = time.time()
            img = imgs[i]
            h, w, nc = img.shape
            thres = h*w*ccv_thres
            if sep_or_joint == 'sep':
                feats = []
                for i in range(3):
                    quan_img = quantize_img(img[:,:,i], binss[i], 'sep')
                    feat = ccv_core(quan_img, thres, len(binss[i])-1 )
                    feats.extend(feat)
            
            if sep_or_joint == 'joint':
                quan_img = quantize_img(img, binss, 'joint')
                feats = ccv_core(quan_img, thres, (len(binss[0])-1) * (len(binss[1])-1) * (len(binss[2])-1))
    
            feats = np.array(feats, dtype=np.float32).reshape(1,-1)
            if norm == 1:
                feats = feats*1./feats.sum()
            elif norm == 2:
                feats = feats*1./np.sqrt((feats**2).sum())

            all_feats.append(feats)
#           t2 = time.time()
#           print(t2-t1)
        all_feats = np.concatenate(all_feats, axis=0)
        all_feats = all_feats.astype(np.float32)
        return all_feats
    

    def cdh(self, imgs):
        assert self.args['params']['colorspace'] == 'lab'
        binss = self.args['params']['binss']
        binss_ori = np.linspace(0, 2*np.pi+1e-2, 18+1).tolist()
        n_bins_color = (len(binss[0])-1)*(len(binss[1])-1)*(len(binss[2])-1)
        n_bins_ori   = 18
        all_feats = []
        for i in range(len(imgs)):
            img = imgs[i]
            img = img.astype(np.int32)
            h, w, nc = img.shape
#           t1 = time.time()

            quan_lab = quantize_img(img, binss, 'joint')
#           assert quan_lab.max() <= n_bins_color
#           t2 = time.time()

            phi = compute_edge_orientation(img)
#           assert phi.max() <= np.pi*2 and 0 <=  phi.min()
#           t3 = time.time()

            quan_ori = quantize_img(phi, binss_ori, 'sep')
#           assert quan_ori.max() <= n_bins_ori
#           t4 = time.time()

            color_diff_1 = (img[:  , 1:  , :] - img[:   , :-1 , :])[:-1 , 1:   ,:]                   # 0
            color_diff_2 = (img[1: , :   , :] - img[:-1 , :   , :])[:   , 1:-1 ,:]                   # 90
            color_diff_3 = (img[1: , 1:  , :] - img[:-1 , :-1 , :])[:   , 1:   ,:]                   # 45
            color_diff_4 = (img[1: , :-1 , :] - img[:-1 , 1:  , :])[:   , :-1  ,:]                   # 135
            sum_color_diff_1 = np.sqrt(np.power(color_diff_1, 2).sum(axis=2), dtype=np.float32)
            sum_color_diff_2 = np.sqrt(np.power(color_diff_2, 2).sum(axis=2), dtype=np.float32)
            sum_color_diff_3 = np.sqrt(np.power(color_diff_3, 2).sum(axis=2), dtype=np.float32)
            sum_color_diff_4 = np.sqrt(np.power(color_diff_4, 2).sum(axis=2), dtype=np.float32)
#           assert color_diff_1.shape[0] == h-1
#           assert color_diff_1.shape[1] == w-2
#           t5 = time.time()

            feats = np.zeros(n_bins_color+n_bins_ori, dtype=np.float32)

            b2_1 = (quan_ori[:  , 1: ] == quan_ori[:   , :-1])[:-1 , 1:  ]
            b2_2 = (quan_ori[1: , :  ] == quan_ori[:-1 , :  ])[:   , 1:-1]
            b2_3 = (quan_ori[1: , 1: ] == quan_ori[:-1 , :-1])[:   , 1:  ]
            b2_4 = (quan_ori[1: , :-1] == quan_ori[:-1 , 1: ])[:   , :-1 ]
            for j in range(n_bins_color):
                b = (quan_lab == j+1)[:-1, 1:-1]
                
                H_color_j = sum_color_diff_1[b & b2_1].sum() + \
                            sum_color_diff_2[b & b2_2].sum() + \
                            sum_color_diff_3[b & b2_3].sum() + \
                            sum_color_diff_4[b & b2_4].sum()
                feats[j] = H_color_j

            b1_1 = (quan_lab[:  , 1: ] == quan_lab[:   , :-1])[:-1, 1:  ]
            b1_2 = (quan_lab[1: , :  ] == quan_lab[:-1 , :  ])[:  , 1:-1]
            b1_3 = (quan_lab[1: , 1: ] == quan_lab[:-1 , :-1])[:  , 1:  ]
            b1_4 = (quan_lab[1: , :-1] == quan_lab[:-1 , 1: ])[:  , :-1 ]
            for j in range(n_bins_color, n_bins_color+n_bins_ori):
                b = (quan_ori == j-n_bins_color+1)[:-1, 1:-1]
            
                H_ori_j = sum_color_diff_1[b & b1_1].sum() + \
                          sum_color_diff_2[b & b1_2].sum() + \
                          sum_color_diff_3[b & b1_3].sum() + \
                          sum_color_diff_4[b & b1_4].sum()
                feats[j] = H_ori_j
#           t6 = time.time()

#           print(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t6-t1) 
            feats = feats.reshape(1, -1)
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)
        return all_feats


    def read_online(self, img):
        img = np.asarray(bytearray(img), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        h,w = img.shape[0:2]
        img = cv2.resize(img, (w//2, h//2))
        return img


    def read_imgs(self, imgnames):
        imgs = []
        for name in imgnames:
            img = cv2.imread(os.path.join(self.imgpath, name))
            h,w = img.shape[0:2]
            img = cv2.resize(img, (w//2, h//2))
            img = self.preprocess(img)
            imgs.append(img)
        return imgs


    def preprocess(self, img):
        h,w = img.shape[0:2]
        img = cv2.blur(img, (3,3))
        if self.args['params']['colorspace'] == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        elif self.args['params']['colorspace'] == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.args['params']['colorspace'] == 'lab':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return img


    def resize(self, image, maxsize=500):
        image = resize(image, maxsize=maxsize)
        return image


