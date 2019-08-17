# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
import subprocess

import yaml
import schedule
import numpy as np
from flask import g

from config import logger, MYPATH
from tools import flask_utils
from tools import download_image
from tools import RWlockReadPreferring
import indexer
import tag

#lock = threading.Lock()


class ImgSearchService(object):
    def __init__(self):
        self.name = 'imgsearch'

        self.n_update = 0
        self.params = yaml.load(open(os.path.join(MYPATH.RESOURCE_PATH, 'config/params.yaml'), 'r'))[self.name]
        logger.info('=== imgsearch params ===')
        logger.info(self.params)

        self.algos = self.params['idx']
        self.load_indexes(self.algos)

        self.update_freq = self.params['update_freq']
        self.update_indexes()

        self.tag_algos = self.params['tag']
        self.load_tagger(self.tag_algos)
        self.offline_update = self.params['offline_update']
#       self.lock = threading.Lock()
        self.lock = RWlockReadPreferring()

    # about requests
    def add_route(self, wrap_app):
        self.wrap_app = wrap_app
        flask_utils.add_route(self.wrap_app.app, mappings=[
            ('/search', self.search),
            ('/tagged', self.tagged),
            ('/tagged_more', self.tagged_more)
            ])


    #  about init 
    def load_indexes(self, algos):
        logger.info('>>>>>>>> initialize indexes')
        try:
            indexes = self.load_all(algos)
        except Exception as e:
            logger.error(e)
            logger.error('FAILED to load indexes on initialization!!!')
            raise
        self.indexes = indexes
        for idxer in self.indexes:
            if idxer.extor.args['name'] == 'vgg':
                idxer.extor.load_model()
                self.vgg_model = [idxer.extor.model, idxer.extor.feat_extractor,\
                        idxer.extor.pca, idxer.extor.trans_mat, idxer.extor.has_load_model]
        logger.info('>>>>>>>> initialize indexes success!')


    def load_tagger(self, algos):
        logger.info('>>>>>>>> load tagger')
        try:
            tag_idxer = None
            for idxer in self.indexes:
                if idxer.extor.args['name'] == 'vgg':
                    tag_idxer = idxer
                    break
            if tag_idxer is None:
                raise(Exception('no vgg index'))

            algo = algos[0]
            algo, configfile = algo
            tagger = getattr(tag, algo)(tag_idxer)
            tagger.readcfg(configfile)
            
        except Exception as e:
            logger.error(e)
            logger.error('FAILED to load tagger!!!')
            raise
        self.tagger = tagger
        logger.info('>>>>>>>> load tagger success!')


    def update_indexes(self):
        task = threading.Thread(target=self.update_sche, args=())
        task.setDaemon(True)
        task.start()


    def update_sche(self):
        schedule.every(self.update_freq).seconds.do(self.update)
        while True:
            schedule.run_pending()
            time.sleep(10)


    def update(self):
        if not self.offline_update:  # 启动多个进程，只能有一个进程负责下载图片，更新特征和索引
            return

        logger.info('>>>>>>>> update indexes {0}'.format(self.n_update))
        prefix = '/home/commonrec/llh/software/glibc-2.23/lib/ld-2.23.so --library-path /home/commonrec/llh/software/glibc-2.23/lib:/lib:/lib/x86_64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu '
        Mypython = prefix+'/home/commonrec/llh/software/anaconda3/bin/python'
        f = open(os.path.join(MYPATH.PROJECT_PATH, 'log/update.log'), 'a')
        cwd = MYPATH.APP_PATH

#       download_py = ' '.join(['python', os.path.join(MYPATH.APP_PATH, 'bin/sql.py')])
        download_py = ' '.join(['python', '-m', 'bin.sql'])
        logger.info('download_py: {0}'.format(download_py))
        subprocess.run(download_py, stdout=f, stderr=f, shell=True, cwd=cwd)

        for index in self.indexes:
            ext = index.args['ext_algo']
            ext_cfg = os.path.join(index.args['vecdir'], 'vec.yaml')
#           ext_py = ' '.join(['python', os.path.join(MYPATH.APP_PATH, 'bin/extidx.py'),\
#                    '-f 1 -a '+ext+' -c '+ext_cfg])
            ext_py = ' '.join(['python', '-m', 'bin.extidx -f 1 -a '+ext+' -c '+ ext_cfg])

            logger.info('ext_py: {0}'.format(ext_py))
            subprocess.run(ext_py, stdout=f, stderr=f, shell=True, cwd=cwd)

            idx = index.args['idx_algo']
            idx_cfg = os.path.join(index.args['idxdir'], 'idx.yaml')
#           idx_py = ' '.join(['python', os.path.join(MYPATH.APP_PATH, 'bin/extidx.py'),\
#                    '-f 2 -a '+idx+' -c '+idx_cfg])
            idx_py = ' '.join(['python', '-m', 'bin.extidx -f 2 -a '+idx+' -c '+idx_cfg])

            logger.info('idx_py: {0}'.format(idx_py))
            subprocess.run(idx_py, stdout=f, stderr=f, shell=True, cwd=cwd)

        tag_algo = self.tagger.args['tag_algo']
        tag_cfg  = os.path.join(self.tagger.args['tagdir'], 'tag.yaml')
#       tag_py   = ' '.join(['python', os.path.join(MYPATH.APP_PATH, 'bin/tag_offline.py'), \
#                  '-a '+tag_algo+' -c '+tag_cfg])
        tag_py = ' '.join(['python', '-m', 'bin.tag_offline', '-a '+tag_algo+' -c '+tag_cfg])
        logger.info('tag_py: {0}'.format(tag_py))
        subprocess.run(tag_py, stdout=f, stderr=f, shell=True, cwd=cwd)
        
        f.close()

        try:
            indexes = self.load_all(self.algos)
            for idxer in indexes:
                if idxer.extor.args['name'] == 'vgg':
                    idxer.extor.model, idxer.extor.feat_extractor, idxer.extor.pca,\
                    idxer.extor.trans_mat, idxer.extor.has_load_model = self.vgg_model
        except Exception as e:
            logger.error(e)
            logger.error('FAILED to update indexes!!!')
            indexes = None
        try:
            tagger = self.load_tagger(self.tag_algos)
        except Exception as e:
            logger.error(e)
            logger.error('FAILED to update tagger!!!')
            tagger = None
#       with self.lock:
        try:
            self.lock.write_acquire()
            self.n_update += 1
            self.indexes = indexes if indexes is not None else self.indexes
            self.tagger  = tagger  if tagger  is not None else self.tagger
        finally:
            self.lock.write_release()
        logger.info('>>>>>>>> update success!')


    def load_all(self, algos):
        indexes = []
        for algo, configfile in algos:
            try:
                idxer = self.load_one(algo, configfile)
            except Exception as e:
                logger.error(e)
                logger.error('FAILED to load {0}, {1}'.format(algo, configfile))
                raise e
            indexes.append(idxer)
        return indexes


    def load_one(self, algo, configfile):
        idxer = getattr(indexer, algo)()
        idxer.readcfg(configfile)
        idxer.load()
        return idxer


    def handle_err(self, req, res, msg, cost, ret, exception):
        logger.error(exception)
        res['msg'], res['ret'], res['cost'] = msg, ret, cost
        self.log(req, res)


    def log(self, req, res=None):
        del req['tm_start']
        if res is not None:
            req['ret'], req['msg'], req['cost'] = res['ret'], res['msg'], res['cost']
        logger.info(req)


    def parse_params(self, req, non_default_keys, default_keys=[]):
        non_default_params = [req['params'][k] for k in non_default_keys]
        default_params = [req['params'].get(k, v) for k, v in default_keys]
        return non_default_params + default_params


    def search(self):
        res = {'cost': 0., 'data': [], 'msg': '', 'ret': -1}
        req = flask_utils.get_request_info()
        g.inner = {}

        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])

        try:
            non_default_keys, default_keys = ['img_url', 'topk', 'algo'], []
            img_url, topk, algo = self.parse_params(req, non_default_keys, default_keys)
            topk = int(topk)
            if algo not in ['vgg', 'hsv' , 'all']:
                raise Exception('wrong algo')
        except Exception as e:
            self.handle_err(req, res, 'failed to parse parameter', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])

        t1 = time.time()
        image = download_image(img_url, timeout=4, save=False)
        if not isinstance(image, np.ndarray):
            self.handle_err(req, res, 'failed to download image {0}'.format(img_url), \
                    time.time()-req['tm_start'], -1, Exception('download image error'))
            return flask_utils.make_response(res, flask_utils.headers[0])

        t2 = time.time()
        g.inner['tm_download'] = t2-t1

#       with self.lock:
        try:
            self.lock.read_acquire()
            for idxer in self.indexes:
                if algo != 'all' and algo != idxer.extor.args['name']:
                    continue

                try:
                    c1 = time.time()
                    vec = idxer.extor.extract(image)
                    c2 = time.time()
                    D, I = idxer.search(vec, topk)
                    c3 = time.time()
                    D = D[0].tolist()
                    info_ids, res_urls = idxer.get_imginfo(I)
                    res['data'].extend([{'info_id': info_id, 'res_url': res_url, 'dist': dist, 'algo': idxer.extor.args['name']}\
                            for info_id, res_url, dist in zip(info_ids, res_urls, D)])
                    c4 = time.time()
                    g.inner[idxer.extor.args['name']] = {'tm_extract': c2-c1, 'tm_search': c3-c2, 'tm_idx2infoid': c4-c3}
                except Exception as e:
                    logger.error(e)
                    continue
        finally:
            self.lock.read_release()

        t3 = time.time()
        g.inner['tm_total'] = t3-t1
        g.inner['n_thread'] = threading.active_count()
        
        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
        logger.info(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])


    def tagged(self):
        res = {'cost': 0., 'data': [], 'msg': '', 'ret': -1}
        req = flask_utils.get_request_info()
        g.inner = {}

        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])

        try:
            non_default_keys = ['img_url']
            img_url, = self.parse_params(req, non_default_keys)
            topk, n_tags = 20, 20
        except Exception as e:
            self.handle_err(req, res, 'failed to parse parameter', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])

        t1 = time.time()
        image = download_image(img_url, timeout=4, save=False)
        if not isinstance(image, np.ndarray):
            self.handle_err(req, res, 'failed to download image {0}'.format(img_url), \
                    time.time()-req['tm_start'], -1, Exception('download image error'))
            return flask_utils.make_response(res, flask_utils.headers[0])

        t2 = time.time()
        g.inner['tm_download'] = t2-t1
        
#       from keras.backend.tensorflow_backend import set_session, get_session
#       s = get_session()
#       print(s)
#       print(s.graph)
#       print(self.tagger.idxer.extor.feat_extractor.input.graph)
#       print(self.tagger.idxer.extor.feat_extractor.output.graph)
#       with self.lock:
        try:
            self.lock.read_acquire()
            try:
                vec = self.tagger.idxer.extor.extract(image)
                D,I = self.tagger.idxer.search(vec, topk)
                D = D[0].tolist()
                info_ids, _ = self.tagger.idxer.get_imginfo(I)

                res_tags = self.tagger.get_tags(info_ids)
                res['data'] = res_tags[:n_tags]
            except Exception as e:
                self.lock.read_release()  # attention 
                self.handle_err(req, res, 'something wrong', time.time()-req['tm_start'], -1, Exception('algo error'))
                return flask_utils.make_response(res, flask_utils.headers[0])
        finally:
            self.lock.read_release()

        t3 = time.time()
        g.inner['tm_total'] = t3-t1
        g.inner['n_thread'] = threading.active_count()

        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
        logger.info(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])


    def tagged_more(self):
        res = {'cost': 0., 'data':[], 'msg': '', 'ret': -1}
        req = flask_utils.get_request_info()
        g.inner = {}

        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])
        
        # check params
        try:
            non_default_keys = ['img_url']
            img_url, = self.parse_params(req, non_default_keys)
        except Exception as e:
            self.handle_err(req, res, 'failed to parse parameter', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])
        
        # core
        t1 = time.time()
        image = download_image(img_url, timeout=4, save=False)
        if not isinstance(image, np.ndarray):
            self.handle_err(req, res, 'failed to download image {0}'.format(img_url), \
                    time.time()-req['tm_start'], -1, Exception('download image error'))
            return flask_utils.make_response(res, flask_utils.headers[0])

        t2 = time.time()
        g.inner['tm_download'] = t2-t1

#       with self.lock:
        try:
            self.lock.read_acquire()
            try:
                vec = self.tagger.idxer.extor.extract(image)
                D,I = self.tagger.idxer.search(vec, 20)
                info_ids, _ = self.tagger.idxer.get_imginfo(I)
                res_tags = self.tagger.get_more_tags(info_ids)
                res['data'] = res_tags[:50]
            except Exception as e:
                self.lock.read_release() # attention
                self.handle_err(req, res, 'something wrong', time.time()-req['tm_start'], -1, e)
                return flask_utils.make_response(res, flask_utils.headers[0])
        finally:
            self.lock.read_release()

        t3 = time.time()
        g.inner['tm_total'] = t3-t1
        g.inner['n_thread'] = threading.active_count()

        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
        logger.info(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])

