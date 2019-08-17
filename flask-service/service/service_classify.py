# -*- coding: utf-8 -*-

import os
import time
import json
import threading

import yaml
import numpy as np
from flask import g

from config import logger, MYPATH
from tools import flask_utils, download_image, download_multithread
import classify

from .service_example import Service


class ClassifyService(Service):
    def __init__(self):
        self.name = 'classify'
        self.params = yaml.load(open(os.path.join(MYPATH.RESOURCE_PATH, 'config/params.yaml'), 'r'))[self.name]
        logger.info('=== classify params ===')
        logger.info(self.params)
        self.initialize_algo()


    def add_route(self, wrap_app):
        self.wrap_app = wrap_app
        flask_utils.add_route(self.wrap_app.app, mappings=[
            ('/classify', self.categorize_multi)
            ])


    def initialize_algo(self):
        classifier_name = self.params['classifier_name']
        classifier_cfg  = self.params['classifier_cfg']
        classifier = getattr(classify, classifier_name)()
        classifier.readcfg(classifier_cfg)
        self.classifier = classifier


    def categorize(self):
        res = {'cost': 0., 'data': [], 'msg': '', 'ret': -1}
        req = flask_utils.get_request_info()
        g.inner = {}

        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])
        
        try:
            non_default_keys, default_keys = ['img_url'], [('topk', 1)]
            img_url, topk = self.parse_params(req, non_default_keys, default_keys)
            topk = int(topk)
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
        g.inner['tm_download'] = t2 - t1

        try:
            res['data'] = [{'category': None, 'score': None}  for i in range(topk)]
            classify_res =  self.classifier.online_categorize(image)[0]
            for i, item in enumerate(res['data']):
                res['data'][i]['category'] = classify_res[i][0]
                res['data'][i]['score']    = round(classify_res[i][1],2)
        except Exception as e:
            self.handle_err(req, res, 'something wrong', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])

        t3 = time.time()
        g.inner['tm_total'] = t3-t1
        g.inner['n_thread'] = threading.active_count()

        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
#       logger.info(g.inner)
        req.update(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])
    

    def categorize_multi(self):
        res = {'cost': 0., 'data': [], 'msg': '', 'ret': -1}
        req = flask_utils.get_request_info()
        g.inner = {}

        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])
            
        try:
            non_default_keys, default_keys = ['img_urls'], [('topk', 1)]
            img_urls, topk = self.parse_params(req, non_default_keys, default_keys)
            img_urls = img_urls.split(',')
#           img_urls = json.loads(img_urls)
            img_urls = list(set(img_urls))  # 注意去重，不要放在后面去重
            topk = int(topk)
        except Exception as e:
            self.handle_err(req, res, 'failed to parse parameter', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])


        # core
        res['data'] = dict(zip(img_urls, \
                               [{'ret': -1, \
                                 'body': [(None, None) for i in range(topk)]} for img_url in img_urls] ))
        t1 = time.time()
        images = download_multithread(img_urls)
        t2 = time.time()
        g.inner['tm_download'] = t2-t1
        
#       import pdb; pdb.set_trace()
        try:
            preproc_imgs, valid_urls = [], []
            for img_url, image in zip(img_urls, images):
                if not isinstance(image, np.ndarray):
                    continue
                image = self.classifier.resize(image)
                image = self.classifier.preprocess(image)
                preproc_imgs.append(image)
                valid_urls.append(img_url)
            preproc_imgs = np.concatenate(preproc_imgs, axis=0)
            classify_reses = self.classifier.categorize(preproc_imgs)

            for valid_url, classify_res in zip(valid_urls, classify_reses):
                for i, item in enumerate(res['data'][valid_url]['body']):
                    res['data'][valid_url]['body'][i] = classify_res[i][0], round(classify_res[i][1], 2)
                res['data'][valid_url]['ret'] = 0
        except Exception as e:
            self.handle_err(req, res, 'something wrong', time.time()-req['tm-start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])

        t3 = time.time()
        g.inner['tm_total'] = t3 - t1
        g.inner['n_thread'] = threading.active_count()

        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
#       logger.info(g.inner)
        req.update(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])

