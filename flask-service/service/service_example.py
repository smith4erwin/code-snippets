# -*- coding: utf-8 -*-

import os
import time
import threading

from flask import g

from config import logger
from tools import flask_utils
#import algo


class Service(object):
    def __init__(self):
        self.initialize_algo()


    def add_route(self, wrap_app):
        self.wrap_app = wrap_app
        flask_utils.add_route(self.wrap_app.app, mappings=[
            ('/service_example', self.route_func), 
            ])


    def initialize_algo(self):
        pass


    def handle_err(self, req, res, msg, cost, ret, exception):
        logger.error(exception)
        res['msg'], res['ret'], res['cost'] = msg, ret, cost
        self.log(req, res)


    def log(self, req, res=None):
        del req['tm_start']
        if res is not None:
            req['ret'], req['msg'], req['cost'], req['data'] = res['ret'], res['msg'], res['cost'], res['data']
        logger.info(req)

    
    def parse_params(self, req, non_default_keys, default_keys=[]):
        non_default_params = [req['params'][k]       for k in non_default_keys]
        default_params     = [req['params'].get(k, v) for k, v in default_keys]
        return non_default_params + default_params


    def route_func(self):
        res = {'data': [], 'cost': 0., 'msg': '', 'ret':-1}
        req = flask_utils.get_request_info()
        g.inner = {}
        
        # check status
        if not self.wrap_app.STATUS:
            self.handle_err(req, res, 'service offline', time.time()-req['tm_start'], -1, Exception('service offline'))
            return flask_utils.make_response(res, flask_utils.headers[0])
        
        # check params
        def check_params(*params): return True 
        try:
            non_default_keys, default_keys  = ['param1', 'param2'], [('default_param1', None)]
            param1, param2, default_param1 = self.parse_params(req, non_default_keys, default_keys)
            if not check_params(param1, param2, default_param1):
                raise Exception('wrong params')
        except Exception as e:
            self.handle_err(req, res, 'failed to parse parameter', time.time()-req['tm_start'], -1, e)
            return flask_utils.make_response(res, flask_utils.headers[0])
        
        # core
        t1 = time.time()
        # process1
        t2 = time.time()
        # process2
        t3 = time.time()
        g.inner['tm_process1'] = t2-t1
        g.inner['tm_process2'] = t3-t2

        res['cost'], res['ret'], res['msg'] = t3-t1, 0, 'success'
        logger.info(g.inner)
        self.log(req, res)
        return flask_utils.make_response(res, flask_utils.headers[0])
