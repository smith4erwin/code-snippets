# -*- coding: utf-8 -*-

import json
import time
from flask import Flask, request


headers = [{'Content-Type': 'application/json; charset=utf-8'},\
           {}]


def app(name):
    return Flask(name)


def add_route(app, mappings=None, methods=None):
    if mappings is not None:
        methods = ('GET', 'POST') if methods is None else methods
        for mapping in mappings:
            rule, view_func = mapping
            app.add_url_rule(rule, view_func=view_func, methods=methods)


def parse_parameter(param_names):
    param = []
    values = request.values.to_dict()
    for name in param_names:
        param.append(values[name])
    return param
        

def view(item):
    return json.dumps(item, ensure_ascii=False)


def make_response(content, header):
    return view(content), header


def get_ip():
    ip = request.headers.get('X-Forwarded-For') or \
         request.headers.get('X-Real-IP') or \
         request.remote_addr or \
         request.environ.get('REMOTE_ADDR')
    return ip


def get_request_info():
    req = {}
    req['client_ip'] = get_ip()
    req['path'] = request.path
    req['method'] = request.method
    req['params'] = request.values.to_dict()
    req['tm_start'] = time.time()
    return req
