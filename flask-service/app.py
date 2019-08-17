# -*- coding: utf-8 -*-

import os
import argparse

from config import logger
from tools import flask_utils
from service import Service, ClassifyService

os.environ['FLASK_ENV'] = 'production'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class App(object):
    WHITE_IP_LIST = ["127.0.0.1"]
    FLAG   = True
    STATUS = True

    def __init__(self, port):
        self.port = port
        self.services = [Service(), ClassifyService()]
        
        self.app = flask_utils.app(__name__)
        flask_utils.add_route(self.app, mappings=[
            ('/health/status',   self.health_status),
            ('/health/offline',  self.health_offline),
            ('/health/activate', self.health_activate),
            ('/', self.welcome),
            ])
        for srv in self.services:
            srv.add_route(self)


    def health_status(self):
        return flask_utils.make_response(self.STATUS, flask_utils.headers[0])


    def health_offline(self):
        if flask_utils.get_ip() in self.WHITE_IP_LIST or self.FLAG:
            self.STATUS = False
        return flask_utils.make_response(self.STATUS, flask_utils.headers[0])


    def health_activate(self):
        if flask_utils.get_ip() in self.WHITE_IP_LIST or self.FLAG:
            self.STATUS = True
        return flask_utils.make_response(self.STATUS, flask_utils.headers[0])


    def welcome(self):
        content = 'welcome!'
        return flask_utils.make_response(content, flask_utils.headers[0])


    def start(self):
        self.app.run(host='0.0.0.0', use_reloader=False, port=self.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--port', action='store', dest='port', default='16000')
    args = parser.parse_args()

    app = App(args.port)
    app.start()

