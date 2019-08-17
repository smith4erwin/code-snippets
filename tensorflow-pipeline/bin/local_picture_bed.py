# -*- coding: utf-8 -*-

import os
import time
import threading

from flask import Flask
from flask import request, g, send_file

from config import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return 'welcome'

@app.route('/getimg', methods=['GET', 'POST'])
def readimg():
    # print(request.values.to_dict())
    g.imgname = request.values.to_dict()['imgname']
#   return send_file(os.path.join(RESOURCE_PATH, 'dataset/cont_pool_picture/images_sql', g.imgname), 'image/jpg')
    return send_file(os.path.join('/home/commonrec/llh/cbir/logo_recognition/resource/dataset/all_logo/JPEGImages', g.imgname), 'image/jpg')
#   return send_file(os.path.join('/home/commonrec/llh/cbir/logo_recognition/tmp/images', g.imgname), 'image/jpg')

if __name__ == '__main__':
    # print("init thread num: ", threading.active_count())
    app.run(host='0.0.0.0', use_reloader=False, port='5000')
