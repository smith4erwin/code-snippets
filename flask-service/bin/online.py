# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse

import cv2
import json
import requests
import numpy as np


from config import logger, MYPATH


def test(port):
#   with open(os.path.join(PROJECT_PATH, 'conf/dbfile', 'query.txt'), 'r') as f:
#       lines = f.readlines()
#       infoids, urls = list(zip(*[line.split('\t')[:2] for line in lines]))
#   print(len(urls))
#   thres = 10
#   topk  = 20
#   req_url = 'http://localhost:'+port+'/search'

#   visudir = time.strftime("%y%m%d-%H%M", time.localtime())
#   fullpath = os.path.join(PROJECT_PATH, 'visualize', visudir)
#   print("result path:", fullpath)
#   mkdr(fullpath)

#   for i, (url,q_infoid) in enumerate(zip(urls, infoids)):
#       print(url, q_infoid)
#       data = {'url':url, 'thres':thres, 'topk':topk}
#       res = requests.post(req_url, data=data)
#       res = json.loads(res.content)
#      # print(res)

#       if 'urls' in res:
#       if len(res['urls']) != 0:
#           print(res['urls'][0])
#           print('\n')
#       else:
#           print(res)
#           print('\n')
#           continue

#       q_img = download_single_image(url, resize, (500, ))
#       if q_img is None:
#           continue

#       r_imgs = []
#       for infoid in res['info_ids']:
#           img = cv2.imread(os.path.join(PROJECT_PATH, 'images_sql', infoid+'.jpg'))
#           img = resize(img, 500)
#           r_imgs.append(img)

#       
#       write_path=os.path.join(fullpath, q_infoid+'.jpg')
#       visualize(q_img, r_imgs, write_path)


        #assert res['urls'][0] == url
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--port', action='store', dest='port', default='16000')
    parser.add_argument('-a', '--algo', action='store', dest='algo')
    parser.add_argument('-c', '--cfg',  action='store', dest='cfg')
    args = parser.parse_args()
    print(args)
    
    test(args.port)



