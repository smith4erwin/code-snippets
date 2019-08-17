# -*- coding: utf-8 -*-

import os
import json
import unittest
import requests

from config import MYPATH

class AppTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')


    def tearDown(self):
        print('tearDown...')


    def test_app(self):
        if 0:
            req_url = 'http://localhost:16000'
#           url = 'http://hzabj-rec25.server.163.org:9191'
#           url = 'http://hzabj-rec24.server.163.org:9191'
#           url = 'http://commonrec.service.163.org/imgsearch'
#           img_url = 'http://nbot-pub.nosdn.127.net/4fb81d368b429d5524ca8d10783fa383'
#           img_url = 'http://nos.netease.com/irec/icrop-20180427-a026fe1fc2234114c0f327ac8095ca7d.jpg'
#           img_url = 'http://recc.nosdn.127.net/irec-20181028-c9eedac6b0e4140f7afe2ebe4962a09d.jpg'
#           img_url='http://nos.netease.com/irec/icrop-20180428-d66d22ab4b7901ef54e19681133204f7.jpg'
#           topk=5
#           algo='all'
#           data = {'img_url': img_url, 'topk': topk, 'algo':algo} 
#   
#           res = requests.get(url+'/search', data=data)
#           res = requests.get(url+'/search'+\
#                   '?img_url=http://nos.netease.com/irec/icrop-20180428-d66d22ab4b7901ef54e19681133204f7.jpg&topk=5&algo=all')
#           res = requests.get(url+'/tagged'+\
#                   '?img_url=http://nos.netease.com/irec/icrop-20180428-d66d22ab4b7901ef54e19681133204f7.jpg&topk=5&algo=all')

#           print(json.loads(res.content))
#           import pdb; pdb.set_trace()
            path = '/tagged_more'
            with open(os.path.join(MYPATH.PROJECT_PATH, 'tmp/test_urls.txt'), 'r') as f:
                img_urls = [line.rstrip('\n') for line in f.readlines()]

            f = open(os.path.join(MYPATH.PROJECT_PATH, 'tmp/results_tagged_more_181211_2015.txt'), 'a')
            for img_url in img_urls:
                try:
                    res = json.loads(requests.get(req_url+path+'?img_url='+img_url, timeout=100).content)
                    if res['ret'] == 0:
#                      res = json.loads(res)
                        res_tags = res['data']
                        f.write('\t'.join([img_url, str(res_tags)])+'\n')
                except Exception as e:
                    print(e)
            f.close()


        if 0:
#           req_url = 'http://localhost:16000'
            req_url = 'http://hzabj-rec24.server.163.org:9191'
            req_url = 'http://hzabj-rec25.server.163.org:9191'
            path = '/tagged_more'
            img_url = 'http://nos.netease.com/irec/icrop-20180424-154ff8b560fe21e278224183d194f7b3.jpg'
            print(json.loads(requests.get(req_url+path+'?img_url='+img_url).content))

        if 0:
#           req_url = 'http://localhost:16000'
            req_url = 'http://hzabj-rec24.server.163.org:9191'
            req_url = 'http://hzabj-rec25.server.163.org:9191'
            path = '/search'
            img_url = 'http://nbot-pub.nosdn.127.net/291f2b6c86f4cb2f64aa5433a857db18.jpg'
            img_url = 'http://nbot-pub.nosdn.127.net/ba6771c622b5719e0d74604aa210cdbc.jpg'
            img_url = 'http://nos.netease.com/reci/irec-20180316-30328c7600f879881308d9c2252665a4.jpg'
            print(json.loads(requests.get(req_url+path+'?img_url='+img_url+'&topk=20&algo=vgg').content))

        if 1:
            req_url = 'http://localhost:9191'
#           req_url = 'http://localhost:16000'
#           req_url = 'http://hzabj-rec24.server.163.org:9191'
#           req_url = 'http://hzabj-rec25.server.163.org:9191'
            path = '/classify'
            img_url = 'http://nbot-pub.nosdn.127.net/291f2b6c86f4cb2f64aa5433a857db18.jpg'
            img_url = 'http://nbot-pub.nosdn.127.net/ba6771c622b5719e0d74604aa210cdbc.jpg'
            img_url = 'http://nos.netease.com/reci/irec-20180316-30328c7600f879881308d9c2252665a4.jpg'
            print(json.loads(requests.get(req_url+path+'?img_url='+img_url).content))
#           print(json.loads(requests.get(req_url+path+'?img_url='+img_url).content))


