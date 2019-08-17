# -*- coding: utf-8 -*-

import os

import cv2
import requests
import numpy as np

from .multiwork import MultiWork
from config import *


def resize(image, maxsize=768):
    h, w, _ = image.shape
    maxedge = max(w, h)
    if maxedge > maxsize:
        ratio = maxsize/maxedge
        image = cv2.resize(image, (int(w*ratio), int(h*ratio)))
    return image


def download(url, timeout=4, save=None, img_dict={}, idx=0):
    try:
#       print(idx, url)
        res = requests.get(url, timeout=timeout)
        image = np.asarray(bytearray(res.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img_dict[idx] = image
        return image
    except Exception as e:
        logger.error(e)
        img_dict[idx] = None
        return None


class MultiDownload(object):
    def __init__(self):
        pass

    def download_multithread(self, n_thread, urls, imgnames, path=None):
        self.urls = urls
        self.imgnames = imgnames
        self.path = path
        results = {}
        multiworker = MultiWork()
        results = multiworker(n_thread, self.download_one, list(zip(self.urls, self.imgnames)), results)
        return results
    
    def download_one(self, url_imgname, results={}, idx=-1):
        url, imgname = url_imgname
        image = download(url)
        if image is None:
            print(url)
            if self.path is None:
                results[idx] = image
            return image
        image = resize(image, 768)

        if self.path is not None:
#           imgname = url.split('/')[-1]
#           if imgname[-4:] != '.jpg':
#               imgname = imgname + '.jpg'
            fullname = os.path.join(self.path, imgname)
            if os.path.exists(fullname):
                return image
            cv2.imwrite(fullname, image)
        else:
            results[idx] = image
        return image




