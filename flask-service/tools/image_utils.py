# -*- coding: utf-8 -*-

import os
import time
import threading

import cv2
import requests
import numpy as np

from config import MYPATH, logger

def resize(image, maxsize=768):
    h, w, _ = image.shape
    maxedge = max(w, h)
    if maxedge > maxsize:
        ratio = maxsize/maxedge
        image = cv2.resize(image, (int(w*ratio), int(h*ratio)))
    return image



class DownloadImage(object):
    
    def __init__(self):
        self.image_cache = os.path.join(MYPATH.PROJECT_PATH, 'tmp/image.cache')
        self.image_path  = os.path.join(MYPATH.PROJECT_PATH, 'tmp/image')
        try:
            if not os.path.exists(self.image_cache):
                open(self.image_cache, 'w').close()
            if not os.path.exists(self.image_path):
                os.makedirs(self.image_path)
        
            with open(self.image_cache, 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip('\n').split('\t') for line in lines]
            self.image_dict = dict(lines)
            self.image_names = set([line[1] for line in lines])
        except Exception as e:
            self.image_dict = {}
            self.image_names = set()
            logger.error(e)


    def __call__(self, url, timeout=4, save=False, img_dict={}, idx=0):
        try:
            image = None
            if url in self.image_dict:
                try:
                    image = cv2.imread(os.path.join(self.image_path, self.image_dict[url]))
                except Exception as e:
                    image = None
            
            if isinstance(image, np.ndarray):
                img_dict[idx] = image
                return image
             
            res = requests.get(url, timeout=timeout)
            image = np.asarray(bytearray(res.content), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if save:
                try:
                    imgname = url.split('/')[-1]#str(time.time())+'.jpg'
                    imgpath = os.path.join(self.image_path, imgname)
                    if not os.path.exists(imgpath):  ## 如果启动多个进程或者多个线程会有问题，启动多个只能把save设为False
                        cv2.imwrite(imgpath, resize(image, maxsize=1000))
                        self.image_dict[url] = imgname
                        self.image_names.add(imgname)
                        with open(self.image_cache, 'a') as f:
                            f.write('\t'.join([url, imgname])+'\n')
                except Exception as e:
                    pass
            img_dict[idx] = image
            return image
        except Exception as e:
            logger.error(e)
            img_dict[idx] = None
            return None

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

#download_image = DownloadImage()
download_image = download

def download_multithread(urls):
    task_threads = [] 
    img_dict = {}
    for i, url in enumerate(urls):
        task = threading.Thread(target=download_image, args=(url, 4, True), kwargs={'img_dict':img_dict, 'idx': i})
        task_threads.append(task)
    for task in task_threads:
        task.start()
    for task in task_threads:
        task.join()

    imgs = [img_dict[i] for i, url in enumerate(urls)]
    return imgs 


def visualize(q_img, res_imgs, path=None):
    q_img = resize(q_img, 500)
    res_imgs = [resize(res_img, 500) for res_img in res_imgs]

    interval = 10
    h, w, c = q_img.shape
    max_h = h
    for res_img in res_imgs:
        max_h = max(max_h, res_img.shape[0])
    
    imgs = []
    h, w, c = q_img.shape
    imgs.append(np.concatenate([np.zeros((max_h-h, w, c), dtype=np.uint8), q_img], axis=0))
    imgs.append(np.zeros((max_h, interval, c), dtype=np.uint8))
    for j in range(len(res_imgs)):
        h, w, c = res_imgs[j].shape
        imgs.append(np.concatenate([np.zeros((max_h-h, w, c), dtype=np.uint8), res_imgs[j]], axis=0))
        imgs.append(np.zeros((max_h, interval, c), dtype=np.uint8))

    imgs = np.concatenate(imgs, axis=1)
    if path is not None:
        cv2.imwrite(path, imgs)
    else:
        return imgs
