# -*- coding: utf-8 -*-

import os
import re
import logging
import logging.config

import tensorflow as tf


# PATH
env = 'dev'
#env = 'online'
class MYPATH(object):
    APP_PATH      = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '..'))
    PROJECT_PATH  = os.path.abspath(os.path.join(APP_PATH, '..'))
    DEPLOY_PATH   = '/home/commonrec/deploy/inforec/img_search_srv/9191'
    RESOURCE_PATH = os.path.join(PROJECT_PATH, 'resource', env)
    DATA_PATH     = os.path.join(PROJECT_PATH, 'data', env)


# logging
logging.config.fileConfig(os.path.join(MYPATH.RESOURCE_PATH, 'config', 'logger.conf'))
logger = logging.getLogger('root')


# gpu
def gpu():
    logger.info('===== config gpu =====')
    try:
        gpu_info = os.popen('gpustat')
        gpu_info = gpu_info.readlines()[1:]
        for info in gpu_info:
            logger.info(info.strip())
        gpu_ids = [re.match('\[(\d)\]', i).group(1) for i in gpu_info ]
        used, total, avail = [], [], []
        for info in gpu_info:
            u, t = info.split('|')[-2].split('/')
            u = int(u.strip())
            t = int(t.split('MB')[0].strip())
            a = t-u
            used.append(u)
            total.append(t)
            avail.append(a)
    
        chosen_gpu = sorted(zip(avail, gpu_ids), key=lambda x:x[0], reverse=True)[0][1]
        logger.info('choose gpu: {0}\n'.format(chosen_gpu))
    except Exception as e:
        chosen_gpu = ''
    return chosen_gpu

chosen_gpu = gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = chosen_gpu

gpu_config = tf.ConfigProto(allow_soft_placement=True)#log_device_placement=True)
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.1



