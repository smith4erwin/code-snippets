# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse

import extractor
import indexer
from config import logger, gpu_config

gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.99


def main():
    parser = argparse.ArgumentParser(description='Extract feature and Build index')
    parser.add_argument('-f', action='store', dest='flag', type=int)
    parser.add_argument('-a', '--algo', action='store', dest='algo')
    parser.add_argument('-c', '--cfg',  action='store', dest='cfg')
    parser.add_argument('-m', action='store', dest='n_thread', type=int, default=5)
    
    args = parser.parse_args()
    
    if args.flag == 1:
        logger.info('\n===== read extractor config file =====')
        extor = getattr(extractor, args.algo)()
        extor.readcfg(args.cfg)
        # 确任加载的确实是想要的ext
        assert extor.args['ext_algo'] == args.algo
        # 确认ext的配置文件中的'vecdir'就是配置文件所在目录
        # 有可能vecdir 并不是所在目录，那么特征就被存在别处了
        assert os.path.realpath(extor.args['vecdir']) == os.path.realpath(os.path.dirname(args.cfg))
        
        st = time.time()
        extor.extract_offline()
        et = time.time()
        logger.info("===== total extracting time: {0}\n".format(et-st))
    
    elif args.flag == 2:
        logger.info('\n===== read indexer config file =====')
        idxer = getattr(indexer, args.algo)()
        idxer.readcfg(args.cfg)
        assert idxer.args['idx_algo'] == args.algo
        assert os.path.realpath(idxer.args['idxdir']) == os.path.realpath(os.path.dirname(args.cfg))
 
        st = time.time()
        idxer.train()
        et = time.time()
        logger.info("===== total training time: {0}\n".format(et-st))



if __name__ == '__main__':
    main()

#python offline2.py -f 1/2  -a ext_algo/idx_algo -c cfgfile
