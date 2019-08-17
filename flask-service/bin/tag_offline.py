# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse

import tag
from config import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', action='store', dest='algo')
    parser.add_argument('-c', '--cfg',  action='store', dest='cfg')

    args = parser.parse_args()

    logger.info('\n===== read tagger config file =====')
    tagger = getattr(tag, args.algo)()
    tagger.readcfg(args.cfg)

    assert tagger.args['tag_algo'] == args.algo
    assert os.path.realpath(tagger.args['tagdir']) == os.path.realpath(os.path.dirname(args.cfg))
    t1 = time.time()    
    tagger.tagged_offline()
    t2 = time.time()
    logger.info("===== total tag time: {0}\n".format(t2-t1))



if __name__ == '__main__':
    main()
