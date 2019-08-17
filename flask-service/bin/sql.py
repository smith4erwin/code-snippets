# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import logging
import argparse
import threading

import cv2
import yaml
import pymysql
import requests
import numpy as np

from config import *
from tools import download_image, resize

dbfile  = os.path.abspath(os.path.join(MYPATH.RESOURCE_PATH, 'dbfile/181031-162300.txt'))
imgpath = os.path.abspath(os.path.join(MYPATH.PROJECT_PATH, '../framework_v0.1/images_sql'))
dwld_n_thread = 100


class Dwld(object):

    def __init__(self, dbfile=dbfile, imgpath=imgpath, n_thread=100):
        filepath, filename  = os.path.split(dbfile)
        self.dbfile_problem = os.path.join(filepath, filename+'.problem')
        self.dbfile   = dbfile
        self.n_thread = n_thread
        self.imgpath  = imgpath

        if not os.path.exists(self.dbfile):
            try:
                os.makedirs(filepath)
            except:
                pass
            open(self.dbfile, 'w').close()
            open(self.dbfile_problem, 'w').close()


    def run(self):
        logger.info('=============== download images ===============')
        try:
            rows = self.fetch()
            assert rows
        except:
            logger.error('=== fetch sql error ===')
            return
        infoid = [row[0] for row in rows]
        assert len(set(infoid)) == len(infoid)
        
        with open(self.dbfile, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split('\t') for line in lines]
        with open(self.dbfile_problem, 'r') as f:
            problem_lines = f.readlines()
        raw_infoid = set([line[0] for line in lines])
        assert len(raw_infoid) == len(lines)
        logger.info('there are {0} images in database'.format(len(raw_infoid)))

        update_rows = [row for row in rows if row[0] not in raw_infoid]
#       if len(update_rows) < 3000:
#           logger.info('there are just {0} updated images, too few!'.format(len(update_rows)))
#           return
        update_rows = update_rows[:]
        noproblem, problem = self.download(update_rows)

        # append dbfile and dbfile_problem
        logger.info('===== success downloading {0} images'.format(len(noproblem)))
        logger.info('===== error downloading {0} images'.format(len(problem)))
        with open(self.dbfile, 'a') as f:
            for item in noproblem:
                f.write(item)
        with open(self.dbfile_problem, 'a') as f:
            for item in problem:
                if item not in problem_lines:
                    f.write(item)
        logger.info('=============== download images completed! ===============\n')


    def fetch(self):
        rows = None
        st = time.time()
#       conn = pymysql.connect(host='10.172.16.251', port=3306, user='inforec', password='t1VMkZLU', db='inforec')
        conn = pymysql.connect(host='inforec-online-mirr.dba.163.org', port=3306, user='inforec', password='t1VMkZLU', db='inforec')
        try:
            with conn.cursor() as cur:
                sql  = 'select info_id, pic_url, tag, category, source from cont_pool_picture where verify_status > 0;'
                sql1 = 'select info_id, pic_url from cont_pool_picture where verify_status > 0;'
                sql2 = 'select info_id, pic_url, tag, category, source from cont_pool_picture;'
                cur.execute(sql2)
                rows = cur.fetchall()
        except Exception as e:
            logger.error(e)
            rows = None
        finally:
            conn.close()
        et = time.time()
        logger.info('===== fetch sql costs: {0}'.format(et-st))
        return rows


    def check(self, **kargs):
        pass


    def download(self, rows):
        n_rows = len(rows)
        logger.info('======== start download {0} images ======='.format(n_rows))

        st = time.time()
        noproblem, problem = [], []
        n_loop = math.ceil(n_rows/self.n_thread)
        for i in range(n_loop):
            s = i * self.n_thread
            e = s + self.n_thread
            if e > n_rows: e = n_rows

            task_threads = []
            logger.info('\tloop {0}: download {1}-{2} images'.format(i, s, e))
            for j in range(s, e):
                row = [attr if attr else '' for attr in rows[j]]
                infoid, pic_url, tag, category, source = row
                problem_item = '\t'.join(row)+'\n'
                try:
                    pic_url = yaml.load(pic_url)[0]['url']
                except:
                    logger.info('some wrong happens for analysis pic_url {0}'.format(pic_url))
                    problem.append(problem_item)
                    continue
                noproblem_item = '\t'.join([infoid, pic_url, tag, category, source])+'\n'
                
                if os.path.exists(os.path.join(self.imgpath, infoid+'.jpg')):
                    noproblem.append(noproblem_item)
                    continue

                task = threading.Thread(target=self.download_single_image, \
                                        args=(pic_url, infoid+'.jpg', noproblem, noproblem_item, problem, problem_item))
                task.setDaemon(True)
                task_threads.append(task)
            for task in task_threads:
                task.start()
            for task in task_threads:
                task.join()
        
        et = time.time()
        logger.info('======= download {0} images costs: {1} ========'.format(n_rows, et-st))
        return noproblem, problem


    def download_single_image(self, pic_url, imgname, noproblem, noproblem_item, problem, problem_item):
        try:
            image = download_image(pic_url, timeout=4)
            image = resize(image, maxsize=768)
            cv2.imwrite(os.path.join(self.imgpath, imgname), image)
            noproblem.append(noproblem_item)
        except Exception as e:
            logger.error(e)
            problem.append(problem_item)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dbfile', action='store', dest='dbfile')
    parser.add_argument('-p', '--path',   action='store', dest='imgpath')
    parser.add_argument('-n', action='store', dest='dwld_n_thread', type=int)

    args = parser.parse_args()
    if args.dbfile:
        dbfile = args.dbfile
    if args.imgpath:
        imgpath = args.imgpath
    if args.dwld_n_thread:
        dwld_n_thread = args.dwld_n_thread

    downloader = Dwld(dbfile=dbfile, imgpath=imgpath, n_thread=dwld_n_thread)
    downloader.run()

