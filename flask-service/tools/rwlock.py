# -*- coding: utf-8 -*-

import time
import threading

class RWlockReadPreferring(object):
    def __init__(self):
        self.lockr = threading.Lock()
        self.lockg = threading.Lock()
        self.n_reader = 0

    
    def read_acquire(self):
        with self.lockr:
            self.n_reader += 1
            if self.n_reader == 1:
                self.lockg.acquire()
#           print("reader notify: acquire, n_reader = ", self.n_reader)
    

    def read_release(self):
        with self.lockr:
            self.n_reader -= 1
            if self.n_reader == 0:
                self.lockg.release()
#           print("reader notify: release, n_reader = ", self.n_reader)
    

    def write_acquire(self):
        self.lockg.acquire()
#       print("writer notify: acquire")
    

    def write_release(self):
        self.lockg.release()
#       print("writer notify: release")

