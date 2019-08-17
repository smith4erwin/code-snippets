# -*- coding: utf-8 -*-

import math
import threading


class MultiWork(object):
    def __init__(self):
        pass


    def __call__(self, n_thread, proc_func, data, results):
        n_data = len(data)
        n_loop = math.ceil(n_data/n_thread)
        for i in range(n_loop):
            s = i * n_thread
            e = s + n_thread
            e = e if e < n_data else n_data
            tasks = []
            for j in range(s, e):
                task = threading.Thread(target=proc_func, args=(data[j], ), kwargs={'results':results, 'idx': j})
                tasks.append(task)
            for task in tasks:
                task.start()
            for task in tasks:
                task.join()
        return results
