# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tools import download_image

class DownloadImageTest(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        pass


    def tearDown(self):
        print('tearDown...')
        pass


    def test_downloadimage(self):
        img_url = 'http://recc.nosdn.127.net/irec-20181017-eb0efd43e826276adbd10b7f1038549d.jpg' 
        img_url = 'http://recc.nosdn.127.net/irec-20181023-91f5af7724044653479b4afdb2f5963d.png'
        img_url = 'http://recc.nosdn.127.net/irec-20181017-ee7b21d24cfd282a679856afcc023831.jpg'
        image = download_image(img_url, save=True)
        print(type(image))
        if isinstance(image, np.ndarray):
            print(image.shape)
#       dwld = sql.Dwld()
#       dwld.run()
        pass





