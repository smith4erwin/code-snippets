# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.signal
import skimage.measure

class Vec(object):

    def __init__(self, vecdir, vecnames):
        self.vecdir   = vecdir
        self.vecnames = vecnames
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.vecnames):
            fullpath = os.path.join(self.vecdir, self.vecnames[self.i])
            vec = np.load(fullpath)
            vec = vec.astype(np.float32)
            self.i += 1
            return vec
        else:
            raise StopIteration



def quantize_img(img, binss, sep_or_joint='sep'):
    if sep_or_joint == 'sep':
#       assert img.ndim == 2
        quan_img = np.zeros_like(img, dtype=np.uint)
        for i in range(len(binss)-1):
            quan_img[(img >= binss[i]) & (img < binss[i+1])] = i+1
        return quan_img

    if sep_or_joint == 'joint':
#       assert img.ndim == 3 and 3 == len(binss)
        quan_img = np.zeros_like(img, dtype=np.uint)
        for i in range(3):
            quan_img[:,:,i] = quantize_img(img[:,:,i], binss[i], 'sep') - 1
        quan_img = quan_img[:,:,0] * (len(binss[1])-1) * (len(binss[2])-1) + \
                   quan_img[:,:,1] * (len(binss[2])-1) + \
                   quan_img[:,:,2] + 1
        return quan_img


def ccv_core(quan_img, thres, n_bin):
#   color coherent vector
#   assert set(np.unique(quan_img)) == set(range(1,n_bin+1))
    quan_conncomp = skimage.measure.label(quan_img, connectivity=2)
#   quan_conn_lables, quan_conn_cnts = np.unique(quan_conncomp, return_counts=True)
    feats = []
    for i in range(1, n_bin+1):
        i_lbles, i_cnts = np.unique(quan_conncomp[quan_img == i], return_counts=True)
        incoherent = i_cnts[i_cnts < thres].sum()
        coherent   = i_cnts[i_cnts >= thres].sum()
        feats.append((coherent, incoherent))
    return feats


def compute_edge_orientation(img):
    x_sobel = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]], dtype=np.int32)
    y_sobel = x_sobel.T

    l_gx = scipy.signal.convolve2d(img[:,:,0], x_sobel, mode='same', boundary='symm')
    a_gx = scipy.signal.convolve2d(img[:,:,1], x_sobel, mode='same', boundary='symm')
    b_gx = scipy.signal.convolve2d(img[:,:,2], x_sobel, mode='same', boundary='symm')

    l_gy = scipy.signal.convolve2d(img[:,:,0], y_sobel, mode='same', boundary='symm')
    a_gy = scipy.signal.convolve2d(img[:,:,1], y_sobel, mode='same', boundary='symm')
    b_gy = scipy.signal.convolve2d(img[:,:,2], y_sobel, mode='same', boundary='symm')

    gxx = np.power(l_gx, 2) + np.power(a_gx, 2) + np.power(b_gx, 2)
    gyy = np.power(l_gy, 2) + np.power(a_gy, 2) + np.power(b_gy, 2)
    gxy = l_gx * l_gy + a_gx * a_gy + b_gx * b_gy

    phi = np.arctan(2*gxy/(gxx-gyy+1e-6)) / 2             # [-pi/4, pi/4]
    b = phi < 0
    phi[b] = phi[b] + np.pi / 2      # [0, pi/2]

    cos_2theta = np.cos(2*phi)
    sin_2theta = np.sin(2*phi)

    gxx_plus_gyy = gxx+gyy
    gxx_sub_gyy_mul_cos2theta = (gxx-gyy)*cos_2theta
    gxy_mul_2sin2theta = 2*gxy*sin_2theta

    G1 = np.sqrt( ( gxx_plus_gyy + gxx_sub_gyy_mul_cos2theta + gxy_mul_2sin2theta ) / 2 + 1e-6 )
    G2 = np.sqrt( ( gxx_plus_gyy - gxx_sub_gyy_mul_cos2theta - gxy_mul_2sin2theta ) / 2 + 1e-6 )

    b = (G1 < G2)
    phi[b] = phi[b] + np.pi/2
#   phi = (phi + np.pi/4)*2
    phi = phi * 2
    return phi













