# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def resize(img, maxsize=512):
    h, w, _ = img.shape
    maxedge = max(h, w)
    ratio = maxsize/maxedge
    if maxedge > maxsize:
        resize_img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
    else:
        resize_img = img
    return resize_img


sift = cv2.xfeatures2d.SIFT_create()
def compute_sift(img=None, path=None, saved_results={}):
    if path is not None:
        img = cv2.imread(path)
    resize_img = resize(img, 768)
    kpts = sift.detect(resize_img)
    kpts, des = sift.compute(resize_img, kpts)
    if len(kpts) == 0:
        des = np.zeros((0, 128), dtype=np.float32)
    if path is not None:
        saved_results[path] = (kpts, des)
    return kpts, des


def train_codebook(all_features, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kmeans.fit(all_features)
    return kmeans.cluster_centers_


def compute_vlad(centers, features):
    if features.shape[0] <= centers.shape[0]:
        return np.ones_like(centers).reshape(-1).astype(np.float32) * 10
    
    dists = euclidean_distances(features, centers)
#     dists_argsort = np.argsort(dists, axis=1)
#     nearest_centers = centers[dists_argsort[:, 0], :]
    mindist_index = np.argmin(dists, axis=1)
    nearest_centers = centers[mindist_index, :]
    residual = features - nearest_centers
    vlad_features = np.zeros((centers.shape[0], centers.shape[1]))
#     for idx, resi in zip(dists_argsort[:, 0], residual):
    for idx, resi in zip(mindist_index, residual):
        vlad_features[idx] = vlad_features[idx] + resi
    vlad_features = vlad_features / (np.sqrt(np.sum(np.power(vlad_features, 2), axis=1, keepdims=True))+1e-8)
    vlad_features = vlad_features.reshape(-1).astype(np.float32)
    return vlad_features


def visualize(q_img, res_imgs, path):
    q_img = resize(q_img, 500)
    res_imgs = [resize(res_img, 500) for res_img in res_imgs]
    
    interval = 10
    h, w, c = q_img.shape
    max_h = h
    for j in range(len(res_imgs)):
        max_h = max(max_h, res_imgs[j].shape[0])
    
    imgs = []
    h, w, c = q_img.shape
    imgs.append(
        np.concatenate(
            [np.zeros((max_h-h, w, c), dtype=np.uint8), q_img], axis=0
        )        
    )
    imgs.append(np.zeros((max_h, interval, c), dtype=np.uint8))
    for j in range(len(res_imgs)):
        h, w, c = res_imgs[j].shape
        imgs.append(
            np.concatenate(
                [np.zeros((max_h-h, w, c), dtype=np.uint8), res_imgs[j]], axis=0
            )
        )
        imgs.append(np.zeros((max_h, interval, c), dtype=np.uint8))

    imgs = np.concatenate(imgs, axis=1)
    cv2.imwrite(path, imgs)



def get_num_good_matches_by_flann(matcher, query_des, saved_results={}, idx=-1):
    dmatches = matcher.knnMatch(query_des, k=2)
    dists = np.array([(dm1.distance, dm2.distance) for dm1, dm2 in dmatches])
    good_matches_mask = dists[:, 0] < 0.7 * dists[:, 1]
    n_good_matches = good_matches_mask.sum()
    saved_results[idx] = (n_good_matches, good_matches_mask)
    return n_good_matches, good_matches_mask


def get_num_good_matches_by_mybf(mybfmatcher, train_des, saved_results={}, idx=-1):
#     t1 = time.time()
    dists = -2*np.dot(mybfmatcher.query_des, train_des.T) + np.square(train_des).sum(axis=1) + mybfmatcher.query_des_squ
#     t2 = time.time()
    rg = np.arange(dists.shape[0])
#     dists = euclidean_distances(query_des, mybfmatcher.train_des)
#     dists_argsort = np.argsort(dists, axis=1)
#     dist1 = dists[rg, dists_argsort[:, 0]]
#     dist2 = dists[rg, dists_argsort[:, 1]]
#     t3 = time.time()
    dist11, dist11_index = np.min(dists, axis=1), np.argmin(dists, axis=1)
    dists[rg, dist11_index] = np.inf
    dist21 = np.min(dists, axis=1)
#     t4 = time.time()
#     assert (dist11 == dist1).sum() == dist1.shape[0]
#     assert (dist21 == dist2).sum() == dist2.shape[0]
#     good_matches_mask = dist1 < 0.49 * dist2
    good_matches_mask = dist11 < 0.49 * dist21
    n_good_matches = good_matches_mask.sum()
#     print(t2-t1, t3-t2, t4-t3)
    saved_results[idx] = (n_good_matches, good_matches_mask)
    return n_good_matches, good_matches_mask


class MyBFMatcher(object):
    def __init__(self, query_des):
        self.query_des = query_des
        self.query_des_squ = np.square(query_des).sum(axis=1, keepdims=True)
