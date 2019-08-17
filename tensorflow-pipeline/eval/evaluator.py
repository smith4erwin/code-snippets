# -*- coding: utf-8 -*-

import os

import numpy as np

import load_data


class Evaluator(object):
    def __init__(self, result_path, te):
        self.result_path = result_path
        self.te = te
        with open(self.te.filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            self.lines = lines


    def get_gt(self):
        gt = []
        for x,y,_ in self.te:
            gt.append(y)
        gt = np.concatenate(gt, axis=0)
        return gt


    def get_pred(self):
        pred = np.load(os.path.join(self.result_path, 'probs.npy'))
        return pred


    def eval_1(self, gt, pred, topk=None, thres=None):
        """
        accuracy, precision, recall, F1
        exclude 0
        """
        assert pred.shape == gt.shape
        pred = self.decision(pred, thres, topk)

        y_or_h  = ((gt == 1) | (pred == 1)).sum(axis=1)
        y_and_h = ((gt == 1) & (pred == 1)).sum(axis=1)

        accuracy  = np.mean(y_and_h / (y_or_h+1e-7))
        precision = np.mean(y_and_h / (pred.sum(axis=1)+1e-7))
        recall    = np.mean(y_and_h / (gt.sum(axis=1)+1e-7))
        F1 = 2*precision*recall / (precision + recall+1e-7)
        return accuracy, precision, recall, F1


    def eval_2(self, gt, pred, topk=None, thres=None):
        """
        hamming loss
        include 0
        """
        assert pred.shape == gt.shape
        pred = self.decision(pred, thres, topk) 
        hammloss = np.mean((pred != gt).sum(axis=1) / pred.shape[1])
        return hammloss


    def decision(self, pred, thres=None, topk=None):
        assert not ( thres is not None and topk is not None)
        pred_cpy = pred.copy()
        if topk is not None:
            pred_argsort = pred_cpy.argsort(axis=1)
            rnge = np.arange(pred_cpy.shape[0])
            pred_cpy[rnge, pred_argsort[:,-topk:].T] = 1
            pred_cpy[rnge, pred_argsort[:,:-topk].T] = 0

        elif thres is None:
            pred_cpy = (pred_cpy > thres).astype(np.float32)

        return pred_cpy


    def save_results(self, pred, topk):
        deci = self.decision(pred, topk=topk)
        with open(os.path.join(self.result_path, 'results.txt'), 'w') as f:
            for i, line in enumerate(self.lines):
                pred_result = np.nonzero(deci[i])[0]
                pred_result = ' '.join([self.te.tags_reverse[j] for j in pred_result])
                towrite = line[:2] + [pred_result]
                f.write('\t'.join(towrite) + '\n')





#def evaluate(result_path, input_param):
#    te = load_data.Input(input_param['testfile'], input_param)
#    evaluator = Evaluator(result_path, te)
#
#
#def main():
#    result_path = os.path.join(MODEL_PATH, '181016-1705/results/181022-1912')
#    input_param = {'imgpath': '/home/commonrec/llh/cbir/framework_v0.1/images_sql/',
#                   'queue_size': 2000,
#                   'batch_size': 32,
#                   'tagfile'  : os.path.join(RESOURCE_PATH, 'chinese_tags.txt'),
#                   'trainfile': os.path.join(RESOURCE_PATH, 'train.txt'),
#                   'validfile': os.path.join(RESOURCE_PATH, 'valid.txt'),
#                   'testfile' : os.path.join(RESOURCE_PATH, '1.txt'),
#                   'mode': 'test' 
#                  }
#    evaluate(result_path, input_param)
#
#
#
#if __name__ == '__main__':
#    main()
