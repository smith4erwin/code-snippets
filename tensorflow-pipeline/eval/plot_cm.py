# -*- coding: utf-8 -*-

import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_cm(cm, labels, normalize, figsize, save_path=None):
    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1, keepdim=True)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    fmt = '.2f'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] == 0.0:
            continue
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', \
                color='white' if cm[i, j] > thresh else 'black')
    ax.tick_params(axis='x', top=True, bottom=True)
    tick_marks = np.arange(len(labels))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if save_path is not None:
        fig.savefig(save_path)
    return fig
