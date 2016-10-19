# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:10:45 2016

@author: marialyu
"""

import os
import re
import numpy as np
import cv2
import pylab
import matplotlib.pyplot as plt
from diacritics_classification import get_scores4primary
from word2subwords import *


def get_scores ():
    imdir = '/home/marialyu/dev/arabic_segmentation/data/'
    idxs = range(1, 7) + range(10, 18) + range(18, 24) + [27, 28]
    # Get image names
    imnames = []
    for f in os.listdir(imdir):
        full_f = os.path.join(imdir, f)
        if os.path.isfile(full_f) and f.startswith('word'):
            idx = int(re.search('\d+', f).group())
            if idx in idxs:
                imnames.append(f)
    imnames.sort(key=natural_sort_key)

    # Get scores
    imscores = []
    for imname in imnames:
        impath = os.path.join(imdir, imname)
        img = cv2.imread(impath)
        img_bw = binarize_image(img)
        cnts = find_contours(img_bw)
        pxl_labels = get_pxl_labels(img_bw.shape, cnts)
        scores = get_scores4primary(img_bw, cnts, pxl_labels)
        imscores.append(scores)

    # Compute accuracy
    true_ps_labels = read_labels(imdir + 'true_ps_labels.txt')
    im_ps_labels = [ss >= 0.15 for ss in imscores]
    compare_labels(true_ps_labels, im_ps_labels)

    # Plot scores
    stack_scores = np.hstack(imscores)
    stack_labels = np.hstack(true_ps_labels)
    plot_scores(stack_scores, stack_labels)


def read_labels (path):
    ps_labels = []
    with open(path) as f:
        for line in f:
            line = line[:-1]
            if line.endswith('.jpg'):
                if ps_labels:
                    ps_labels[-1] = np.array(ps_labels[-1], bool)
                ps_labels.append([])
                labels = ps_labels[-1]
            else:
                labels.append(int(re.findall('\d+', line)[1]))
    return ps_labels


def compare_labels (true, predicted):
    means = [np.mean(true[i] == predicted[i]) for i in range(len(true))]
    acc = sum(means) / float(len(means))
    print(acc)


def plot_scores (scores, labels):
    x = y = scores
    colors = labels.astype(int)
    plt.scatter(x, y, s=150, c=colors, alpha=0.5)
    pylab.show()


if __name__ == '__main__':
    get_scores()