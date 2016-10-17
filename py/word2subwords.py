# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:17:32 2016

@author: marialyu
"""

import os
import re
import pylab
import matplotlib.pyplot as plt
import numpy as np
import cv2
from diacritics_classification import get_scores4primary, make_diacritics_img


def natural_sort_key (x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    res = [convert(c) for c in re.split('([0-9]+)', x)]
    return res


def binarize_image (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.ximgproc.dtFilter(gray.copy(), gray, sigmaSpatial=10,
                                 sigmaColor=10)
    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=31, C=10)
    return img_bw


def find_contours (img_bw):
    _, cnts, _ = cv2.findContours((255 - img_bw).copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if c.shape[0] > 2]
    return cnts


def get_pxl_labels (img_shape, cnts):
    pxl_labels = np.ones(img_shape, dtype=float) * -1
    for i in range(len(cnts)):
        cv2.drawContours(pxl_labels, cnts, i, i, -1)
    return pxl_labels


def run0 (impath, outpath):
    # Read image
    img = cv2.imread(impath)
    img_bw = binarize_image(img)
    # Find contours
    cnts = find_contours(img_bw)
    # Make labeled img
    pxl_labels = get_pxl_labels(img_bw.shape, cnts)
    # Classify to primary and secondary parts
    thresh = 0.15
    scores = get_scores4primary(img_bw, cnts, pxl_labels)
    is_primary = scores >= thresh
    # Output result
    res_img = make_diacritics_img(img_bw, cnts, is_primary)
    if outpath:
        cv2.imwrite(outpath, res_img)
    else:
        cv2.imshow("Image", res_img)
        cv2.waitKey(0)
    return scores


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


def run ():
    imdir = '/home/marialyu/dev/arabic_segmentation/data/'
    outdir = '/home/marialyu/dev/arabic_segmentation/results/'
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
        outpath = os.path.join(outdir, 'ps_' + imname)
        scores = run0(impath, outpath)
        imscores.append(scores)

    # Compute accuracy
    true_ps_labels = read_labels(imdir + 'true_ps_labels.txt')
    im_ps_labels = [ss >= 0.15 for ss in imscores]
    compare_labels(true_ps_labels, im_ps_labels)

    # Plot scores
    stack_scores = np.hstack(imscores)
    stack_labels = np.hstack(true_ps_labels)
    plot_scores(stack_scores, stack_labels)


if __name__ == '__main__':
    run()