# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:17:32 2016

@author: marialyu
"""

import os
import re
from random import randint
import pylab
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    pylab.show()


def run ():
    imdir = '/home/marialyu/dev/arabic_segmentation/data/'
    outdir = '/home/marialyu/dev/arabic_segmentation/results/'
    idxs = range(1, 7) + range(10, 18) + range(18, 24) + [27, 28]

    for f in os.listdir(imdir):
        full_f = os.path.join(imdir, f)
        if os.path.isfile(full_f) and f.startswith('word'):
            idx = int(re.search('\d+', f).group())
            if idx in idxs:
                outpath = os.path.join(outdir, 'ps_' + f)
                run0(full_f, outpath)


def get_word_line_y (mc_coords, areas):
    repmat_areas = np.tile(areas, (2, 1)).T
    common_mc_coord = np.sum(repmat_areas * mc_coords, axis=0) / np.sum(areas)
    word_line_y = int(common_mc_coord[1])
    return word_line_y


def compute_dot_area (th, labeled, cnts):
    # Prepare
    areas = compute_areas(cnts)
    num_cnts = len(cnts)

    # Skeleton
#    sk = skeletonize(np.logical_not(th))
#    sk_labeled = np.ones_like(labeled) * -1
#    sk_labeled[sk] = labeled[sk]
#    sk_lens = np.array([np.sum(sk_labeled == l) for l in range(num_cnts)])
#    is_small_enough = sk_lens / float(min(sk_lens)) < 2.0
#    is_small_enough = np.ones(num_cnts, dtype=bool)
#    areas = np.array([np.sum(labeled == l) for l in range(num_ctns)])

    # Detect holes
    with_hole = []
    for l in range(num_cnts):
        l_mask = labeled == l
        bg_mask = 255-th == 0
        with_hole.append(np.any(np.logical_and(l_mask, bg_mask)))

    # Compactness
    compactness_scores = np.zeros(num_cnts)
    for i in range(num_cnts):
        if not with_hole[i]:
            perimeter = cv2.arcLength(cnts[i], closed=True)
            compactness_scores[i] = 4 * np.pi * areas[i] / perimeter ** 2
    compactness_mask = compactness_scores >= 0.7

    res_mask = np.logical_and(compactness_mask, np.logical_not(with_hole))
    dot_area = np.median(areas[res_mask])
    return dot_area


def compute_areas (cnts):
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    return np.array(areas)

def compute_mc_coords (cnts):
    cnts_moments = [cv2.moments(c) for c in cnts]
    mc_coords = [(M["m10"]/M["m00"], M["m01"]/M["m00"]) for M in cnts_moments]
    mc_coords = np.array(mc_coords)
    return mc_coords


def compute_dist_scores (cnts, mc_coords, line_y):
    # Get bb for whole word
    stacked_cnt = np.vstack(cnts)
    common_bb = cv2.boundingRect(stacked_cnt)   # x, y, w, h

    num_cnts = len(cnts)
    scores = []
    for i in range(num_cnts):
        h = max(line_y, common_bb[-1] - line_y)
        dist = 1 - abs(mc_coords[i][1] - line_y) / float(h)
        scores.append(dist)
    return np.array(scores)


def compute_area_scores (areas):
    scores = [area / float(np.max(areas)) for area in areas]
    return scores


def compute_area_scores2 (dot_area, areas):
    num_cnts = areas.size
    if dot_area is None:
        return np.zeros(num_cnts, dtype=float)

    scores = np.array([area / float(dot_area) for area in areas])
    scores /= 15.0
    scores[scores > 1.0] = 1.0
    return scores


def compute_area_scores3 (dot_area, areas):
    if dot_area is None:
        dot_area = np.min(areas)
    max_area = min(np.max(areas), 20.0 * dot_area)
    scores = np.array([area / float(max_area) for area in areas])
    scores[areas <= dot_area] = 0.0
    return scores


def compute_crossline_scores (labeled, line_y):
    num_labels = int(np.max(labeled)) + 1
    scores = []
    for i in range(num_labels):
        area_above = np.sum(labeled[:line_y, :] == i)
        area_below = np.sum(labeled[line_y:, :] == i)

        if area_above and area_below:
            area = area_above + area_below
            score = min(area_above, area_below) / float(area)
        else:
            score = 0
        scores.append(score)
    return np.array(scores)


def compute_min_dist2line_scores (cnts, line_y):
    # Get bb for whole word
    stacked_cnt = np.vstack(cnts)
    common_bb = cv2.boundingRect(stacked_cnt)   # x, y, w, h
    h = max(line_y, common_bb[-1] - line_y)

    scores = []
    for cnt in cnts:
        y_top = np.min(cnt[:, :, 1])
        y_bottom = np.max(cnt[:, :, 1])
        if y_top > line_y:
            dist = y_top - line_y
        elif y_bottom < line_y:
            dist = line_y - y_bottom
        else:
            dist = 0.0
        score = dist / float(h)
        scores.append(score)
    return np.array(scores)


def find_min_max (labeled, axis):
    num_els = labeled.shape[axis]
    num_unique_in_els = np.zeros(num_els)
    min_max = {}
    for idx in range(num_els):
        # Find part labels in row/column
        if axis:
            unique = set(np.unique(labeled[:, idx]))
        else:
            unique = set(np.unique(labeled[idx, :]))
        unique.discard(-1)
        # Save number of labels
        num_unique_in_els[idx] = len(unique)
        # Update min max info
        for l in unique:
            if l not in min_max:
                min_max[l] = [idx, idx]
            else:
                min_max[l][1] = idx
    return min_max


def compute_hhole_scores (labeled):
    min_max_x = find_min_max (labeled, axis=1)
    num_cols = labeled.shape[1]
    num_unique_in_cols = np.zeros(num_cols)
    for col_idx in range(num_cols):
        # Find part labels in column
        cols_unique = set(np.unique(labeled[:, col_idx]))
        cols_unique.discard(-1)
        # Save number of labels
        num_unique_in_cols[col_idx] = len(cols_unique)

    num_labels = np.max(labeled) + 1
    scores = np.zeros(num_labels, dtype=float)
    for l, (min_x, max_x) in min_max_x.iteritems():
        scores[l] = np.mean(num_unique_in_cols[min_x:max_x+1] - 1 == 0)
    return scores


def compute_h_scores (th, labeled):

    sk = skeletonize(np.logical_not(th))
    sk_labeled = np.ones_like(labeled) * -1
    sk_labeled[sk] = labeled[sk]

    min_max_y = find_min_max(sk_labeled, axis=0)
    min_max_x = find_min_max(sk_labeled, axis=1)

    scores = []
    num_labels = int(np.max(labeled) + 1)
    for l in range(num_labels):
        h = min_max_y[l][1] - min_max_y[l][0]
        w = min_max_x[l][1] - min_max_x[l][0] + 1.0
        score = float(h) / w / 8.0
        scores.append(score)
    return np.array(scores)


def compute_scores (dist_scores, crossline_scores, min_dist_scores,
                    area_scores, area_scores2, area_scores3, hhole_scores,
                    h_scores):
    scores = []
    num_cnts = dist_scores.size
    for i in range(num_cnts):
        dist = dist_scores[i] / 2.0
        crossline_ratio = crossline_scores[i]
        min_dist = (min_dist_scores[i])
        area_ratio = area_scores[i]
        area_ratio2 = area_scores2[i]
        area_ratio3 = area_scores3[i]
        hhole_score = hhole_scores[i]
        h_score = h_scores[i] ** 2.0
#        score = (dist + area_ratio) / 2 + crossline_ratio
#        score = 0.4 * dist + 0.2 * area_ratio + 0.4 * crossline_ratio
#        score = (dist + area_ratio2 + hhole_score) / 3.0 #thresh = 0.2
        score = (2 * area_ratio3 + hhole_score + crossline_ratio - min_dist +
                 2 * h_score) / 4.0
        scores.append(score)
        print('%3d: %.2f (crl=%.2f, md=%.2f, a3=%.2f, hh=%.2f, h=%.2f)' %
              (i, score, crossline_ratio, min_dist, area_ratio3, hhole_score,
               h_score))
#        print('%3d: %.2f (d=%.2f, crl=%.2f, md=%.2f, a=%.2f, a2=%.2f, a3=%.2f,'
#              ' hh=%.2f, h=%.2f)' % (i, score, dist, crossline_ratio, min_dist,
#                                     area_ratio, area_ratio2, area_ratio3,
#                                     hhole_score, h_score))
    return scores


def run0 (impath, outpath):
    # Read image
    img = cv2.imread(impath)
#    img = cv2.resize(img, None, fx=5.0, fy=5.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.ximgproc.dtFilter(gray.copy(), gray, sigmaSpatial=10, sigmaColor=10)
    # Convert to bw
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=31, C=10)

    # Find contours
    _, cnts, _ = cv2.findContours((255 - th).copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if c.shape[0] > 2]
    num_cnts = len(cnts)

    # Make labeled img
    labeled = np.ones(th.shape, dtype=float) * -1
    for i in range(num_cnts):
        cv2.drawContours(labeled, cnts, i, i, -1)
#    labeled[255-th == 0] = -1

    # Find coordinates of centers of mass for each part
    areas = compute_areas(cnts)
    mc_coords = compute_mc_coords(cnts)
    # Find horizontal word line
    word_line_y = get_word_line_y (mc_coords, areas)

    # Compute distances to word line
    dist_scores = compute_dist_scores(cnts, mc_coords, word_line_y)
    # Compute min distances to word line
    min_dist_scores = compute_min_dist2line_scores (cnts, word_line_y)
    # Compute cross line score
    crossline_scores = compute_crossline_scores(labeled, word_line_y)
    # Compute area score
    area_scores = compute_area_scores(areas)
    # Find dot
    dot_area = compute_dot_area(th, labeled, cnts)
    area_scores2 = compute_area_scores2(dot_area, areas)
    #
    area_scores3 = compute_area_scores3(dot_area, areas)
    # Find horizontal holes score
    hhole_scores = compute_hhole_scores(labeled)
    #
    h_scores = compute_h_scores(th, labeled)
    # Compute result score
    scores = compute_scores(dist_scores, crossline_scores, min_dist_scores,
                            area_scores, area_scores2, area_scores3,
                            hhole_scores, h_scores)

    # Draw on image
    cv2.line(img, (0, word_line_y), (img.shape[1], word_line_y), (255, 0, 125), 1)
    thresh = 0.15
    cnts_primary = [cnt for i, cnt in enumerate(cnts) if scores[i] >= thresh]
    cnts_secondary = [cnt for i, cnt in enumerate(cnts) if scores[i] < thresh]
    cv2.drawContours(img, cnts_primary, -1, (255, 125, 0), -1)
    cv2.drawContours(img, cnts_secondary, -1, (0, 125, 255), -1)

    sc = 1800.0 / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, None, fx=sc, fy=sc)

    for i, mc in enumerate(mc_coords):
        cv2.putText(img, str(i), (int(mc[0]*sc), int(mc[1]*sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
#    cv2.imwrite(outpath, img)


if __name__ == '__main__':
    run()