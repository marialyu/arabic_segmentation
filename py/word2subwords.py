# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import cv2
from diacritics_classification import classify_diacritics, bound_diacritics, \
    compute_mc_coords
from viz import draw_subwords, draw_subwords_vertically, extract_subword_imgs


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
    pxl_labels = np.ones(img_shape, dtype=np.int32) * -1
    for i in range(len(cnts)):
        cv2.drawContours(pxl_labels, cnts, i, i, -1)
    return pxl_labels


def get_subword_cnts (cnts, secondary2primary):
    # Get indices of primary components
    num_cnts = len(cnts)
    secondary_labels = secondary2primary.keys()
    primary_labels = set(range(num_cnts)) - set(secondary_labels)
    # Get list of contours for each subword
    label2cnts = {l: [cnts[l]] for l in primary_labels}
    for sl, pl in secondary2primary.iteritems():
        label2cnts[pl].append(cnts[sl])
    subword_cnts = label2cnts.values()
    return subword_cnts


def sort_subwords (subword_cnts):
    subword_primary_cnts = [scnts[0] for scnts in subword_cnts]
    mc_coords = compute_mc_coords(subword_primary_cnts)
    num_subwords = len(subword_cnts)
    tmp = [(subword_cnts[i], mc_coords[i][0]) for i in range(num_subwords)]
    tmp.sort(key=lambda x: x[1], reverse=True)
    sorted_subword_cnts = [x[0] for x in tmp]
    return sorted_subword_cnts

def string2subwords (img, delete_diacritics=False):
    # Get binary image
    img_bw = binarize_image(img)
    # Find contours
    cnts = find_contours(img_bw)
    # Make labeled img
    pxl_labels = get_pxl_labels(img_bw.shape, cnts)
    # Classify to primary and secondary components
    thresh = 0.15
    is_primary = classify_diacritics(img_bw, cnts, pxl_labels, thresh)

    if delete_diacritics:
        cnts = [cnt for i, cnt in enumerate(cnts) if is_primary[i]]
        secondary2primary = {}
    else:
        # Bound secondary components to primary
        secondary2primary = bound_diacritics(pxl_labels, cnts, is_primary)
    # Get list of contours for each subword
    # subwords are sorted through x coord
    subword_cnts = get_subword_cnts(cnts, secondary2primary)
    subword_cnts = sort_subwords(subword_cnts)
    return subword_cnts

# =============================================================================

def run (impath):
    # Read image
    img = cv2.imread(impath)
    # Get contours of each subword
    subword_cnts = string2subwords(img, delete_diacritics=False)
    # Get list of subword images
    if 1:
        subwords = extract_subword_imgs(img.shape, subword_cnts)
        for subword in subwords:
            cv2.imshow('subword', subword)
            cv2.waitKey(300)
    # Draw all on one image
    if 1:
        color_subwords_img = draw_subwords(img.shape, subword_cnts)
        cv2.imshow('Subwords segmentation', color_subwords_img)
    # Get vertical list of subwords
    if 1:
        vsubwords = draw_subwords_vertically(img.shape, subword_cnts)
        cv2.imshow('vsubwords', vsubwords)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        run(args[0])
    else:
        print 'Wrong number of arguments.' + '\n' + \
              'Usage: python word2subwords.py path2image'
