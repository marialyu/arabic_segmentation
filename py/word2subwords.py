# -*- coding: utf-8 -*-

from random import randint
import re
import numpy as np
import cv2
from diacritics_classification import classify_diacritics, bound_diacritics, \
    compute_mc_coords


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
    # Sort subwords by x coordinate of center of mass
    mc_coords = compute_mc_coords(cnts)
    tmp = [(scnts, mc_coords[l][0]) for l, scnts in label2cnts.iteritems()]
    tmp.sort(key=lambda x: x[1])
    sorted_subword_cnts = [x[0] for x in tmp]
    return sorted_subword_cnts


def extract_subword_imgs (img_shape, subword_cnts):
    res = []
    for cnts in subword_cnts:
        img = np.ones(img_shape, dtype=np.uint8) * 255
        cv2.drawContours(img, cnts, -1, 0, -1)
        res.append(img)
#        min_x = min([np.min(cnt[:, :, 0]) for cnt in cnts])
#        max_x = max([np.max(cnt[:, :, 0]) for cnt in cnts])
#        min_y = min([np.min(cnt[:, :, 1]) for cnt in cnts])
#        max_y = max([np.max(cnt[:, :, 1]) for cnt in cnts])
#        res.append(img[min_y:max_y+1, min_x:max_x+1])
    return res


def draw_subwords (img, subword_cnts):
    # Convert to bgr
    img = binarize_image(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
              (0, 255, 255), (125, 0, 125), (125, 125, 0), (125, 0, 0),
              (0, 125, 0), (0, 0, 125)]
    # Draw contours
    prev_color = -1
    for cnts in subword_cnts:
        while 1:
            color = colors[randint(0, len(colors)-1)]
            if color != prev_color:
                prev_color = color
                break
        cv2.drawContours(img, cnts, -1, color, -1)
    # Resize image
    sc = min(300.0 / img.shape[0], 1500.0 / img.shape[1])
    img = cv2.resize(img, None, fx=sc, fy=sc)
    return img


def string2subwords (img):
    # Get binary image
    img_bw = binarize_image(img)
    # Find contours
    cnts = find_contours(img_bw)
    # Make labeled img
    pxl_labels = get_pxl_labels(img_bw.shape, cnts)
    # Classify to primary and secondary components
    thresh = 0.15
    is_primary = classify_diacritics(img_bw, cnts, pxl_labels, thresh)
    # Bound secondary components to primary
    secondary2primary = bound_diacritics(pxl_labels, cnts, is_primary)
    # Get list of contours for each subword
    # subwords are sorted through x coord
    subword_cnts = get_subword_cnts(cnts, secondary2primary)
    return subword_cnts


def run ():
    impath = 'path/to/image'
    # Read image
    img = cv2.imread(impath)
    # Get contours of each subword
    subword_cnts = string2subwords(img)
    # Get list of subword images
    if 1:
        subwords = extract_subword_imgs(img.shape, subword_cnts)
        for subword in subwords:
            cv2.imshow('subword', subword)
            cv2.waitKey(300)
    # Draw all on one image
    if 1:
        color_subwords_img = draw_subwords(img, subword_cnts)
        cv2.imshow('Subwords segmentation', color_subwords_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    run()
