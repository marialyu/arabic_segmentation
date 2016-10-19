# -*- coding: utf-8 -*-

from random import randint
import numpy as np
import cv2


def extract_subword_imgs (img_shape, subword_cnts):
    res = []
    for cnts in subword_cnts:
        img = np.ones(img_shape, dtype=np.uint8) * 255
        cv2.drawContours(img, cnts, -1, 0, -1)
        res.append(img)
    return res


def draw_subwords_vertically (img_shape, subword_cnts):
    # Compute width of result image
    num_subwords = len(subword_cnts)
    min_xs = [min([np.min(c[:, :, 0]) for c in cnts]) for cnts in subword_cnts]
    max_xs = [max([np.max(c[:, :, 0]) for c in cnts]) for cnts in subword_cnts]
    res_w = max([max_xs[i] - min_xs[i] + 1 for i in range(num_subwords)])
    # Lines
    line0 = np.ones((1, res_w, 3), dtype=np.uint8) * 255
    line1 = np.zeros((2, res_w, 3), dtype=np.uint8)
    line1[:, :, 2] = 255
    # Crop subwords
    croped_subwords = []
    for i, cnts in enumerate(subword_cnts):
        min_x = min_xs[i]
        max_x = max_xs[i]
        min_y = min([np.min(cnt[:, :, 1]) for cnt in cnts])
        max_y = max([np.max(cnt[:, :, 1]) for cnt in cnts])
        h = max_y - min_y + 1
        w = max_x - min_x + 1
        # Shift contours
        x_shift = (res_w - w) / 2
        cnts_shifted = []
        for cnt in cnts:
            cnt_shifted = cnt.copy()
            cnt_shifted[:, :, 0] +=  x_shift - min_x
            cnt_shifted[:, :, 1] -=  min_y
            cnts_shifted.append(cnt_shifted)
        # Draw on crop
        crop = np.ones((h, res_w, 3), dtype=np.uint8) * 255
        cv2.drawContours(crop, cnts_shifted, -1, 0, -1)
        croped_subwords.append(crop)
        # Add line
        if i < num_subwords - 1:
            croped_subwords += [line0, line1, line0]
    # Stack subwords vertically
    res = np.vstack(croped_subwords)
    return res


def draw_subwords (img_shape, subword_cnts):
    img = np.ones(img_shape, dtype=np.uint8) * 255
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