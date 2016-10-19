# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.morphology import skeletonize


def get_word_line_y (mc_coords, areas):
    repmat_areas = np.tile(areas, (2, 1)).T
    common_mc_coord = np.sum(repmat_areas * mc_coords, axis=0) / np.sum(areas)
    word_line_y = int(common_mc_coord[1])
    return word_line_y


def find_min_max (labeled, axis):
    num_els = labeled.shape[axis]
    num_unique_in_els = np.zeros(num_els)
    num_labels = np.max(labeled) + 1
    min_max = np.ones([num_labels, 2], dtype=int) * -1
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
            if min_max[l][0] == -1:
                min_max[l] = [idx, idx]
            else:
                min_max[l][1] = idx
    return min_max


def find_min_max_cnt (cnt, axis):
    min_c = np.min(cnt[:, :, axis])
    max_c = np.max(cnt[:, :, axis])
    return min_c, max_c


def compute_areas (cnts):
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    return np.array(areas)


def compute_mc_coords (cnts):
    cnts_moments = [cv2.moments(c) for c in cnts]
    mc_coords = [(M["m10"]/M["m00"], M["m01"]/M["m00"]) for M in cnts_moments]
    mc_coords = np.array(mc_coords)
    return mc_coords


def compute_dot_area (th, labeled, cnts):
    # Prepare
    areas = compute_areas(cnts)
    num_cnts = len(cnts)

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


def compute_area_scores (dot_area, areas):
    if dot_area is None:
        dot_area = np.min(areas)
    max_area = min(np.max(areas), 20.0 * dot_area)
    scores = np.array([area / float(max_area) for area in areas])
    scores[areas <= dot_area] = 0.0
    scores[scores > 1.0] = 1.0
    return scores


def compute_crossline_scores (labeled, line_y):
    min_max_y = find_min_max (labeled, axis=0)
    scores = []
    for l, (ymin, ymax) in enumerate(min_max_y):
        score = min(line_y - ymin, ymax - line_y) / float(ymax - ymin)
        score = max(0, score) * 2
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


def compute_hcover_scores (labeled):
    min_max_x = find_min_max(labeled, axis=1)
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
    for l, (min_x, max_x) in enumerate(min_max_x):
        not_covered = num_unique_in_cols[min_x:max_x+1] - 1 == 0
        scores[l] = np.mean(not_covered)
    return scores


def compute_vline_scores (th, labeled):

    sk = skeletonize(np.logical_not(th))
    sk_labeled = np.ones_like(labeled) * -1
    sk_labeled[sk] = labeled[sk]

    min_max_y = find_min_max(sk_labeled, axis=0)
    min_max_x = find_min_max(sk_labeled, axis=1)

    num_labels = np.max(labeled) + 1
    scores = np.zeros(num_labels, dtype=float)
    h = min_max_y[:, 1] - min_max_y[:, 0]
    w = min_max_x[:, 1] - min_max_x[:, 0] + 1.0
    scores = (h.astype(float) / w / 8.0) ** 2
    scores[scores <= 0.1] = 0.0
    scores[scores > 1.0] = 1.0
    return scores


def compute_scores (crossline_scores, min_dist_scores, area_scores,
                    hcover_scores, vline_scores):
    scores = []
    num_cnts = min_dist_scores.size

    for i in range(num_cnts):
        crossline_ratio = crossline_scores[i]
        min_dist = min_dist_scores[i]
        area_ratio = area_scores[i]
        hcover_score = hcover_scores[i]
        vline_score = vline_scores[i]

        score = (2 * area_ratio + hcover_score + 0.5 * crossline_ratio - \
                 min_dist + vline_score) / 4.0  # thresh = 0.15
        scores.append(score)
    return np.array(scores)


def get_scores4primary (img_bw, cnts, pxl_labels):
    # Find coordinates of centers of mass for each part
    areas = compute_areas(cnts)
    mc_coords = compute_mc_coords(cnts)
    # Find horizontal word line
    word_line_y = get_word_line_y(mc_coords, areas)
    # Compute min distances to word line
    min_dist_scores = compute_min_dist2line_scores(cnts, word_line_y)
    # Compute cross line score
    crossline_scores = compute_crossline_scores(pxl_labels, word_line_y)
    # Compute area score
    dot_area = compute_dot_area(img_bw, pxl_labels, cnts)
    area_scores = compute_area_scores(dot_area, areas)
    # Find horizontal holes score
    hcover_scores = compute_hcover_scores(pxl_labels)
    # How much component looks like vertical line
    vline_scores = compute_vline_scores(img_bw, pxl_labels)
    # Compute result score
    scores = compute_scores(crossline_scores, min_dist_scores, area_scores,
                            hcover_scores, vline_scores)
    return scores


def fix_not_vcovered_secondary (pxl_labels, cnts, scores, is_primary):
    for l in range(len(cnts)):
        if is_primary[l]:
            continue
        # Find if secondary component is covered vertically by any primary one
        min_x, max_x = find_min_max_cnt(cnts[l], axis=0)
        rect = pxl_labels[:, min_x:max_x+1]
        primary_labels = np.where(is_primary)[0]
        is_primary_rect = np.in1d(rect, primary_labels).reshape(rect.shape)
        # If there is no such component -- look for secondary one that
        # covers vertically current component (including it)
        # has maximal score
        # => change it to be primary
        if not np.any(is_primary_rect):
            secondary_labels = np.where(np.logical_not(is_primary))[0]
            sec_labels_rect = np.setdiff1d(rect, secondary_labels)
            sec_labels_rect = np.setdiff1d(rect, [-1])
            new_pl = sec_labels_rect[np.argmax(scores[sec_labels_rect])]
            is_primary[new_pl] = 1
    return is_primary


def classify_diacritics (img_bw, cnts, pxl_labels, thresh):
    scores = get_scores4primary(img_bw, cnts, pxl_labels)
    is_primary = scores >= thresh
    is_primary = fix_not_vcovered_secondary(pxl_labels, cnts, scores,
                                            is_primary)
    return is_primary


def find_closest_primary (search_rect, primary_labels, search_coords):
    # Find closest column with primary parts
    is_primary = np.in1d(search_rect, primary_labels).reshape(search_rect.shape)
    primary_cover_mask = np.any(is_primary, axis=0)
    if not np.any(primary_cover_mask):
        return
    primary_cover_x = np.where(primary_cover_mask)[0]
    diff = abs(primary_cover_x - search_coords[0])
    col_idx = primary_cover_x[np.argmin(diff)]
    column = search_rect[:, col_idx]
    # Find closest primary in column
    col_primary_mask = np.in1d(column, primary_labels)
    col_primary_y = np.where(col_primary_mask)[0]
    y_diff = abs(col_primary_y - search_coords[1])
    closest_primary_y = col_primary_y[np.argmin(y_diff)]
    closest_primary_label = column[closest_primary_y]
    return closest_primary_label


def bound_diacritics (pxl_labels, cnts, is_primary):
    primary_labels = np.where(is_primary)[0]
    secondary_labels = np.where(np.logical_not(is_primary))[0]
    mc_coords = compute_mc_coords(cnts)
    min_max_x = find_min_max(pxl_labels, axis=1)

    new_labels = {}
    for sl in secondary_labels:
        min_x, max_x = min_max_x[sl]
        rect = pxl_labels[:, min_x:max_x+1]
        mc = mc_coords[sl]
        search_coords = [mc[0]-min_x, mc[1]]
        new_sl = find_closest_primary(rect, primary_labels, search_coords)
        new_labels[sl] = new_sl
    return new_labels
