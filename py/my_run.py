# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:15:07 2016

@author: marialyu
"""

import os
from word2subwords import *


def my_run ():
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

    # Divide string in subwords
    for imname in imnames:
        # Read image
        impath = os.path.join(imdir, imname)
        img = cv2.imread(impath)
        # Get contours of each subword
        subword_cnts = string2subwords(img, delete_diacritics=True)
        # Get list of subword images
        if 0:
            subwords = extract_subword_imgs(img.shape, subword_cnts)
            for subword in subwords:
                cv2.imshow('subword', subword)
                cv2.waitKey(150)
        # Draw all on one image
        if 1:
            color_subwords_img = draw_subwords(img.shape, subword_cnts)
            outpath = os.path.join(outdir, 'primary_' + imname)
            cv2.imwrite(outpath, color_subwords_img)
#            cv2.imshow('Subwords segmentation', color_subwords_img)
#            cv2.waitKey(0)
        # Get list of subword images
        if 0:
            vsubwords = draw_subwords_vertically(img.shape, subword_cnts)
            cv2.imshow('vsubwords', vsubwords)
            cv2.waitKey(0)


if __name__ == '__main__':
    my_run()