from cProfile import label
from operator import le
import string
import sys

sys.path.append('/home/viki/Master_Thesis/auvlib/scripts')
sys.path.append('/home/viki/Master_Thesis/SSS-Canonical-Representation')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, multidim_intersect
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from AlongTrack_Deconvolution import canonical_trans
import cv2 as cv

patch_outpath = '/home/viki/Master_Thesis/SSS-Canonical-Representation/ssh174/patch_pairs/ssh172'
number_of_pair = int(len(os.listdir(patch_outpath)) / 2)
patch_comparision = np.full((number_of_pair, 3), '', dtype=object)

# TODO: for some patch pairs, noise peak exits in one of the patch, should add noise removal func using img histogram.

# get the raw and canonical patch pairs stored in the same patch no
for filename in os.listdir(patch_outpath):
    patch_no = filename.split('_')[0]
    if not patch_no in patch_comparision:
        ind = np.argwhere(patch_comparision[:,0] == '')[0][0]
        patch_comparision[ind, 0] = patch_no
        patch_comparision[ind, 1] = filename
    else:
        ind = np.argwhere(patch_comparision[:,0] == patch_no)[0][0]
        patch_comparision[ind, 2] = filename

for i in range(number_of_pair):
    with open(os.path.join(patch_outpath, patch_comparision[i,1]), 'rb') as f1:
        patch1= pickle.load(f1)
    with open(os.path.join(patch_outpath, patch_comparision[i,2]), 'rb') as f2:
        patch2= pickle.load(f2)
    
    # if len(patch1.annotated_keypoints1) != len(patch2.annotated_keypoints1):
    #     print(f'Canonical and raw kypoints number not equal!!!')
    #     continue

    if patch1.is_canonical:
        patch_cano = patch1
        patch_raw = patch2
    else:
        patch_cano = patch2
        patch_raw = patch1

    corlt_cano = (np.multiply(patch_cano.sss_waterfall_image1, patch_cano.sss_waterfall_image2)).sum() / np.sqrt((np.square(patch_cano.sss_waterfall_image1)).sum()) / np.sqrt((np.square(patch_cano.sss_waterfall_image2)).sum())
    corlt_raw = (np.multiply(patch_raw.sss_waterfall_image1, patch_raw.sss_waterfall_image2)).sum() / np.sqrt((np.square(patch_raw.sss_waterfall_image1)).sum()) / np.sqrt((np.square(patch_raw.sss_waterfall_image2)).sum())

    print(f'Raw correlation, {corlt_raw}...... Cano correlation, {corlt_cano}...... Improvement, {abs(corlt_cano - corlt_raw) / abs(corlt_raw)}...... Patch no, {patch_comparision[i,0]}')