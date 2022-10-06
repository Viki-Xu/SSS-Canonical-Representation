import sys

sys.path.append('/home/viki/Master_Thesis/auvlib/scripts')
sys.path.append('/home/viki/Master_Thesis/SSS-Canonical-Representation')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, patch_rotated
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from canonical_transformation import canonical_trans
from desc_gen_script import desc_evaluation, similarity_compare
import cv2 as cv
from tabulate import tabulate


data_path = '/home/viki/Master_Thesis/SSS-Canonical-Representation/data/ssh'
sss_no = ['170', '171', '172', '173', '174']
cano_match = []
raw_match = []
similarity_comparision = []
for i in sss_no:
    for j in sss_no:
        if i == j:
            continue
        patch_outpath = data_path + i + '/patch_pairs/ssh' + j
        matcher = 'KnnMatcher'
        descType = 'sift'
        rotate = False
        # if (int(i)+int(j))%2:
        #     rotate = True
        cano_match, raw_match, similarity_comparision = desc_evaluation(patch_outpath, matcher, descType, rotate, cano_match, raw_match, similarity_comparision)

cano_match = np.array(cano_match, dtype=object)
raw_match = np.array(raw_match, dtype=object)
similarity_comparision = np.array(similarity_comparision)
imprv_res = similarity_comparision[:,2]

raw_correct_sum = raw_match[:,0].sum()
raw_match_sum = raw_match[:,1].sum()
cano_correct_sum = cano_match[:,0].sum()
cano_match_sum = cano_match[:,1].sum()
print(tabulate(similarity_comparision))
print(f'Overall Raw correct, {raw_correct_sum}...... Raw matched, {raw_match_sum}...... ACC, {raw_correct_sum/raw_match_sum}')
print(f'Overall Cano correct, {cano_correct_sum}...... Cano matched, {cano_match_sum}...... ACC, {cano_correct_sum/cano_match_sum}')
print(f'Chi-Square distance of {len(imprv_res[imprv_res<0])} pairs out of {len(imprv_res)} pairs decrease, ratio: {len(imprv_res[imprv_res<0])/len(imprv_res)}')

plt.hist(similarity_comparision[:,2])
plt.show()
# print(tabulate(cano_match))
# print(tabulate(raw_match))