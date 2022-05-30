from cProfile import label
from operator import le
import string
import sys

sys.path.append('/home/weiqi/auvlib/scripts')
sys.path.append('/home/weiqi/Canonical Correction')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, multidim_intersect
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from AlongTrack_Deconvolution import canonical_trans
import cv2 as cv

path1 = '/home/weiqi/Canonical Correction/ssh174/ssh174_raw.npy'
path2 = '/home/weiqi/Canonical Correction/ssh174/ssh170_raw.npy'

canonical_path1 = '/home/weiqi/Canonical Correction/ssh174/ssh174_canonical.npy'
canonical_path2 = '/home/weiqi/Canonical Correction/ssh174/ssh170_canonical.npy'

xtf_file1 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0174-l05s01-20210210-114538.XTF"
xtf_file2 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0170-l01s01-20210210-111341.XTF"

draping_res_folder = '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/9-0169to0182-nbr_pings-5204'
annotator = SSSFolderAnnotator(draping_res_folder, '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/annotations/SSH-0174/correspondence_annotations_SSH-0174.json')

filename1 = 'SSH-0174-l05s01-20210210-114538.cereal'
filename2 = 'SSH-0170-l01s01-20210210-111341.cereal'

matched_kps1 = []
matched_kps2 = []

for kp_id, kp_dict in annotator.annotations_manager.correspondence_annotations.items(
        ):
            if filename1 in kp_dict and filename2 in kp_dict:
                matched_kps1.append(list(kp_dict[filename1]))
                matched_kps2.append(list(kp_dict[filename2]))

matched_kps1 = np.array(matched_kps1).astype(np.float32)
matched_kps2 = np.array(matched_kps2).astype(np.float32)

# do canonical transform and save img / kps
raw_img1, canonical_img1, r1, rg1, rg_bar1, canonical_kps1 = canonical_trans(xtf_file1, matched_kps1, len_pings = 1301)
raw_img2, canonical_img2, r2, rg2, rg_bar2, canonical_kps2 = canonical_trans(xtf_file2, matched_kps2, len_pings = 1301)

np.save(path1, raw_img1)
np.save(path2, raw_img2)
np.save(canonical_path1, canonical_img1)
np.save(canonical_path2, canonical_img2)

patch_size = 500
patch_outpath = '/home/weiqi/Canonical Correction/ssh173/patch_pairs/ssh172'

generate_patches_pair(path1, path2, matched_kps1, matched_kps2, False, patch_size, patch_outpath)
generate_patches_pair(canonical_path1, canonical_path2, canonical_kps1, canonical_kps2, True, patch_size, patch_outpath)

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher()
lowe_ratio = 0.89

number_of_pair = int(len(os.listdir(patch_outpath)) / 2)
patch_comparision = np.full((number_of_pair, 3), '', dtype=object)
distance_cano = []
distance_raw = []
cano_correct = []
raw_correct = []
cano_matched = []
raw_matched = []

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
    
    sift = cv.SIFT_create()

    if not(patch_cano.sss_waterfall_image1.size and patch_cano.sss_waterfall_image2.size and patch_raw.sss_waterfall_image1.size and patch_raw.sss_waterfall_image2.size):
        print(f'Patch img does not exit!!!')
        continue

    patch_cano_annotated_kps1, patch_cano_desc1, patch_cano_img1_normalized = compute_desc_at_annotated_locations(patch_cano.sss_waterfall_image1, patch_cano.annotated_keypoints1, sift, kp_size=16)
    patch_cano_annotated_kps2, patch_cano_desc2, patch_cano_img2_normalized = compute_desc_at_annotated_locations(patch_cano.sss_waterfall_image2, patch_cano.annotated_keypoints2, sift, kp_size=16)
    patch_raw_annotated_kps1, patch_raw_desc1, patch_raw_img1_normalized = compute_desc_at_annotated_locations(patch_raw.sss_waterfall_image1, patch_raw.annotated_keypoints1, sift, kp_size=16)
    patch_raw_annotated_kps2, patch_raw_desc2, patch_raw_img2_normalized = compute_desc_at_annotated_locations(patch_raw.sss_waterfall_image2, patch_raw.annotated_keypoints2, sift, kp_size=16)

    patch_cano_matches = bf.knnMatch(patch_cano_desc1, patch_cano_desc2, k=2)
    patch_raw_matches = bf.knnMatch(patch_raw_desc1, patch_raw_desc2, k=2)
    # patch_cano_matches = bf.match(patch_cano_desc1,patch_cano_desc2)
    # patch_raw_matches = bf.match(patch_raw_desc1,patch_raw_desc2)
    # Apply ratio test
    patch_cano_good = []
    patch_cano_correct = []
    patch_raw_good = []
    patch_raw_correct = []

    # for k in patch_cano_matches:
    #     if k.trainIdx == k.queryIdx:
    #         patch_cano_correct.append(k)
    # if len(patch_cano_matches) > 0:
    #     accuracy_patch_cano = len(patch_cano_correct) / len(patch_cano_matches)
    #     print(f'Canonical accuracy, {accuracy_patch_cano}...... Correct, {len(patch_cano_correct)}...... Matched, {len(patch_cano_matches)}...... Patch no, {patch_comparision[i,0]}')

    # for k in patch_raw_matches:
    #     if k.trainIdx == k.queryIdx:
    #         patch_raw_correct.append(k)
    # if len(patch_raw_matches) > 0:
    #     accuracy_patch_raw = len(patch_raw_correct) / len(patch_raw_matches)
    #     print(f'Raw accuracy, {accuracy_patch_raw}...... Correct, {len(patch_raw_correct)}...... Matched, {len(patch_raw_matches)}...... Patch no, {patch_comparision[i,0]}')
    if len(patch_cano_matches) > 1:
        for m,n in patch_cano_matches:
            if m.distance < lowe_ratio*n.distance:
                patch_cano_good.append([m])
        for k in patch_cano_good:
            if k[0].trainIdx == k[0].queryIdx:
                patch_cano_correct.append(k[0])
        if len(patch_cano_good) > 0:
            accuracy_patch_cano = len(patch_cano_correct) / len(patch_cano_good)
            print(f'Canonical accuracy, {accuracy_patch_cano}...... Correct, {len(patch_cano_correct)}...... Matched, {len(patch_cano_good)}...... Patch no, {patch_comparision[i,0]}')

    if len(patch_raw_matches) > 1:
        for m,n in patch_raw_matches:
            if m.distance < lowe_ratio*n.distance:
                patch_raw_good.append([m])
        for k in patch_raw_good:
            if k[0].trainIdx == k[0].queryIdx:
                patch_raw_correct.append(k[0])
        if len(patch_raw_good) > 0:
            accuracy_patch_raw = len(patch_raw_correct) / len(patch_raw_good)
            print(f'Raw accuracy, {accuracy_patch_raw}...... Correct, {len(patch_raw_correct)}...... Matched, {len(patch_raw_good)}...... Patch no, {patch_comparision[i,0]}')

    patch_cano_matched_img = cv.drawMatchesKnn(patch_cano_img1_normalized,patch_cano_annotated_kps1,patch_cano_img2_normalized,patch_cano_annotated_kps2,patch_cano_good, None, flags=2)
    # patch_cano_matched_img = cv.drawMatches(patch_cano_img1_normalized,patch_cano_annotated_kps1,patch_cano_img2_normalized,patch_cano_annotated_kps2,patch_cano_matches, None, flags=2)
    plt.figure()
    plt.title('canonical' + patch_comparision[i,0])
    plt.imshow(patch_cano_matched_img)

    patch_raw_matched_img = cv.drawMatchesKnn(patch_raw_img1_normalized,patch_raw_annotated_kps1,patch_raw_img2_normalized,patch_raw_annotated_kps2,patch_raw_good, None, flags=2)
    # patch_raw_matched_img = cv.drawMatches(patch_raw_img1_normalized,patch_raw_annotated_kps1,patch_raw_img2_normalized,patch_raw_annotated_kps2,patch_raw_matches, None, flags=2)
    plt.figure()
    plt.imshow(patch_raw_matched_img)
    plt.title('raw' + patch_comparision[i,0])

    # pt_patch_cano_img1 = np.array([[patch_cano_annotated_kps1[i].pt[1], patch_cano_annotated_kps1[i].pt[0]] for i in range(len(patch_cano_annotated_kps1))]).astype(np.float32)
    # pt_patch_cano_img2 = np.array([[patch_cano_annotated_kps2[i].pt[1], patch_cano_annotated_kps2[i].pt[0]] for i in range(len(patch_cano_annotated_kps2))]).astype(np.float32)
    # pt_patch_raw_img1 = np.array([[patch_raw_annotated_kps1[i].pt[1], patch_raw_annotated_kps1[i].pt[0]] for i in range(len(patch_raw_annotated_kps1))]).astype(np.float32)
    # pt_patch_raw_img2 = np.array([[patch_raw_annotated_kps2[i].pt[1], patch_raw_annotated_kps2[i].pt[0]] for i in range(len(patch_raw_annotated_kps2))]).astype(np.float32)

    # if not(len(pt_patch_cano_img1) and len(pt_patch_cano_img2) and len(pt_patch_raw_img1) and len(pt_patch_raw_img2)):
    #     print(f'No kps kept in descriptor!!!')
    #     continue

    # kept_pt1_patch_cano, _ = multidim_intersect(patch_cano.annotated_keypoints1, pt_patch_cano_img1)
    # kept_pt2_patch_cano, _ = multidim_intersect(patch_cano.annotated_keypoints2, pt_patch_cano_img2)
    # kept_pt1_patch_raw, _ = multidim_intersect(patch_raw.annotated_keypoints1, pt_patch_raw_img1)
    # kept_pt2_patch_raw, _ = multidim_intersect(patch_raw.annotated_keypoints2, pt_patch_raw_img2)

    # intersected_kps = kept_pt1_patch_cano * kept_pt1_patch_raw * kept_pt2_patch_cano * kept_pt2_patch_raw

    # _, kps1_for_eval_patch_cano = multidim_intersect(patch_cano.annotated_keypoints1[intersected_kps], pt_patch_cano_img1)
    # _, kps2_for_eval_patch_cano = multidim_intersect(patch_cano.annotated_keypoints2[intersected_kps], pt_patch_cano_img2)
    # _, kps1_for_eval_patch_raw = multidim_intersect(patch_raw.annotated_keypoints1[intersected_kps], pt_patch_raw_img1)
    # _, kps2_for_eval_patch_raw = multidim_intersect(patch_raw.annotated_keypoints2[intersected_kps], pt_patch_raw_img2)

    # patch_cano_desc1_for_eval = patch_cano_desc1[kps1_for_eval_patch_cano]
    # patch_cano_desc2_for_eval = patch_cano_desc2[kps2_for_eval_patch_cano]
    # patch_raw_desc1_for_eval = patch_raw_desc1[kps1_for_eval_patch_raw]
    # patch_raw_desc2_for_eval = patch_raw_desc2[kps2_for_eval_patch_raw]

    # desc_distance_patch_cano = [cv.norm(patch_cano_desc1_for_eval[i],patch_cano_desc2_for_eval[i],cv.NORM_HAMMING) for i in range(patch_cano_desc2_for_eval.shape[0])]
    # desc_distance_patch_raw = [cv.norm(patch_raw_desc1_for_eval[i],patch_raw_desc2_for_eval[i],cv.NORM_HAMMING) for i in range(patch_raw_desc2_for_eval.shape[0])]

    # distance_cano.append(desc_distance_patch_cano)
    # distance_raw.append(desc_distance_patch_raw)
    cano_correct.append(len(patch_cano_correct))
    cano_matched.append(len(patch_cano_good))
    raw_correct.append(len(patch_raw_correct))
    raw_matched.append(len(patch_raw_good))

    # plt.figure()
    # plt.plot(desc_distance_patch_cano, label = 'canonical')
    # plt.plot(desc_distance_patch_raw, label = 'raw')
    # plt.legend()
    # # find the intersection of the four lists' pos

plt.show()
distance_cano_1d = sum(distance_cano,[])
distance_raw_1d = sum(distance_raw,[])
decrease = np.array(distance_raw_1d) - np.array(distance_cano_1d)
np.count_nonzero(decrease > 0)
len(decrease)