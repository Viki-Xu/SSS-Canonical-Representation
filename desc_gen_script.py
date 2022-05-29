from cProfile import label
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

path1 = '/home/weiqi/Canonical Correction/ssh170/ssh170_raw.npy'
path2 = '/home/weiqi/Canonical Correction/ssh170/ssh174_raw.npy'

canonical_path1 = '/home/weiqi/Canonical Correction/ssh170/ssh170_canonical.npy'
canonical_path2 = '/home/weiqi/Canonical Correction/ssh170/ssh174_canonical.npy'

xtf_file1 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0170-l01s01-20210210-111341.XTF"
xtf_file2 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0174-l05s01-20210210-114538.XTF"

draping_res_folder = '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/9-0169to0182-nbr_pings-5204'
annotator = SSSFolderAnnotator(draping_res_folder, '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/annotations/SSH-0170/correspondence_annotations_SSH-0170.json')

filename1 = 'SSH-0170-l01s01-20210210-111341.cereal'
filename2 = 'SSH-0174-l05s01-20210210-114538.cereal'

matched_kps1 = []
matched_kps2 = []

for kp_id, kp_dict in annotator.annotations_manager.correspondence_annotations.items(
        ):
            if filename1 in kp_dict and filename2 in kp_dict:
                matched_kps1.append(list(kp_dict[filename1]))
                matched_kps2.append(list(kp_dict[filename2]))

matched_kps1 = np.array(matched_kps1).astype(np.float32)
matched_kps2 = np.array(matched_kps2).astype(np.float32)

patch_size = 500
patch_outpath = '/home/weiqi/Canonical Correction/ssh170/patch_pairs'

generate_patches_pair(path1, path2, matched_kps1, matched_kps2, False, patch_size, patch_outpath)

# do canonical transform and save img / kps
canonical_img1, r1, rg1, rg_bar1, canonical_kps1 = canonical_trans(xtf_file1, matched_kps1, len_pings = 1301)
canonical_img2, r2, rg2, rg_bar2, canonical_kps2 = canonical_trans(xtf_file2, matched_kps2, len_pings = 1301)

np.save(canonical_path1, canonical_img1)
np.save(canonical_path2, canonical_img2)

generate_patches_pair(canonical_path1, canonical_path2, canonical_kps1, canonical_kps2, True, patch_size, patch_outpath)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
lowe_ratio = 0.89

number_of_pair = int(len(os.listdir(patch_outpath)) / 2)
patch_comparision = np.full((number_of_pair, 3), '', dtype=object)

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
    
    if len(patch1.annotated_keypoints1) != len(patch2.annotated_keypoints1):
        print(f'Canonical and raw kypoints number not equal!!!')
        continue

    orb = cv.ORB_create()
    patch1_annotated_kps1, patch1_desc1, patch1_img1_normalized = compute_desc_at_annotated_locations(patch1.sss_waterfall_image1, patch1.annotated_keypoints1, orb, kp_size=16)
    patch1_annotated_kps2, patch1_desc2, patch1_img2_normalized = compute_desc_at_annotated_locations(patch1.sss_waterfall_image2, patch1.annotated_keypoints2, orb, kp_size=16)
    patch2_annotated_kps1, patch2_desc1, patch2_img1_normalized = compute_desc_at_annotated_locations(patch2.sss_waterfall_image1, patch2.annotated_keypoints1, orb, kp_size=16)
    patch2_annotated_kps2, patch2_desc2, patch2_img2_normalized = compute_desc_at_annotated_locations(patch2.sss_waterfall_image2, patch2.annotated_keypoints2, orb, kp_size=16)

    patch1_matches = bf.knnMatch(patch1_desc1, patch1_desc2, k=2)
    patch2_matches = bf.knnMatch(patch2_desc1, patch2_desc2, k=2)

    # Apply ratio test
    patch1_good = []
    patch1_correct = []
    patch2_good = []
    patch2_correct = []

    for m,n in patch1_matches:
        if m.distance < lowe_ratio*n.distance:
            patch1_good.append([m])
    for k in patch1_good:
        if k[0].trainIdx == k[0].queryIdx:
            patch1_correct.append(k[0])
    accuracy_patch1 = len(patch1_correct) / len(patch1_good)
    print(f'Accuracy, {accuracy_patch1}...... Is canonical, {patch1.is_canonical}...... Patch no, {patch_comparision[i,0]}')

    for m,n in patch2_matches:
        if m.distance < lowe_ratio*n.distance:
            patch2_good.append([m])
    for k in patch2_good:
        if k[0].trainIdx == k[0].queryIdx:
            patch2_correct.append(k[0])
    accuracy_patch2 = len(patch2_correct) / len(patch2_good)
    print(f'Accuracy, {accuracy_patch2}...... Is canonical, {patch2.is_canonical}...... Patch no, {patch_comparision[i,0]}')

    patch1_matched_img = cv.drawMatchesKnn(patch1_img1_normalized,patch1_annotated_kps1,patch1_img2_normalized,patch1_annotated_kps2,patch1_good, None, flags=2)
    plt.figure()
    plt.title(str(patch1.is_canonical) + patch_comparision[i,0])
    plt.imshow(patch1_matched_img)

    patch2_matched_img = cv.drawMatchesKnn(patch2_img1_normalized,patch2_annotated_kps1,patch2_img2_normalized,patch2_annotated_kps2,patch2_good, None, flags=2)
    plt.figure()
    plt.imshow(patch2_matched_img)
    plt.title(str(patch2.is_canonical) + patch_comparision[i,0])

    pt_patch1_img1 = np.array([[patch1_annotated_kps1[i].pt[1], patch1_annotated_kps1[i].pt[0]] for i in range(len(patch1_annotated_kps1))]).astype(np.float32)
    pt_patch1_img2 = np.array([[patch1_annotated_kps2[i].pt[1], patch1_annotated_kps2[i].pt[0]] for i in range(len(patch1_annotated_kps2))]).astype(np.float32)
    pt_patch2_img1 = np.array([[patch2_annotated_kps1[i].pt[1], patch2_annotated_kps1[i].pt[0]] for i in range(len(patch2_annotated_kps1))]).astype(np.float32)
    pt_patch2_img2 = np.array([[patch2_annotated_kps2[i].pt[1], patch2_annotated_kps2[i].pt[0]] for i in range(len(patch2_annotated_kps2))]).astype(np.float32)

    kept_pt1_patch1, _ = multidim_intersect(patch1.annotated_keypoints1, pt_patch1_img1)
    kept_pt2_patch1, _ = multidim_intersect(patch1.annotated_keypoints2, pt_patch1_img2)
    kept_pt1_patch2, _ = multidim_intersect(patch2.annotated_keypoints1, pt_patch2_img1)
    kept_pt2_patch2, _ = multidim_intersect(patch2.annotated_keypoints2, pt_patch2_img2)

    intersected_kps = kept_pt1_patch1 * kept_pt1_patch2 * kept_pt2_patch1 * kept_pt2_patch2

    _, kps1_for_eval_patch1 = multidim_intersect(patch1.annotated_keypoints1[intersected_kps], pt_patch1_img1)
    _, kps2_for_eval_patch1 = multidim_intersect(patch1.annotated_keypoints2[intersected_kps], pt_patch1_img2)
    _, kps1_for_eval_patch2 = multidim_intersect(patch2.annotated_keypoints1[intersected_kps], pt_patch2_img1)
    _, kps2_for_eval_patch2 = multidim_intersect(patch2.annotated_keypoints2[intersected_kps], pt_patch2_img2)

    patch1_desc1_for_eval = patch1_desc1[kps1_for_eval_patch1]
    patch1_desc2_for_eval = patch1_desc2[kps2_for_eval_patch1]
    patch2_desc1_for_eval = patch2_desc1[kps1_for_eval_patch2]
    patch2_desc2_for_eval = patch2_desc2[kps2_for_eval_patch2]

    desc_distance_patch1 = [cv.norm(patch1_desc1_for_eval[i],patch1_desc2_for_eval[i],cv.NORM_HAMMING) for i in range(patch1_desc2_for_eval.shape[0])]
    desc_distance_patch2 = [cv.norm(patch2_desc1_for_eval[i],patch2_desc2_for_eval[i],cv.NORM_HAMMING) for i in range(patch2_desc2_for_eval.shape[0])]

    plt.figure()
    plt.plot(desc_distance_patch1, label = str(patch1.is_canonical))
    plt.plot(desc_distance_patch2, label = str(patch2.is_canonical))
    plt.legend()
    # find the intersection of the four lists' pos

plt.show()

for filename in os.listdir(patch_outpath):
    with open(os.path.join(patch_outpath, filename), 'rb') as f:
        patch = pickle.load(f)
    
    if patch.annotated_keypoints1.shape[0] != patch.annotated_keypoints2.shape[0]:
        print(f'Keypoints number not equal!!!')
        continue
    
    orb = cv.ORB_create()
    annotated_kps1, desc1, patch1_normalized = compute_desc_at_annotated_locations(patch.sss_waterfall_image1, patch.annotated_keypoints1, orb, kp_size=16)
    annotated_kps2, desc2, patch2_normalized = compute_desc_at_annotated_locations(patch.sss_waterfall_image1, patch.annotated_keypoints2, orb, kp_size=16)
    # if desc1.shape[0] != patch.annotated_keypoints1.shape[0] or desc2.shape[0] != patch.annotated_keypoints2.shape[0]:
    #     print(f'Keypoints cannot be computed!!!')
    #     continue
    
    '''
    pt = [[annotated_kps1[i].pt[1], annotated_kps1[i].pt[0]] for i in range(len(annotated_kps1))]
    can get the pos of kps and cv2.KeyPoint kps
    Long list:  A
    Sub list: B, C
    use A.index(B) to get the sub index?
    By comparing pos, we can map A to B and C by bool list, then get the bool list of A's kps in B and C
    But how to map the bool list of A to B and C?
    how to get the index of same kps of 2 cv2.KeyPoint list?
    '''
    
    # distance = [cv.norm(desc1[i],desc2[i],cv.NORM_HAMMING) for i in range(desc1.shape[0])]
    # plt.figure()
    # plt.plot(distance)
    # plt.title(filename)

    cano_matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    cano_good = []

    for m,n in cano_matches:
        if m.distance < lowe_ratio*n.distance:
            cano_good.append([m])
    score_cano = len(cano_good) / max(len(desc1), len(desc2))
    print(f'Score, {score_cano}...... Is canonical, {patch.is_canonical}')

    for i in range(len(cano_good)):
        cano_good[i][0].trainIdx = cano_good[i][0].queryIdx
    cano_matched_img = cv.drawMatchesKnn(patch1_normalized,annotated_kps1,patch2_normalized,annotated_kps2,cano_good, None, flags=2)
    plt.figure()
    plt.imshow(cano_matched_img)
 
plt.show()


# patch_path  = '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170/test/patch50_SSH-0170_pings_1180to1420_bins_0to1301_isport_False.pkl'

# with open(patch_path, 'rb') as f:
#     patch = pickle.load(f)

# orb = cv2.ORB_create()
# annotated_kps, desc = compute_descriptors_at_annotated_locations(patch, orb, kp_size=16, use_orig_sss_intensities=True)
