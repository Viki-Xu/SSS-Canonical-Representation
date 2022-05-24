import sys
sys.path.append('/home/weiqi/auvlib/scripts')
sys.path.append('/home/weiqi/Canonical Correction')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations
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

for filename in os.listdir(patch_outpath):
    with open(os.path.join(patch_outpath, filename), 'rb') as f:
        patch = pickle.load(f)
    orb = cv.ORB_create()
    annotated_kps1, desc1, patch1_normalized = compute_desc_at_annotated_locations(patch.sss_waterfall_image1, patch.annotated_keypoints1, orb, kp_size=16)
    annotated_kps2, desc2, patch2_normalized = compute_desc_at_annotated_locations(patch.sss_waterfall_image1, patch.annotated_keypoints2, orb, kp_size=16)
    cano_matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    cano_good = []

    for m,n in cano_matches:
        if m.distance < lowe_ratio*n.distance:
            cano_good.append([m])
    score_cano = len(cano_good) / max(len(desc1), len(desc2))
    print(f'Score, {score_cano}...... Is canonical, {patch.is_canonical}')
    if desc1.shape[0] != len(annotated_kps1) or desc2.shape[0] != len(annotated_kps2):
        print(f'Keypoints cannot be computed!!!')
    cano_matched_img = cv.drawMatchesKnn(patch1_normalized,annotated_kps1,patch2_normalized,annotated_kps2,cano_matches, None, flags=2)
    plt.figure()
    plt.imshow(cano_matched_img)
 
plt.show()


# patch_path  = '/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170/test/patch50_SSH-0170_pings_1180to1420_bins_0to1301_isport_False.pkl'

# with open(patch_path, 'rb') as f:
#     patch = pickle.load(f)

# orb = cv2.ORB_create()
# annotated_kps, desc = compute_descriptors_at_annotated_locations(patch, orb, kp_size=16, use_orig_sss_intensities=True)
