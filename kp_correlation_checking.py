import sys
sys.path.append('/home/weiqi/auvlib/scripts')
sys.path.append('/home/weiqi/Canonical Correction')

import matplotlib.pyplot as plt
from auvlib.bathy_maps import mesh_map, map_draper, base_draper
from auvlib.data_tools import xtf_data, std_data, csv_data
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
import sss_annotation.utils as utils
import numpy as np
import random
import math
import cv2 as cv
import argparse
from PIL import Image
from canonical_transformation import canonical_trans
from patch_gen_scripts import _get_kps_pos_in_patch, compute_desc_at_annotated_locations

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
# matched_kps1 = matched_kps1.reshape(matched_kps1.shape[0],1,2)
matched_kps2 = np.array(matched_kps2).astype(np.float32)
# matched_kps2 = matched_kps2.reshape(matched_kps2.shape[0],1,2)

xtf_file1 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0170-l01s01-20210210-111341.XTF"
xtf_file2 = "/home/weiqi/auvlib/data/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0174-l05s01-20210210-114538.XTF"

raw_img1 = np.load('/home/weiqi/Canonical Correction/ssh170/ssh170_raw.npy')
raw_img2 = np.load('/home/weiqi/Canonical Correction/ssh170/ssh174_raw.npy')
# canonical_img1 = np.load('/home/weiqi/Canonical Correction/ssh170/ssh170_range_correct.npy')
# canonical_img2 = np.load('/home/weiqi/Canonical Correction/ssh170/ssh174_range_correct.npy')

# r: the slant range
# rg: the un-downsampled ground range list
# rg_bar: downsampled ground range
canonical_img1, r1, rg1, rg_bar1 = canonical_trans(xtf_file1, len_pings = 1301, gamma = 0.03, num_iter = 5)
canonical_img2, r2, rg2, rg_bar2 = canonical_trans(xtf_file2, len_pings = 1301, gamma = 0.03, num_iter = 5)

rows1 = raw_img1.shape[0]
raw_img2 = raw_img2[ : rows1, : ]
cano_column2 = canonical_img2.shape[1]
cano_column1 = canonical_img1.shape[1]
canonical_img2 = canonical_img2[ : rows1, : ]
# canonical_img1 = canonical_img1[ : , : cano_column2]

################# Canonical  Representation of .Cereal data #######################
column1 = raw_img1.shape[1]
len_pings = int(column1/2)
r = r1
#last ping of xtf is removed in cereal ---- should be fine if all are from xtf data

# convert kps in image wrt port and starboard pings first
nbr_bins = 1301
nbr_bins_canonical1 = int(cano_column1 / 2)
nbr_bins_canonical2 = int(cano_column2 / 2)
kp_ind1 = np.linspace(nbr_bins-1, 0, num=nbr_bins).astype(np.int16)
kp_ind2 = np.linspace(0, nbr_bins-1, num=nbr_bins).astype(np.int16)
kp_ind = np.concatenate([kp_ind1, kp_ind2])

flag1 = np.zeros(matched_kps1.shape[0])
flag2 = np.zeros(matched_kps2.shape[0])
flag1[matched_kps1[:,1] > nbr_bins-1] = 1
flag2[matched_kps2[:,1] > nbr_bins-1] = 1

divd_matched_kps1 = matched_kps1.copy()
divd_matched_kps2 = matched_kps2.copy()
divd_matched_kps1[:,1] = kp_ind[matched_kps1[:,1].astype(np.int16)]
divd_matched_kps2[:,1] = kp_ind[matched_kps2[:,1].astype(np.int16)]

# convert kps in raw to canonical image
delta_r = r[0]
r_g1 = rg1[divd_matched_kps1[:,0].astype(np.int16), divd_matched_kps1[:,1].astype(np.int16)]
rg_bar1 = rg_bar1[:]+delta_r
ind1 = np.array([np.argmax(rg_bar1 > yg) for yg in r_g1])
canonical_kps1 = divd_matched_kps1
canonical_kps1[flag1>0,1] = ind1[flag1>0] + nbr_bins_canonical1
canonical_kps1[flag1==0,1] = nbr_bins_canonical1 - ind1[flag1==0] -1

r_g2 = rg2[divd_matched_kps2[:,0].astype(np.int16), divd_matched_kps2[:,1].astype(np.int16)]
rg_bar2 = rg_bar2[:]+delta_r
ind2 = np.array([np.argmax(rg_bar2 > yg) for yg in r_g2])
canonical_kps2 = divd_matched_kps2
canonical_kps2[flag2>0,1] = ind2[flag2>0] + nbr_bins_canonical2
canonical_kps2[flag2==0,1] = nbr_bins_canonical2 - ind2[flag2==0] - 1

canonical_img1_test = canonical_img1[:, nbr_bins_canonical1 : ]
canonical_img2_test = canonical_img2[:,  : nbr_bins_canonical2]
canonical_kps1[:,1] = canonical_kps1[:,1] - nbr_bins_canonical1

################## calc similarity ######################
# raw img patch selection
# kps1 = matched_kps1[(matched_kps1[:,0]<1125)*(matched_kps1[:,0]>500)]
# kps2 = matched_kps2[(matched_kps2[:,0]<1200)*(matched_kps2[:,0]>600)]
kps1 = matched_kps1[matched_kps1[:,0]>1125]
kps2 = matched_kps2[matched_kps2[:,0]>1200]
center_kps1 = np.ceil([(kps1[:,1].max() + kps1[:,1].min())/2, (kps1[:,0].max() + kps1[:,0].min())/2])
center_kps2 = np.ceil([(kps2[:,1].max() + kps2[:,1].min())/2, (kps2[:,0].max() + kps2[:,0].min())/2])
radius_x1 = (np.abs(kps1[:,1]-center_kps1[0])).max() + 18
radius_y1 = (np.abs(kps1[:,0]-center_kps1[1])).max() + 18
radius_x2 = (np.abs(kps2[:,1]-center_kps2[0])).max() + 18
radius_y2 = (np.abs(kps2[:,0]-center_kps2[1])).max() + 18
radius_x_raw = max(radius_x1, radius_x2)
radius_y_raw = max(radius_y1, radius_y2)
# radius_raw1 = max(np.square(kps1[:,0]-center_kps1[1]) + np.square(kps1[:,1]-center_kps1[0]))
# radius_raw2 = max(np.square(kps2[:,0]-center_kps2[1]) + np.square(kps2[:,1]-center_kps2[0]))
# radius_raw = np.sqrt(max(radius_raw2, radius_raw1)) + 10

# select circle / rectangular mask from raw

# Y_raw, X_raw = np.ogrid[:raw_img1.shape[0], :raw_img1.shape[1]]
# dist_from_center_raw1 = np.sqrt((X_raw - center_kps1[0])**2 + (Y_raw-center_kps1[1])**2)
# mask_raw1 = dist_from_center_raw1 <= radius_raw
# raw_masked_img1 = raw_img1.copy()
# raw_masked_img1[~mask_raw1] = 0

# Y_raw, X_raw = np.ogrid[:raw_img2.shape[0], :raw_img2.shape[1]]
# dist_from_center_raw2 = np.sqrt((X_raw - center_kps2[0])**2 + (Y_raw-center_kps2[1])**2)
# mask_raw2 = dist_from_center_raw2 <= radius_raw
# raw_masked_img2 = raw_img2.copy()
# raw_masked_img2[~mask_raw2] = 0

patch1 = raw_img1[int(center_kps1[1]-radius_y_raw):int(center_kps1[1]+radius_y_raw),int(center_kps1[0]-radius_x_raw):int(center_kps1[0]+radius_x_raw)]
patch2 = raw_img2[int(center_kps2[1]-radius_y_raw):int(center_kps2[1]+radius_y_raw),int(center_kps2[0]-radius_x_raw):int(center_kps2[0]+radius_x_raw)]
corlt1 = (np.multiply(patch1, patch2)).sum() / np.sqrt((np.square(patch1)).sum()) / np.sqrt((np.square(patch2)).sum()) # -0.8045106055342898  # 0.6113418276265802 for circle
# corlt1 = - (np.multiply(raw_masked_img1, raw_masked_img2)).sum() / np.sqrt((np.square(raw_masked_img1)).sum()) / np.sqrt((np.square(raw_masked_img2)).sum())

# transform the kps from raw coord to patch coord
kps_patch_1 = _get_kps_pos_in_patch(kps1, int(center_kps1[1]-radius_y_raw), int(center_kps1[0]-radius_x_raw))
kps_patch_2 = _get_kps_pos_in_patch(kps2, int(center_kps2[1]-radius_y_raw), int(center_kps2[0]-radius_x_raw))

# generate orb desc from the same kps in two img
orb = cv.ORB_create()
annotated_kps1, desc1, patch1_normalized = compute_desc_at_annotated_locations(patch1, kps_patch_1, orb, kp_size=16)
annotated_kps2, desc2, patch2_normalized = compute_desc_at_annotated_locations(patch2, kps_patch_2, orb, kp_size=16)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
# matches = bf.match(desc1,desc2)

# Apply ratio test
matches = bf.knnMatch(desc1,desc2, k=2)
good = []
lowe_ratio = 0.89
for m,n in matches:
    if m.distance < lowe_ratio*n.distance:
        good.append([m])
score_raw = len(good) / max(len(desc1), len(desc2))
matched_img = cv.drawMatchesKnn(patch1_normalized,annotated_kps1,patch2_normalized,annotated_kps2,matches, None, flags=2)
plt.imshow(matched_img)
plt.show()

# select circle / rectangular mask from canonical img
# cano_kps1 = canonical_kps1[(canonical_kps1[:,0]<1125)*(canonical_kps1[:,0]>500)]
# cano_kps2 = canonical_kps2[(canonical_kps2[:,0]<1200)*(canonical_kps1[:,0]>600)]
cano_kps1 = canonical_kps1[canonical_kps1[:,0]>1125]
cano_kps2 = canonical_kps2[canonical_kps1[:,0]>1200]
cano_center_kps1 = np.ceil([(cano_kps1[:,1].max() + cano_kps1[:,1].min())/2, (cano_kps1[:,0].max() + cano_kps1[:,0].min())/2])
cano_center_kps2 = np.ceil([(cano_kps2[:,1].max() + cano_kps2[:,1].min())/2, (cano_kps2[:,0].max() + cano_kps2[:,0].min())/2])
cano_radius_x1 = (np.abs(cano_kps1[:,1]-cano_center_kps1[0])).max() + 16
cano_radius_y1 = (np.abs(cano_kps1[:,0]-cano_center_kps1[1])).max() + 16
cano_radius_x2 = (np.abs(cano_kps2[:,1]-cano_center_kps2[0])).max() + 16
cano_radius_y2 = (np.abs(cano_kps2[:,0]-cano_center_kps2[1])).max() + 16
radius_x_cano = max(cano_radius_x1, cano_radius_x2)
radius_y_cano = max(cano_radius_y1, cano_radius_y2)

# radius_cano1 = max(np.square(cano_kps1[:,0]-cano_center_kps1[1]) + np.square(cano_kps1[:,1]-cano_center_kps1[0]))
# radius_cano2 = max(np.square(cano_kps2[:,0]-cano_center_kps2[1]) + np.square(cano_kps2[:,1]-cano_center_kps2[0]))
# radius_cano = np.sqrt(max(radius_cano2, radius_cano1))

# Y_cano, X_cano = np.ogrid[:canonical_img1_test.shape[0], :canonical_img1_test.shape[1]]

# dist_from_center_cano1 = np.sqrt((X_cano - cano_center_kps1[0])**2 + (Y_cano-cano_center_kps1[1])**2)
# mask_cano1 = dist_from_center_cano1 <= radius_cano
# cano_masked_img1 = canonical_img1_test.copy()
# cano_masked_img1[~mask_cano1] = 0
# dist_from_center_cano2 = np.sqrt((X_cano - cano_center_kps2[0])**2 + (Y_cano-cano_center_kps2[1])**2)
# mask_cano2 = dist_from_center_cano2 <= radius_cano
# cano_masked_img2 = canonical_img2_test.copy()
# cano_masked_img2[~mask_cano2] = 0

cano_patch1 = canonical_img1_test[int(cano_center_kps1[1]-radius_y_cano):int(cano_center_kps1[1]+radius_y_cano),int(cano_center_kps1[0]-radius_x_cano):int(cano_center_kps1[0]+radius_x_cano)]
cano_patch2 = canonical_img2_test[int(cano_center_kps2[1]-radius_y_cano):int(cano_center_kps2[1]+radius_y_cano),int(cano_center_kps2[0]-radius_x_cano):int(cano_center_kps2[0]+radius_x_cano)]
corlt2 = (np.multiply(cano_patch1, cano_patch2)).sum() / np.sqrt((np.square(cano_patch1)).sum()) / np.sqrt((np.square(cano_patch2)).sum()) # -0.8987460385373434  # 0.6717979315440795 for circle
# corlt2 = - (np.multiply(cano_masked_img1, cano_masked_img2)).sum() / np.sqrt((np.square(cano_masked_img1)).sum()) / np.sqrt((np.square(cano_masked_img2)).sum())

cano_kps_patch_1 = _get_kps_pos_in_patch(cano_kps1, int(cano_center_kps1[1]-radius_y_cano), int(cano_center_kps1[0]-radius_x_cano))
cano_kps_patch_2 = _get_kps_pos_in_patch(cano_kps2, int(cano_center_kps2[1]-radius_y_cano), int(cano_center_kps2[0]-radius_x_cano))

cano_annotated_kps1, cano_desc1, cano_patch1_normalized = compute_desc_at_annotated_locations(cano_patch1, cano_kps_patch_1, orb, kp_size=16)
cano_annotated_kps2, cano_desc2, cano_patch2_normalized = compute_desc_at_annotated_locations(cano_patch2, cano_kps_patch_2, orb, kp_size=16)

# cano_matches = bf.match(cano_desc1,cano_desc2)
cano_matches = bf.knnMatch(cano_desc1, cano_desc2, k=2)

# Apply ratio test
cano_good = []

for m,n in cano_matches:
    if m.distance < lowe_ratio*n.distance:
        cano_good.append([m])
score_cano = len(cano_good) / max(len(cano_desc1), len(cano_desc2))

cano_matched_img = cv.drawMatchesKnn(cano_patch1_normalized,cano_annotated_kps1,cano_patch2_normalized,cano_annotated_kps2,cano_matches, None, flags=2)
plt.imshow(cano_matched_img)
plt.show()

# print(corlt2)

# plt.imshow(cano_patch1)
# plt.figure()
# plt.imshow(cano_patch2)
# plt.show()