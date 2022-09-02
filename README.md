# Sidescan Sonar Image Canonical Representation
canonical representation and evaluation for sss images

## Set Up
Install auvlib according to [instruction](https://github.com/nilsbore/auvlib)

## How to use
1. Prepare the `.npz` mesh file corresponding to the sss images by [auvlib](https://github.com/nilsbore/auvlib). 
For keypoints canonical transformation, annotations are also needed.
2. `canonical_trans` imported from `canonical_transformation.py` is all you need to conduct canonical transformation for `.xtf` files (and keypoints).
3. `Patch` class and script `patch_gen_scripts.py` is used to generated patch pairs which are extracted around same keypoints. 
This patch pair dataset is used for my thesis' evaluation.
4. Other scripts are used for similarity evaluation and currently under construction.

Below is an example of conducting canonical transformation for a `.xtf` image pair and their mutual keypoints, 
and then generating patches in the size of 500 pings.

```python
import sys

sys.path.append('/home/viki/Master_Thesis/auvlib/scripts')
sys.path.append('/home/viki/Master_Thesis/SSS-Canonical-Representation')

import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from patch_gen_scripts import generate_patches_pair, compute_desc_at_annotated_locations, multidim_intersect
from sss_annotation.sss_correspondence_finder import SSSFolderAnnotator
from canonical_transformation import canonical_trans
import cv2 as cv

def cano_img_gen(
        path1, path2, canonical_path1, canonical_path2, xtf_file1, xtf_file2, draping_res_folder, annotation_file, filename1, filename2, patch_outpath,
        mesh_file = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/EM2040/9-0159toend/mesh/mesh-data-roll0.35.cereal_resolution0.5m.npz",
        svp_file = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/processed_svp.txt"):
    '''
    Generate canonical image and Patch class pairs from given xtf files, annotation and mesh

    Parameters
    ----------
    path1/2: str
        path to save raw sss images as np.array
    canonical_path1/2: str
        path to save canonical sss images as np.array
    xtf_file1/2: str
        path to xtf files
    draping_res_folder: str
        path to draping_res_folder, needed for annotator object
    annotation_file: str
        path to annotation
    filename1/2: str
        path to the annotated images, needed to obtain mutual kps of image pairs
    patch_outpath: str
        path to save patch pairs
    mesh_file: str
        corresponding mesh file of the current dataset
    svp_file: str
        processed_svp file of the current dataset
    '''
    matched_kps1 = []
    matched_kps2 = []
    annotator = SSSFolderAnnotator(draping_res_folder, annotation_file)

    for kp_id, kp_dict in annotator.annotations_manager.correspondence_annotations.items(
            ):
                if filename1 in kp_dict and filename2 in kp_dict:
                    matched_kps1.append(list(kp_dict[filename1]))
                    matched_kps2.append(list(kp_dict[filename2]))

    matched_kps1 = np.array(matched_kps1).astype(np.float32)
    matched_kps2 = np.array(matched_kps2).astype(np.float32)

    # do canonical transform and save img / kps
    raw_img1, canonical_img1, r1, rg1, rg_bar1, canonical_kps1 = canonical_trans(xtf_file1, matched_kps1, mesh_file, svp_file, len_bins = 1301, LambertianModel = "sin_square")
    raw_img2, canonical_img2, r2, rg2, rg_bar2, canonical_kps2 = canonical_trans(xtf_file2, matched_kps2, mesh_file, svp_file, len_bins = 1301, LambertianModel = "sin_square")

    np.save(path1, raw_img1)
    np.save(path2, raw_img2)
    np.save(canonical_path1, canonical_img1)
    np.save(canonical_path2, canonical_img2)

    # generate and save patch pairs from raw and cano images
    patch_size = 500
    generate_patches_pair(path1, path2, matched_kps1, matched_kps2, False, patch_size, patch_outpath)
    generate_patches_pair(canonical_path1, canonical_path2, canonical_kps1, canonical_kps2, True, patch_size, patch_outpath)
```
