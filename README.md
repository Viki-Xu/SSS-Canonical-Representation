# Sidescan Sonar Image Canonical Representation
canonical representation and evaluation for sss images

## Set Up
Install auvlib according to [instruction](https://github.com/nilsbore/auvlib)

## How to use
1. Prepare the `.npz` mesh file corresponding to the sss images by [auvlib](https://github.com/nilsbore/auvlib). 
For keypoints canonical transformation, annotations are also needed.
2. `canonical_trans` imported from `canonical_transformation.py` is all you need to conduct canonical transformation for `.xtf` files (*and keypoints, optional*).
3. `Patch` class and script `patch_gen_scripts.py` is used to generated patch pairs which are extracted around same keypoints. 
This patch pair dataset is used for my thesis' evaluation.
4. Other scripts are used for similarity evaluation and currently under construction.

Below is an example of conducting canonical transformation for a `.xtf` image.
```python
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from canonical_transformation import canonical_trans
import cv2 as cv

xtf_file1 = "/path/to/GullmarsfjordSMaRC20210209/pp/ETPro/ssh/9-0169to0182/SSH-0174-l05s01-20210210-114538.XTF"
mesh_file = "/path/to/mesh-data-roll0.35.cereal_resolution0.5m.npz"
svp_file = "/path/to/GullmarsfjordSMaRC20210209/pp/processed_svp.txt"
raw_img1, canonical_img1, r1, rg1, rg_bar1 = canonical_trans(xtf_file1, mesh_file, svp_file, len_bins = 1301, LambertianModel = "sin_square")
```

## Citation

If you find this code useful, please cite the following paper:

    @article{xu2023evaluation,
    title={Evaluation of a Canonical Image Representation for Sidescan Sonar},
    author={Xu, Weiqi and Ling, Li and Xie, Yiping and Zhang, Jun and Folkesson, John},
    journal={arXiv preprint arXiv:2304.09243},
    year={2023}
    }