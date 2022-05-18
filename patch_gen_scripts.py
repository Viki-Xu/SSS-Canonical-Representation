from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import pickle
import shutil
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from Patch import Patch
import cv2
from auvlib.bathy_maps.map_draper import sss_meas_data

def _get_annotated_keypoints_in_patch(path: str, annotations_dir: str,
                                      start_ping: int, end_ping: int,
                                      start_bin: int, end_bin: int) -> dict:
    """
    Returns a list of annotated keypoints found in the patch bounded by start and end pings and
    bins.

    Parameters
    ----------
    path: str
        File path to sss_meas_data file used for patch generation.
    annotations_dir: str
        Path to the directory containing subdirectories with annotations. The annotations are json
        files with names of 'correspondence_annotations_{file_ids}.json'
    start_ping: int
        The index of the first ping in the patch.
    end_ping: int
        The index of the first ping after the patch, i.e. the patch contains pings inside the slice
        of [start_ping:end_ping]
    start_bin: int
        The index of the first bin in the patch
    end_bin: int
        The index of the first bin after the patch, i.e. the patch contains bins inside the slice of
        [start_bin:end_bin]

    Returns
    -------
    keypoints: dict
        A dictionary of keypoint hahshes whose locations fall into the patch.
        The dictionary has the following structure:
            {keypoint hash: {"pos": (ping_idx, bin_idx), "annotation_file": path to the annotation
            file containing this keypoint}
        Note that the keypoint position in "pos" are given in the index of the patch.
        i.e. for a keypoint with (ping_idx, bin_idx), the same keypoint is found in the original
        sss_meas_data at (ping_idx+start_ping, bin_idx + start_bin)
    """
    patch_filename = os.path.basename(path)
    keypoints = {}

    for (dirpath, _, filenames) in os.walk(annotations_dir):
        for filename in filenames:
            if not 'correspondence_annotations' in filename:
                continue
            annotation_filepath = os.path.join(dirpath, filename)
            with open(annotation_filepath, 'r',
                      encoding='utf-8') as annotations_file:
                annotations = json.load(annotations_file)
                for kp_hash, annotations_dict in annotations.items():
                    if patch_filename not in annotations_dict.keys():
                        continue
                    kp_ping_nbr, kp_bin_nbr = annotations_dict[patch_filename]
                    if start_ping <= kp_ping_nbr < end_ping and start_bin <= kp_bin_nbr < end_bin:
                        keypoints[kp_hash] = {
                            "pos":
                            (kp_ping_nbr - start_ping, kp_bin_nbr - start_bin),
                            "annotation_file":
                            annotation_filepath
                        }
    return keypoints

def _get_kps_pos_in_patch(annotated_kps: np.array,
                            start_ping: int,
                            start_bin: int,) -> np.array:
    
    annotated_kps[:,0] -= start_ping
    annotated_kps[:,1] -= start_bin
    return annotated_kps

def compute_desc_at_annotated_locations(
        img: np.array,
        kps: np.array,
        algo: cv2.Feature2D,
        kp_size: int = 16):
    """Compute traditional descriptors using OpenCV's cv2.Feature2D class for a given SSSPatch.

    Parameters
    ----------
    img: np.array
        The descriptors are computed from the normalized 8 bit image of the sss_waterfall_image of
        the patch.
    algo: cv2.Feature2D
        A Feature2D instance from cv2 used to compute the descriptors. e.g. SIFT, SURF, ORB.
    kp_size: int = 16
        The diameter of the neighbourhood to be included in the descriptor computation of a given
        keypoint.

    Returns
    -------
    annotated_kps: np.array
        An array of annotated keypoints of shape (nbr_kps, 2).
        Each array contains the (bin_nbr (x value), ping_nbr (y value)) of one keypoint.
    desc: np.array
        The descriptors computed at the annotated keypoint locations. Shape = (nbr_kps, 128)
    """
    annotated_kps = []
    for i in range(kps.shape[0]):
        ping_nbr, bin_nbr = kps[i,0], kps[i,1]
        annotated_kps.append([bin_nbr, ping_nbr])

    annotated_kps_as_cv2_kp = [
        cv2.KeyPoint(bin_nbr, ping_nbr, _size=kp_size)
        for (bin_nbr, ping_nbr) in annotated_kps
    ]

    normalized_8bit_img = cv2.normalize(img, None, 0, 255,
                                        cv2.NORM_MINMAX).astype('uint8')
    annotated_kps_as_cv2_kp, desc = algo.compute(normalized_8bit_img,
                                                 annotated_kps_as_cv2_kp)
    return annotated_kps_as_cv2_kp, desc, normalized_8bit_img


def generate_patches_pair(file_id: str,
                          path1: str,
                          path2: str,
                          matched_kps1: np.array,
                          matched_kps2: np.array,
                          is_canonical: bool,
                          patch_size: int,
                          step_size: int,
                          patch_outpath: str,
                          patch_id_init_val: int = 0) -> int:
    """
    Generates patches of class Patch from given nparray img and kps with the required specifications.

    Parameters
    ----------
    file_id: str
        File id of the sss_meas_data used for patch generation.
    path1 / 2: str
        File path to sss_meas_data file used for patch generation.
    is_canonical: bool
        if the img and kps have been transformed canonically
    patch_size: int
        Divide one img first, get the kps inside, extract the corresponding kps index list, apply to the other one?
        The number of pings to be included in each patch, i.e. the patch height.
        Note that the patch width is determined by the width of the sss_meas_data.
    step_size: int
        The number of pings each consecutive patch would differ.
    patch_outpath: str
        The path to the directory where the newly generated SSSPatch objects should be stored.
    patch_id_init_val: int
        The initial value of patch_id. This value is set so that the patch_id for each SSSPatch
        is unique in one dataset.

    Returns
    -------
    patch_id: int
        The first unused patch_id.
    """
    img1 = np.load(path1)
    img2 = np.load(path2)
    nadir = int(img1.shape[1] / 2)
    # matched_kps_port1 = matched_kps1[matched_kps1[:,1]<nadir]
    # matched_kps_stdb1 = matched_kps1[not(matched_kps1[:,1]<nadir)]
    # matched_kps_port2 = matched_kps2[matched_kps2[:,1]<nadir]
    # matched_kps_stdb2 = matched_kps2[not(matched_kps2[:,1]<nadir)]
    nbr_patches = np.ceil(img1.shape[0]/patch_size).astype(int)
    for i in range(nbr_patches):
        patch1_kps1_port_ind = (matched_kps_port1[:,0] < (i-1)*patch_size) * (matched_kps_port1[:,0] >= i*patch_size)
        patch1_kps1_stbd_ind = (matched_kps_stdb1[:,0] < (i-1)*patch_size) * (matched_kps_stdb1[:,0] >= i*patch_size)
        if patch1_kps1_port_ind.any() and matched_kps2[]:
            patch_pair_gen()
        if patch1_kps1_stbd_ind.any():
            patch_pair_gen()
        # divide kps1 based pings_size
        # get the corresponding kps in kps2
        # divide to stdb and port
        # check if kps2 across nadir
        # extract patch around kps and save
    return patch_id

def generate_sss_patches(file_id: str,
                         path: str,
                         annotations_dir: str,
                         patch_size: int,
                         step_size: int,
                         patch_outpath: str,
                         patch_id_init_val: int = 0) -> int:
    """
    Generates patches of class SSSPatch from the sss_meas_data with the required specifications.

    Parameters
    ----------
    file_id: str
        File id of the sss_meas_data used for patch generation.
    path: str
        File path to sss_meas_data file used for patch generation.
    valid_idx: list[tuple]
        A list of tuples that indicates the ping ids/indices to be included in the patch
        creation. Each tuple contains a start and end index for a segment of valid pings
        for patch generation.
    annotations_dir: str
        Path to the directory containing subdirectories with annotations. The annotations are json
        files with names of 'correspondence_annotations_{file_ids}.json'
    patch_size: int
        The number of pings to be included in each patch, i.e. the patch height.
        Note that the patch width is determined by the width of the sss_meas_data.
    step_size: int
        The number of pings each consecutive patch would differ.
    patch_outpath: str
        The path to the directory where the newly generated SSSPatch objects should be stored.
    patch_id_init_val: int
        The initial value of patch_id. This value is set so that the patch_id for each SSSPatch
        is unique in one dataset.

    Returns
    -------
    patch_id: int
        The first unused patch_id.
    """
    # replace cereal file with npy array.
    sss_data = sss_meas_data.read_single(path)
    nbr_pings, nbr_bins = sss_data.sss_waterfall_image.shape
    nadir = int(nbr_bins / 2)
    stbd_bins = (0, nadir)
    port_bins = (nadir, nbr_bins)
    pos = np.array(sss_data.pos)

    if not os.path.isdir(patch_outpath):
        os.makedirs(patch_outpath)

    patch_id = patch_id_init_val
    for (seg_start_ping, seg_end_ping) in valid_idx:
        start_ping = seg_start_ping
        end_ping = start_ping + patch_size

        while end_ping <= seg_end_ping:
            patch_pos = pos[start_ping:end_ping, :]

            for start_bin, end_bin in [stbd_bins, port_bins]:
                kps = _get_annotated_keypoints_in_patch(path,
                                                        annotations_dir,
                                                        start_ping=start_ping,
                                                        end_ping=end_ping,
                                                        start_bin=start_bin,
                                                        end_bin=end_bin)
                is_port = (start_bin == port_bins[0])

                if is_port:
                    patch = Patch(
                        patch_id=patch_id,
                        file_id=file_id,
                        filename=os.path.basename(path),
                        start_ping=start_ping,
                        end_ping=end_ping,
                        start_bin=start_bin,
                        end_bin=end_bin,
                        sss_waterfall_image=sss_data.sss_waterfall_image[
                            start_ping:end_ping, start_bin:end_bin],
                        is_port=is_port,
                        annotated_keypoints=kps)

                # For stbd side patch: rotate it so that it looks like is from port side
                else:
                    # Update kp positions
                    for kp_hash, kp_dict in kps.items():
                        orig_ping_idx, orig_bin_idx = kp_dict['pos']
                        new_ping_idx = (end_ping -
                                        start_ping) - orig_ping_idx - 1
                        new_bin_idx = (end_bin - start_bin) - orig_bin_idx - 1
                        kps[kp_hash]['pos'] = (new_ping_idx, new_bin_idx)

                    patch = Patch(
                        patch_id=patch_id,
                        file_id=file_id,
                        filename=os.path.basename(path),
                        start_ping=start_ping,
                        end_ping=end_ping,
                        start_bin=start_bin,
                        end_bin=end_bin,
                        sss_waterfall_image=sss_data.sss_waterfall_image[start_ping:end_ping,
                                                         start_bin:end_bin],
                        is_port=is_port,
                        annotated_keypoints=kps)
                patch_filename = (
                    f'patch{patch_id}_{file_id}_pings_{start_ping}to{end_ping}_'
                    f'bins_{start_bin}to{end_bin}_isport_{is_port}.pkl')
                with open(os.path.join(patch_outpath, patch_filename),
                          'wb') as f:
                    pickle.dump(patch, f)

                patch_id += 1
            # Update start and end idx for the generation of a new SSSPatch
            start_ping += step_size
            end_ping = start_ping + patch_size
    return patch_id