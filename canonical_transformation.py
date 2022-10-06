from ast import arg
from heapq import heappush
import numpy as np
import matplotlib.pyplot as plt
from auvlib.bathy_maps import mesh_map, map_draper, base_draper, draw_map
from auvlib.data_tools import xtf_data, std_data, csv_data
from scipy.signal import convolve2d as conv2
from scipy.stats import norm
from scipy.optimize import nnls
from skimage.restoration import richardson_lucy
import time
import pdb

def canonical_trans(xtf_file, mesh_file, svp_file, *keyPoint, deconv=False, len_bins = 1301, LambertianModel = "sin_square"):
    """
    Canonical transformation for sss images and kps.

    Parameters
    ----------
    xtf_file: str
        File path to xtf_data file used for canonical transformation calculation.
    *keyPoint: np.array, optional
        Numpy array which contains kps coordination in this downsized xtf image in the form of [nbr_ping, nbr_bin] for each point, whose coordinate system is the same as annotation file.
    mesh_file, svp_file: str
        Files to load base_draper and heights for each ping
        For the "GullmarsfjordSMaRC20210209" dataset, default path in this client is:
        mesh_file = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/EM2040/9-0159toend/mesh/mesh-data-roll0.35.cereal_resolution0.5m.npz",
        svp_file = "/home/viki/Master_Thesis/auvlib/data/GullmarsfjordSMaRC20210209/pp/processed_svp.txt"
    len_bins: int
        Number of bins contained in each ping after downsampling. It should be the same as the annotated waterfall image's nbr_bin
    LambertianModel: str
        The lambertian model used for intensity correction: sin, sin^2, tan, tan^2
    
    Returns
    -------
    raw_pings: np.array
        The raw image stored as numpy array
    pings: np.array
        The canonical image stored in numpy array
    r: np.array
        The slant range of each bin
    rg: np.array
        The projected ground range of each bin
    rg_bar: np.array
        The resampled ground range of each bin
    canonical_kps: np.array
        The kps coordination in the canonical image in the form of [nbr_ping, nbr_bin] for each point.
    """
    xtf_pings = xtf_data.xtf_sss_ping.parse_file(xtf_file)

    # load xtf_sss_pings as np.array
    starboard = [ping.stbd.pings for ping in xtf_pings]  # (1870, 20816)
    port = [ping.port.pings for ping in xtf_pings]       # (1870, 20816)
    starboard = np.array(starboard)
    port = np.array(port)
    # pings = np.concatenate((np.fliplr(starboard),port), axis=1)   # (1870, 41632)

    time_duration = xtf_pings[0].stbd.time_duration # time duration is 0.22645726799964905s
    rows = starboard.shape[0]
    starboard = starboard.reshape(rows, 1301, -1).mean(2)
    port = port.reshape(rows, 1301, -1).mean(2)
    raw_pings = np.concatenate((np.fliplr(starboard),port), axis=1)

    r = np.linspace(time_duration / len_bins, time_duration, num=len_bins) * xtf_pings[0].sound_vel_  # a list of ranges = n * time interval * sound velocity, shape (20816,1)

    ########################  Image Deconvolution  ###############################

    if deconv == True:
        # Gaussian Beam pattern:
        gamma = 0.02  # beam angualr width
        n = np.ceil(2 * r * np.tanh(gamma/2) * 2.5).astype(int) # list of beam width in pixel, shape (20816,1)
        beam_var = 0.125 * np.power(n, 4) / np.log(2)  # the list of variance of Gaussian distribution, shape (20816,1)
        std_var = np.sqrt(beam_var)
        mu = 0.5 * n
        num_iter = 3

        # discrete beam pattern / convolution kernel, a 20816 long list, while each element contains an kernel with size (n,1)
        psf = [[np.power(norm.cdf(i+1, loc=mu[j], scale=std_var[j]) - norm.cdf(i, mu[j], std_var[j]), 2) for i in range(n[j])] for j in range(len_bins)]

        # deconvolved_pings = np.ndarray(np.shape(pings))
        deconvolved_starboard_pings = np.array([richardson_lucy(starboard[:,j], np.array(psf[j]), num_iter, clip=False) for j in range(len_bins)])
        deconvolved_port_pings = np.array([richardson_lucy(port[:,j], np.array(psf[j]), num_iter, clip=False) for j in range(len_bins)])
        deconvolved_starboard_pings = np.transpose(deconvolved_starboard_pings)
        deconvolved_starboard_pings = np.ceil(deconvolved_starboard_pings).astype(int)
        deconvolved_port_pings  = np.transpose(deconvolved_port_pings)
        deconvolved_port_pings = np.ceil(deconvolved_port_pings).astype(int)
        # pings = np.concatenate((np.fliplr(deconvolved_starboard_pings),deconvolved_port_pings), axis=1)   # (1870, 41632)
    else:
        deconvolved_starboard_pings = starboard
        deconvolved_port_pings = port

    ########################  Intensity Correction  ###############################
    data = np.load(mesh_file)
    V, F, bounds = data['V'], data['F'], data['bounds']
    sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(svp_file)

    # initialize a draper object that will accept sidescan pings
    draper = base_draper.BaseDraper(V, F, bounds, sound_speeds)

    # read height of current AUV position from the draper
    heights = np.array([draper.project_altimeter(ping.pos_) for ping in xtf_pings])  # (1870,1)
    incid_angle = np.array([np.arcsin(height / r) for height in heights])  # (1870, 20816), pixels where intensities are zero are NAN

    # Lambertian Model
    lambert_tan = np.tan(incid_angle) # (1870, 20816)
    lambert_tan_sq = np.square(np.tan(incid_angle))
    lambert_sin = np.sin(incid_angle) # (1870, 20816)
    lambert_sin_sq = np.square(np.sin(incid_angle))

    LambertianModelDict = {
        "tan": lambert_tan,
        "tan_square": lambert_tan_sq,
        "sin": lambert_sin,
        "sin_square": lambert_sin_sq
    }

    lambert = LambertianModelDict.get(LambertianModel)

    # True relectivity
    deconv_inten_starboard_pings = np.divide(deconvolved_starboard_pings, lambert)  # elements divide by NAN are still NAN
    deconv_inten_port_pings = np.divide(deconvolved_port_pings, lambert)

    # Normalize the intensity to [0,1]
    inten_deconv_port_pings = np.nan_to_num(deconv_inten_port_pings)
    inten_deconv_starboard_pings = np.nan_to_num(deconv_inten_starboard_pings)

    ########################  Slant Correction  ###############################

    # convert range distance to ground distnce
    # take the first swath for test, calculate the min and max ground range
    # downsample from incidence angle = 60
    delta_r = r[0]
    delta_y = 2 * delta_r
    rg_start = heights / np.sqrt(3)

    # find the optimal start index for interpolation, mean for norm2 case
    rg_start_pos = rg_start.mean()
    # the un-downsampled ground range list
    rg = np.multiply(r, np.cos(incid_angle))
    rg = np.nan_to_num(rg)

    # downsampled ground range
    rg_ind_max = np.floor((rg[:,-1] - rg_start_pos) / delta_y).astype(int)
    ind_max = rg_ind_max.min()
    rg_bar = np.linspace(0, ind_max, num=ind_max+1) * delta_y +rg_start_pos
    corrected_stbd = []
    corrected_port = []
    # for each ground range bar, find out the [y_{j-1}. y_{j+1}] which covers the [rg_bar - delta_r, rg_bar + delta_r]
    for i in range(len(heights)):
        rg_test = rg[i,:]
        # tic = time.perf_counter()
        stbd_tmp = []
        port_tmp = []
        for y_bar in rg_bar[0:-1]:
            ind_1_tmp = np.argmax(rg_test > (y_bar - delta_r))
            ind_e_tmp = np.argmax(rg_test > (y_bar + delta_r))
            ind_1 = ind_1_tmp if (y_bar - delta_r) > ((rg_test[ind_1_tmp]+rg_test[ind_1_tmp-1])/2) else (ind_1_tmp-1)
            if ind_e_tmp == 0:
                ind_e = len(rg_test)-1
            else:
                ind_e = ind_e_tmp if (y_bar + delta_r) > ((rg_test[ind_e_tmp]+rg_test[ind_e_tmp-1])/2) else (ind_e_tmp-1)
            
            if ind_e == ind_1:
                stbd_tmp.append(inten_deconv_starboard_pings[i, ind_1])
                port_tmp.append(inten_deconv_port_pings[i, ind_1])
            else:
                ind = np.linspace(ind_1, ind_e, num=(ind_e - ind_1 + 1)).astype(int)
                w_front = (rg_test[ind_1] + rg_test[ind[1]])/2 - (y_bar - delta_r)
                w_end = y_bar + delta_r - (rg_test[ind_e] + rg_test[ind[-2]])/2
                w = [(rg_test[j+1] - rg_test[j-1])/2 for j in ind[1:-1]]
                w.insert(0,w_front)
                w.append(w_end)
                stbd_tmp.append(np.multiply(w, inten_deconv_starboard_pings[i, ind]).sum())
                port_tmp.append(np.multiply(w, inten_deconv_port_pings[i, ind]).sum())
        corrected_stbd.append(stbd_tmp)
        corrected_port.append(port_tmp)
        # toc = time.perf_counter()
        # print(f"One ping corrected in {toc - tic:0.4f} seconds")

    pings = np.concatenate((np.fliplr(corrected_stbd),corrected_port), axis=1)
    
    if len(keyPoint):
        kps = keyPoint[0]
        nbr_bins_canonical = int(pings.shape[1] / 2)
        kp_ind1 = np.linspace(len_bins-1, 0, num=len_bins).astype(np.int16)
        kp_ind2 = np.linspace(0, len_bins-1, num=len_bins).astype(np.int16)
        kp_ind = np.concatenate([kp_ind1, kp_ind2])

        flag = np.zeros(kps.shape[0])
        flag[kps[:,1] > len_bins-1] = 1
        divd_matched_kps = kps.copy()
        divd_matched_kps[:,1] = kp_ind[kps[:,1].astype(np.int16)]

        r_g1 = rg[divd_matched_kps[:,0].astype(np.int16), divd_matched_kps[:,1].astype(np.int16)]
        rg_bar_pr = rg_bar + delta_r

        ind1 = np.array([np.argmax(rg_bar_pr > yg) for yg in r_g1])
        canonical_kps = divd_matched_kps
        canonical_kps[flag>0,1] = ind1[flag>0] + nbr_bins_canonical
        canonical_kps[flag==0,1] = nbr_bins_canonical - ind1[flag==0] -1

        return raw_pings, pings, r, rg, rg_bar, canonical_kps
    else:
        return raw_pings, pings, r, rg, rg_bar
