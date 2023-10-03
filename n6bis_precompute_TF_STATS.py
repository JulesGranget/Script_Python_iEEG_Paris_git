
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import cv2

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False









################################
######## COMPUTE STATS ########
################################



#tf, nchan = tf_plot, n_chan
def get_tf_stats(tf, pixel_based_distrib):

    #### thresh data
    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] < pixel_based_distrib[wavelet_i, 0], tf_thresh[wavelet_i, :] > pixel_based_distrib[wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes, tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf.shape[-1])

        plt.pcolormesh(time, frex, tf, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh




#cond = 'RD_SV'
def precompute_tf_STATS(sujet, cond, electrode_recording_type):

    #### params
    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    print(f'#### COMPUTE TF STATS {sujet} ####', flush=True)

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(f'{sujet}_tf_STATS_{cond}.npy'):
            print('ALL COND ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(f'{sujet}_tf_STATS_{cond}_bi.npy'):
            print('ALL COND ALREADY COMPUTED', flush=True)
            return

    #### params
    if cond == 'AC':
        stretch_point = stretch_point_TF_ac_resample
        phase_list = phase_stats[cond]
    if cond == 'SNIFF':
        stretch_point = stretch_point_TF_sniff_resampled
        phase_list = phase_stats[cond]

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        tf_stretch_baselines = np.load(f'{sujet}_tf_FR_CV.npy', mmap_mode='r')
    else:
        tf_stretch_baselines = np.load(f'{sujet}_tf_FR_CV_bi.npy', mmap_mode='r')

    #### load cond
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if electrode_recording_type == 'monopolaire':
        tf_stretch_alldata = np.load(f'{sujet}_tf_{cond}.npy')
    else:
        tf_stretch_alldata = np.load(f'{sujet}_tf_{cond}_bi.npy')

    os.chdir(path_memmap)
    tf_stretch_cond = np.memmap(f'{sujet}_{cond}_tf_stretch_cond_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', 
                            shape=(len(chan_list_ieeg), len(phase_list), tf_stretch_alldata.shape[1], nfrex, int(stretch_point/len(phase_list))))
    
    for phase_i, phase_name in enumerate(phase_list):

        start = phase_i * int(stretch_point/len(phase_list))
        stop = phase_i * int(stretch_point/len(phase_list)) + int(stretch_point/len(phase_list))
        tf_stretch_cond[:,phase_i,:,:,:] = tf_stretch_alldata[:,:,:,start:stop]

    del tf_stretch_alldata

    ######## COMPUTE SURROGATES & STATS ########

    print('SURROGATES', flush=True)

    pixel_based_distrib = np.memmap(f'{sujet}_{cond}_tf_pixel_surr_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', 
                shape=(len(chan_list_ieeg), len(phase_list), nfrex, 2))

    #phase_i, phase_name = 0, phase_list[0]
    for phase_i, phase_name in enumerate(phase_list):

        print(f'COMPUTE {cond} {phase_name}', flush=True)
            
        #nchan = 40
        def get_min_max_pixel_based_distrib(nchan, phase_i):

            print_advancement(nchan, len(chan_list_ieeg), steps=[25, 50, 75])

            #### define ncycle
            n_trial_baselines = tf_stretch_baselines.shape[1]
            n_trial_cond = tf_stretch_cond.shape[2]

            #### space allocation
            _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
            pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
            tf_shuffle = np.zeros((n_trial_cond, nfrex, int(stretch_point/len(phase_list))))

            #surrogates_i = 0
            for surrogates_i in range(n_surrogates_tf):

                #### random selection
                draw_indicator = np.random.randint(low=0, high=2, size=n_trial_cond)
                sel_baseline = np.random.randint(low=0, high=n_trial_baselines, size=(draw_indicator == 1).sum()) #change for choice
                sel_cond = np.random.randint(low=0, high=n_trial_cond, size=(draw_indicator == 0).sum())

                #### extract max min
                tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
                tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, phase_i, sel_cond, :, :]

                _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                
                pixel_based_distrib_i[:, surrogates_i, 0] = _min
                pixel_based_distrib_i[:, surrogates_i, 1] = _max

            min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1)
            # min, max = np.percentile(pixel_based_distrib_i[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib_i[:,:,1], tf_percentile_sel_stats_up, axis=1) 

            if debug:

                wavelet_i = 0
                # thresh_up = np.median(pixel_based_distrib_i[wavelet_i,:,0], axis=0)
                thresh_up = np.percentile(pixel_based_distrib_i[wavelet_i,:,0], tf_percentile_sel_stats_dw)
                # thresh_down = np.median(pixel_based_distrib_i[wavelet_i,:,1], axis=0) 
                thresh_down = np.percentile(pixel_based_distrib_i[wavelet_i,:,1], tf_percentile_sel_stats_up) 
                count, _, _ = plt.hist(pixel_based_distrib_i[wavelet_i,:,:].reshape(-1), bins=500)
                plt.vlines([thresh_up, thresh_down], ymin=0, ymax=count.max(), color='r')
                plt.show()

                tf_nchan = np.median(tf_stretch_cond[nchan,phase_i,:,:,:], axis=0)

                time = np.arange(tf_nchan.shape[-1])

                plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
                plt.contour(time, frex, get_tf_stats(tf_nchan, min, max), levels=0, colors='g')
                plt.yscale('log')
                plt.show()

                #wavelet_i = 0
                for wavelet_i in range(20):
                    count, _, _ = plt.hist(tf_nchan[wavelet_i, :], bins=500)
                    plt.vlines([min[wavelet_i], max[wavelet_i]], ymin=0, ymax=count.max(), color='r')
                    plt.show()

            pixel_based_distrib[nchan, phase_i, :, 0] = min
            pixel_based_distrib[nchan, phase_i, :, 1] = max
        
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_min_max_pixel_based_distrib)(nchan, phase_i) for nchan, _ in enumerate(chan_list_ieeg))

    ######## SAVE ########

    print(f'SAVE {cond}', flush=True)

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_STATS_{cond}.npy', pixel_based_distrib)
    else:
        np.save(f'{sujet}_tf_STATS_{cond}_bi.npy', pixel_based_distrib)

    os.chdir(path_memmap)
    os.remove(f'{sujet}_{cond}_tf_pixel_surr_{electrode_recording_type}.dat')




                    
                
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    

        #electrode_recording_type = 'monopolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            #cond = 'AC'
            for cond in ['AC', 'SNIFF']:
    
                # precompute_tf_STATS(sujet, cond, electrode_recording_type)
                execute_function_in_slurm_bash_mem_choice('n6bis_precompute_TF_STATS', 'precompute_tf_STATS', [sujet, cond, electrode_recording_type], '30G')

        







