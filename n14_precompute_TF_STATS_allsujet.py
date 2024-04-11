
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
# import cv2

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False








########################################
######## PREP ALLPLOT ANALYSIS ########
########################################



def get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = np.unique(nomenclature_df['Our correspondances'].values)
    lobe_list = np.unique(nomenclature_df['Lobes'].values)

    #### fill dict with anat names
    ROI_dict_count = {}
    ROI_dict_plots = {}
    for i, _ in enumerate(ROI_list):
        ROI_dict_count[ROI_list[i]] = 0
        ROI_dict_plots[ROI_list[i]] = []

    lobe_dict_count = {}
    lobe_dict_plots = {}
    for i, _ in enumerate(lobe_list):
        lobe_dict_count[lobe_list[i]] = 0
        lobe_dict_plots[lobe_list[i]] = []

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, electrode_recording_type)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if electrode_recording_type == 'monopolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        if electrode_recording_type == 'bipolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        #nchan = chan_list_ieeg_csv[0]
        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict_count[ROI_tmp] = ROI_dict_count[ROI_tmp] + 1
            lobe_dict_count[lobe_tmp] = lobe_dict_count[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    #### exclude ROi and Lobes with 0 counts
    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict_count[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict_count[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots










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

    return tf_thresh



# #tf, nchan = tf_plot, n_chan
# def get_tf_stats(tf, pixel_based_distrib):

#     #### thresh data
#     tf_thresh = tf.copy()
#     #wavelet_i = 0
#     for wavelet_i in range(tf.shape[0]):
#         mask = np.logical_or(tf_thresh[wavelet_i, :] < pixel_based_distrib[wavelet_i, 0], tf_thresh[wavelet_i, :] > pixel_based_distrib[wavelet_i, 1])
#         tf_thresh[wavelet_i, mask] = 1
#         tf_thresh[wavelet_i, np.logical_not(mask)] = 0

#     if debug:

#         plt.pcolormesh(tf_thresh)
#         plt.show()

#     #### if empty return
#     if tf_thresh.sum() == 0:

#         return tf_thresh

#     #### thresh cluster
#     tf_thresh = tf_thresh.astype('uint8')
#     nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
#     #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
#     sizes = stats[1:, -1]
#     nb_blobs -= 1
#     min_size = np.percentile(sizes, tf_stats_percentile_cluster)  

#     if debug:

#         plt.hist(sizes, bins=100)
#         plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
#         plt.show()

#     tf_thresh = np.zeros_like(im_with_separated_blobs)
#     for blob in range(nb_blobs):
#         if sizes[blob] >= min_size:
#             tf_thresh[im_with_separated_blobs == blob + 1] = 1

#     if debug:
    
#         time = np.arange(tf.shape[-1])

#         plt.pcolormesh(time, frex, tf, shading='gouraud', cmap='seismic')
#         plt.contour(time, frex, tf_thresh, levels=0, colors='g')
#         plt.yscale('log')
#         plt.show()

#     return tf_thresh


#cond = 'AL'
def precompute_tf_ROI_STATS(ROI, cond, electrode_recording_type):

    print(f'#### COMPUTE TF STATS {ROI} ####', flush=True)

    #### params
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type)
    site_list = ROI_dict_plots[ROI]

    if cond == 'AC':
        stretch_point = stretch_point_TF_ac_resample
    if cond == 'SNIFF':
        stretch_point = stretch_point_TF_sniff_resampled
    if cond == 'AL':
        stretch_point = resampled_points_AL
    
    phase_list = phase_stats[cond]

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}.npy'):
            print('ALREADY COMPUTED', flush=True)
            return
    else:
        if os.path.exists(f'allsujet_{ROI}_tf_STATS_{cond}_bi.npy'):
            print('ALREADY COMPUTED', flush=True)
            return

    #### load baselines
    print('#### LOAD BASELINES ####', flush=True)

    cycle_baseline_tot = 0

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
        chan_i = chan_list_ieeg.index(chan_name)

        if electrode_recording_type == 'monopolaire':
            tf_load = np.load(f'{sujet}_tf_FR_CV.npy')[chan_i,:,:,:]
        else:
            tf_load = np.load(f'{sujet}_tf_FR_CV_bi.npy')[chan_i,:,:,:]

        tf_load_cycle_n = tf_load.shape[0]

        cycle_baseline_tot += tf_load_cycle_n

    os.chdir(path_memmap)
    tf_stretch_baselines = np.memmap(f'{ROI}_{cond}_tf_STATS_baseline_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', 
                            shape=(cycle_baseline_tot, nfrex, stretch_point_TF))
    
    cycle_baseline_i = 0

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
        chan_i = chan_list_ieeg.index(chan_name)

        if electrode_recording_type == 'monopolaire':
            tf_load = np.load(f'{sujet}_tf_FR_CV.npy')[chan_i,:,:,:]
        else:
            tf_load = np.load(f'{sujet}_tf_FR_CV_bi.npy')[chan_i,:,:,:]

        tf_load_cycle_n = tf_load.shape[0]

        tf_stretch_baselines[cycle_baseline_i:(cycle_baseline_i+tf_load_cycle_n),:,:] = tf_load

        cycle_baseline_i += tf_load_cycle_n

    del tf_load

    #### load cond
    print(f'#### LOAD {cond} ####', flush=True)

    cycle_cond_tot = 0

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        if cond == 'AC':
            re_count = get_ac_starts(sujet).shape[0]
        if cond == 'SNIFF':
            re_count = get_sniff_starts(sujet).shape[0]
        if cond == 'AL':
            re_count = 3

        cycle_cond_tot += re_count

    os.chdir(path_memmap)
    tf_stretch_cond = np.memmap(f'{ROI}_{cond}_tf_STATS_cond_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', 
                            shape=(cycle_cond_tot, len(phase_stats[cond]), nfrex, int(stretch_point/len(phase_list))))
    
    cycle_cond_i = 0

    #sujet_i, sujet, chan_name = 1, site_list[1][0], site_list[1][1] 
    for sujet_i, (sujet, chan_name) in enumerate(site_list):

        print_advancement(sujet_i, len(site_list), steps=[25, 50, 75])

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
        chan_i = chan_list_ieeg.index(chan_name)

        if electrode_recording_type == 'monopolaire':
            tf_load = np.load(f'{sujet}_tf_{cond}.npy')[chan_i,:,:,:]
        else:
            tf_load = np.load(f'{sujet}_tf_{cond}_bi.npy')[chan_i,:,:,:]

        tf_load_cycle_n = tf_load.shape[0]

        for phase_i in range(len(phase_stats[cond])):

            start = phase_i * int(stretch_point/len(phase_list))
            stop = phase_i * int(stretch_point/len(phase_list)) + int(stretch_point/len(phase_list))
            tf_stretch_cond[cycle_cond_i:int(cycle_cond_i+tf_load_cycle_n),phase_i,:,:] = tf_load[:,:,start:stop]

        cycle_cond_i += tf_load_cycle_n

    del tf_load

    ######## COMPUTE SURROGATES & STATS ########

    print('SURROGATES', flush=True)

    #### define ncycle
    n_trial_baselines = tf_stretch_baselines.shape[0]
    n_trial_cond = tf_stretch_cond.shape[0]

    pixel_based_distrib = np.zeros((len(phase_list), nfrex, 2))
    tf_shuffle = np.zeros((n_trial_cond, nfrex, int(stretch_point/len(phase_list))))

    #phase_i, phase_name = 1, phase_list[1]
    for phase_i, phase_name in enumerate(phase_list):

        print(f'COMPUTE {cond} {phase_name}', flush=True)

        #### space allocation
        pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)

        #surrogates_i = 0
        for surrogates_i in range(n_surrogates_tf):

            print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

            #### random selection
            draw_indicator = np.random.randint(low=0, high=2, size=n_trial_cond)
            sel_baseline = np.random.randint(low=0, high=n_trial_baselines, size=(draw_indicator == 1).sum())
            sel_cond = np.random.randint(low=0, high=n_trial_cond, size=(draw_indicator == 0).sum())

            #### extract max min
            tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[sel_baseline, :, :]
            tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[sel_cond, phase_i, :, :]

            _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
            
            pixel_based_distrib_i[:, surrogates_i, 0] = _min
            pixel_based_distrib_i[:, surrogates_i, 1] = _max

        min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1)
        # min, max = np.percentile(pixel_based_distrib_i[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib_i[:,:,1], tf_percentile_sel_stats_up, axis=1) 

        if debug:

            tf_nchan = np.median(tf_stretch_cond, axis=0)

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

        pixel_based_distrib[phase_i, :, 0] = min
        pixel_based_distrib[phase_i, :, 1] = max
    
    ######## SAVE ########

    print(f'SAVE {cond}', flush=True)

    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

    if electrode_recording_type == 'monopolaire':
        np.save(f'allsujet_{ROI}_tf_STATS_{cond}.npy', pixel_based_distrib)
    else:
        np.save(f'allsujet_{ROI}_tf_STATS_{cond}_bi.npy', pixel_based_distrib)

    os.chdir(path_memmap)
    os.remove(f'{ROI}_{cond}_tf_STATS_baseline_{electrode_recording_type}.dat')
    os.remove(f'{ROI}_{cond}_tf_STATS_cond_{electrode_recording_type}.dat')





                    
                
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #cond = 'SNIFF'
    for cond in ['AC', 'SNIFF', 'AL']:

        #### load anat
        ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type)    

        #electrode_recording_type = 'bipolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            #ROI = ROI_to_include[1]
            for ROI in ROI_to_include:
    
                # precompute_tf_ROI_STATS(ROI, cond, electrode_recording_type)
                execute_function_in_slurm_bash_mem_choice('n14_precompute_TF_STATS_allsujet', 'precompute_tf_ROI_STATS', [ROI, cond, electrode_recording_type], '30G')

        







