
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import cv2

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False






########################################
######## FUNCTION ANALYSIS ########
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









def get_tf_stats(cond, tf_plot, pixel_based_distrib, nfrex):

    tf_thresh = np.zeros(tf_plot.shape)

    if cond == 'AC':
        stretch_point = stretch_point_TF_ac_resample
    if cond == 'SNIFF':
        stretch_point = stretch_point_TF_sniff_resampled
    
    phase_list = phase_stats[cond]
    phase_point = int(stretch_point/len(phase_list))

    #phase_i, phase_name = 0, phase_list[0]
    for phase_i, phase_name in enumerate(phase_list):

        start = phase_point * phase_i
        stop = phase_point * phase_i + phase_point

        #wavelet_i = 0
        for wavelet_i in range(nfrex):

            mask = np.logical_or(tf_plot[wavelet_i, start:stop] < pixel_based_distrib[phase_i, wavelet_i, 0], tf_plot[wavelet_i, start:stop] > pixel_based_distrib[phase_i, wavelet_i, 1])
            tf_thresh[wavelet_i, start:stop] = mask*1

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster_allplot)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf_plot.shape[-1])

        plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh





########################################
######## PLOT & SAVE TF & ITPC ########
########################################




#ROI_i, ROI = 1, ROI_to_include[1]
def save_TF_ITPC_ROI(ROI_i, ROI):

    cond_to_plot = [cond for cond in conditions if cond != 'AL']

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', electrode_recording_type)

    ROI_count = [len(ROI_dict_plots[ROI]) for ROI in ROI_to_include]

    #### scale
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
    
    data_allcond = {}
    #cond = cond_to_plot[0]
    for cond in cond_to_plot:

        if electrode_recording_type == 'monopolaire':
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI.nc').loc[ROI,:,:]
        if electrode_recording_type == 'bipolaire':
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI_bi.nc').loc[ROI,:,:]

    vals = np.array([])

    #cond = cond_to_plot[0]
    for cond in cond_to_plot:

        vals = np.append(vals, data_allcond[cond].values.reshape(-1))

    median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

    vmin = np.median(vals) - median_diff
    vmax = np.median(vals) + median_diff

    del vals

    #### plot 
    fig, axs = plt.subplots(ncols=len(cond_to_plot))

    if electrode_recording_type == 'monopolaire':
        plt.suptitle(f'{ROI} count : {ROI_count[ROI_i]}')
    if electrode_recording_type == 'bipolaire':
        plt.suptitle(f'{ROI}_bi count : {ROI_count[ROI_i]}')

    fig.set_figheight(5)
    fig.set_figwidth(15)

    #### for plotting l_gamma down
    #c, cond = 1, cond_to_plot[1]
    for c, cond in enumerate(cond_to_plot):

        ax = axs[c]
        ax.set_title(cond, fontweight='bold', rotation=0)

        #### generate time vec
        if cond == 'FR_CV':
            time_vec = np.arange(stretch_point_TF)

        if cond == 'AC':
            time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac_resample)

        if cond == 'SNIFF':
            time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff_resampled)

        if cond == 'AL':
            time_vec = np.linspace(0, AL_chunk_pre_post_time*2, resampled_points_AL)

        #### plot
        ax.pcolormesh(time_vec, frex, data_allcond[cond].values, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        #### stats
        if cond not in ['AL', 'FR_CV']:
            if electrode_recording_type == 'monopolaire':
                pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}.npy')
            else:
                pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_bi.npy')

            if get_tf_stats(cond, data_allcond[cond].values, pixel_based_distrib, nfrex).sum() != 0:
                ax.contour(time_vec, frex, get_tf_stats(cond, data_allcond[cond].values, pixel_based_distrib, nfrex), levels=0, colors='g')

        if cond == 'FR_CV':
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
        if cond == 'AC':
            ax.vlines([0, AC_length], ymin=frex[0], ymax=frex[-1], colors='g')
        if cond == 'SNIFF':
            ax.vlines(0, ymin=frex[0], ymax=frex[-1], colors='g')
        if cond == 'AL':
            ax.vlines(AL_chunk_pre_post_time, ymin=frex[0], ymax=frex[-1], colors='g')

        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

    #plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF'))

    if electrode_recording_type == 'monopolaire':
        fig.savefig(f'{ROI}.jpeg', dpi=150)
    if electrode_recording_type == 'bipolaire':
        fig.savefig(f'{ROI}_bi.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()







#n_chan, chan_name = 0, prms['chan_list_ieeg'][0]
def save_TF_ITPC_ROI_AL(ROI_i, ROI):

    cond_to_plot = ['AL', 'AL_long']

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', electrode_recording_type)

    ROI_count = [len(ROI_dict_plots[ROI]) for ROI in ROI_to_include]

    #### scale
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
    
    data_allcond = {}
    #cond = cond_to_plot[0]
    for cond in cond_to_plot:

        if electrode_recording_type == 'monopolaire':
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI.nc').loc[ROI,:,:]
        if electrode_recording_type == 'bipolaire':
            data_allcond[cond] = xr.open_dataarray(f'allsujet_{cond}_ROI_bi.nc').loc[ROI,:,:]

    vals = np.array([])

    #cond = cond_to_plot[0]
    for cond in cond_to_plot:

        vals = np.append(vals, data_allcond[cond].values.reshape(-1))

    median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

    vmin = np.median(vals) - median_diff
    vmax = np.median(vals) + median_diff

    del vals

    #### plot 
    fig, axs = plt.subplots(nrows=len(cond_to_plot))

    if electrode_recording_type == 'monopolaire':
        plt.suptitle(f'{ROI} count : {ROI_count[ROI_i]}')
    if electrode_recording_type == 'bipolaire':
        plt.suptitle(f'{ROI}_bi count : {ROI_count[ROI_i]}')

    fig.set_figheight(10)
    fig.set_figwidth(15)

    time_vec = np.linspace(0, AL_chunk_pre_post_time*2, resampled_points_AL)

    #### for plotting l_gamma down
    #c, cond = 1, cond_to_plot[1]
    for c, cond in enumerate(cond_to_plot):

        ax = axs[c]
        ax.set_title(cond, fontweight='bold', rotation=0)

        #### plot
        ax.pcolormesh(time_vec, frex, data_allcond[cond].values, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        if cond == 'AL':
            ax.vlines(AL_chunk_pre_post_time, ymin=frex[0], ymax=frex[-1], colors='g')

        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

    #plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF'))

    if electrode_recording_type == 'monopolaire':
        fig.savefig(f'{ROI}_AL.jpeg', dpi=150)
    if electrode_recording_type == 'bipolaire':
        fig.savefig(f'{ROI}_AL_bi.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()








########################################
######## COMPILATION FUNCTION ########
########################################



def compilation_compute_TF_ITPC(electrode_recording_type):
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:

        if tf_mode == 'ITPC':
            continue
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')

        ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', electrode_recording_type)

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_ROI)(ROI_i, ROI) for ROI_i, ROI in enumerate(ROI_to_include))

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_ROI_AL)(ROI_i, ROI) for ROI_i, ROI in enumerate(ROI_to_include))

    print('done')






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #### TF & ITPC
        compilation_compute_TF_ITPC(electrode_recording_type)
        # execute_function_in_slurm_bash_mem_choice('n12_res_power_analysis', 'compilation_compute_TF_ITPC', [electrode_recording_type], '30G')



        