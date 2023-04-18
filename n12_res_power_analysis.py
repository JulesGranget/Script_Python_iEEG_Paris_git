
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






def get_tf_itpc_stretch_allcond(sujet, tf_mode, electrode_recording_type):

    source_path = os.getcwd()

    if electrode_recording_type == 'monopolaire':

        if tf_mode == 'TF':

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)


        elif tf_mode == 'ITPC':
            
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)

    if electrode_recording_type == 'bipolaire':

        if tf_mode == 'TF':

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            with open(f'allcond_{sujet}_tf_stretch_bi.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)


        elif tf_mode == 'ITPC':
            
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            with open(f'allcond_{sujet}_itpc_stretch_bi.pkl', 'rb') as f:
                tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond




########################################
######## PLOT & SAVE TF & ITPC ########
########################################



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

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

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





#n_chan, chan_name = 0, chan_list_ieeg[0]
def save_TF_ITPC_n_chan(sujet, n_chan, chan_name, tf_mode):

    #### load prms
    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)
    
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'TF', 'summary', f'{sujet}_{chan_name}_{chan_loca}.jpeg')):
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'TF', 'summary', f'{sujet}_{chan_name}_{chan_loca}_bi.jpeg')):
            return

    print_advancement(n_chan, len(chan_list_ieeg), steps=[25, 50, 75])

    #### scale
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    vals = np.array([])

    #cond = conditions[0]
    for cond in conditions:

        if electrode_recording_type == 'monopolaire':
            data = np.median(np.load(f'{sujet}_tf_{cond}.npy')[n_chan,:,:,:], axis=0)
        if electrode_recording_type == 'bipolaire':
            data = np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[n_chan,:,:,:], axis=0)

        vals = np.append(vals, data.reshape(-1))

    median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

    vmin = np.median(vals) - median_diff
    vmax = np.median(vals) + median_diff

    del vals

    #### plot
    fig, axs = plt.subplots(ncols=len(conditions))

    if electrode_recording_type == 'monopolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')
    if electrode_recording_type == 'bipolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}_bi')

    fig.set_figheight(5)
    fig.set_figwidth(15)

    #c, cond = 1, conditions[1]
    for c, cond in enumerate(conditions):

        if electrode_recording_type == 'monopolaire':
            tf_plot = np.median(np.load(f'{sujet}_tf_{cond}.npy')[n_chan,:,:,:], axis=0)
        if electrode_recording_type == 'bipolaire':
            tf_plot = np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[n_chan,:,:,:], axis=0)

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
        ax.pcolormesh(time_vec, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        #### stats
        if cond not in ['FR_CV', 'AL']:

            if electrode_recording_type == 'monopolaire':
                pixel_based_distrib = np.load(f'{sujet}_tf_STATS_{cond}.npy')[n_chan,:,:,:]
            else:
                pixel_based_distrib = np.load(f'{sujet}_tf_STATS_{cond}_bi.npy')[n_chan,:,:,:]

            if get_tf_stats(cond, tf_plot, pixel_based_distrib, nfrex).sum() != 0:
                ax.contour(time_vec, frex, get_tf_stats(cond, tf_plot, pixel_based_distrib, nfrex), levels=0, colors='g')

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
    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

    if electrode_recording_type == 'monopolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}.jpeg', dpi=150)
    if electrode_recording_type == 'bipolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_bi.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    del tf_plot
    gc.collect()







#n_chan, chan_name = 0, prms['chan_list_ieeg'][0]
def save_TF_ITPC_n_chan_AL(sujet, n_chan, chan_name, tf_mode):

    cond = 'AL_long'

    #### load prms
    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)
    
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'TF', 'summary', 'AL', f'{sujet}_{chan_name}_{chan_loca}_AL.jpeg')):
            return
    else:
        if os.path.exists(os.path.join(path_results, sujet, 'TF', 'summary', 'AL', f'{sujet}_{chan_name}_{chan_loca}_AL_bi.jpeg')):
            return

    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    #### get AL time
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    df_AL_time = pd.read_excel(f'{sujet}_count_session.xlsx')
    AL_time = []
    for session_i in range(AL_n):
        AL_time.append(int(df_AL_time[f'AL_{session_i+1}'].values[0]))

    #### scale
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    vals = np.array([])

    #session_i = 0
    for session_i in range(AL_n):

        if electrode_recording_type == 'monopolaire':
            data = np.load(f'{sujet}_tf_AL_{session_i+1}.npy')[n_chan,:,:]
        if electrode_recording_type == 'bipolaire':
            data = np.load(f'{sujet}_tf_AL_{session_i+1}_bi.npy')[n_chan,:,:]

        vals = np.append(vals, data.reshape(-1))

    median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

    vmin = np.median(vals) - median_diff
    vmax = np.median(vals) + median_diff

    del vals

    #### plot
    fig, axs = plt.subplots(nrows=AL_n)

    if electrode_recording_type == 'monopolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')
    if electrode_recording_type == 'bipolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}_bi')

    fig.set_figheight(5)
    fig.set_figwidth(10)

    #### for plotting l_gamma down
    for session_i in range(AL_n):

        if electrode_recording_type == 'monopolaire':
            tf_plot = np.load(f'{sujet}_tf_AL_{session_i+1}.npy')[n_chan,:,:]
        if electrode_recording_type == 'bipolaire':
            tf_plot = np.load(f'{sujet}_tf_AL_{session_i+1}_bi.npy')[n_chan,:,:]

        ax = axs[session_i]
        ax.set_title(f'{AL_time[session_i]}s', fontweight='bold', rotation=0)

        time_vec = np.linspace(0, AL_time[session_i], resampled_points_AL)

        #### plot
        ax.pcolormesh(time_vec, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

    #plt.show()

    #### save
    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'AL'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary', 'AL'))

    if electrode_recording_type == 'monopolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_AL.jpeg', dpi=150)
    if electrode_recording_type == 'bipolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_AL_bi.jpeg', dpi=150)

    del tf_plot
    fig.clf()
    plt.close('all')
    gc.collect()








########################################
######## COMPILATION FUNCTION ########
########################################



def compilation_compute_TF_ITPC(sujet, electrode_recording_type):

    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:

        if tf_mode == 'ITPC':
            continue
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(sujet, n_chan, chan_name, tf_mode) for n_chan, chan_name in enumerate(chan_list_ieeg))

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan_AL)(sujet, n_chan, chan_name, tf_mode) for n_chan, chan_name in enumerate(chan_list_ieeg))

    print('done')






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #electrode_recording_type = 'monopolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            print(sujet, electrode_recording_type)

            #### TF & ITPC
            compilation_compute_TF_ITPC(sujet, electrode_recording_type)
            # execute_function_in_slurm_bash_mem_choice('n12_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet, electrode_recording_type], '30G')



        