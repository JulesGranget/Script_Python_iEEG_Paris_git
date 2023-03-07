

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import gc
import sklearn
from sklearn import linear_model

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False









########################################
######## COMPUTE FUNCTIONS ########
########################################




def compute_chunk_FR_CV(sujet, data, srate):

    #### get resp features
    resp_features = load_respfeatures(sujet)
    sniff_starts = resp_features['FR_CV'][0]['inspi_index'].values 

    #### verify chunk at beginning and end
    if int(sniff_starts[0] + t_start_SNIFF*srate) < 0:
        sniff_starts = sniff_starts[1:]

    if int(sniff_starts[-1] + t_stop_SNIFF*srate) > data.shape[-1]:
        sniff_starts = sniff_starts[:-1]

    #### chunk
    chunk_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    data_stretch = np.zeros((len(sniff_starts), data.shape[0], int(chunk_point_TF_sniff)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(sniff_starts[1:]):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_stretch[start_i, nchan, :] = zscore(x[t_start: t_stop])

    #### inspect
    if debug:

        plt.plot(data_stretch.mean(axis=0)[0,:])
        plt.show()

    return data_stretch




def compute_chunk_AC(data, ac_starts, srate):

    #### chunk
    chunk_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)

    data_stretch = np.zeros((len(ac_starts), data.shape[0], int(chunk_point_TF_ac)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            data_stretch[start_i, nchan, :] = zscore(x[t_start: t_stop])

    #### inspect
    if debug:

        plt.plot(data_stretch.mean(axis=0)[0,:])
        plt.show()

    return data_stretch




def compute_chunk_SNIFF(data, sniff_starts, srate):

    #### chunk
    chunk_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    data_stretch = np.zeros((len(sniff_starts), data.shape[0], int(chunk_point_TF_sniff)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_stretch[start_i, nchan, :] = zscore(x[t_start: t_stop])

    #### inspect
    if debug:

        plt.plot(data_stretch.mean(axis=0)[0,:])
        plt.show()

    return data_stretch



def compute_lm_on_ERP(sujet, data_stretch_allcond, electrode_recording_type):

    srate = get_params(sujet, electrode_recording_type)['srate']

    lm_data = {}

    for band_prep in data_stretch_allcond:

        lm_data[band_prep] = {}

        for cond in ['FR_CV', 'SNIFF']:
            
            lm_data[band_prep][cond] = {}
            lm_data[band_prep][cond]['coeff'] = np.zeros((2,data_stretch_allcond[band_prep][cond].shape[1]))

            time_vec_lm = np.arange(SNIFF_lm_time[0], SNIFF_lm_time[-1], 1/srate)
            lm_data[band_prep][cond]['Y_pred'] = np.zeros((data_stretch_allcond[band_prep][cond].shape[1],time_vec_lm.shape[0]))

            for nchan in range(data_stretch_allcond[band_prep][cond].shape[1]):
            
                data = data_stretch_allcond[band_prep][cond][:, nchan, :].mean(axis=0)
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, data.shape[0])
                time_vec_mask = (time_vec > SNIFF_lm_time[0]) & (time_vec < SNIFF_lm_time[-1])
                Y = data[time_vec_mask].reshape(-1,1)
                X = time_vec[time_vec_mask].reshape(-1,1)

                lm = linear_model.LinearRegression()

                lm.fit(X, Y)

                Y_pred = lm.predict(X)

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                lm_data[band_prep][cond]['coeff'][0, nchan] = np.round(r_value**2, 5)
                lm_data[band_prep][cond]['coeff'][1, nchan] = np.round(p_value, 5)

                lm_data[band_prep][cond]['Y_pred'][nchan, :] = Y_pred.reshape(-1)

                #### verif
                if debug:
        
                    plt.plot(X, Y)
                    plt.plot(X, Y_pred, color="b", linewidth=3)

                    plt.show()

    return lm_data




################################
######## RESPI ANALYSIS ########
################################


def compute_ERP(sujet, electrode_recording_type):

    print('ERP PRECOMPUTE')

    srate = get_params(sujet, electrode_recording_type)['srate']

    data_stretch_allcond = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        data_stretch_allcond[band_prep] = {}

        #cond = 'FR_CV'
        for cond in ['FR_CV', 'AC', 'SNIFF']:

            #### select data without aux chan
            data = load_data(sujet, cond, electrode_recording_type, band_prep=band_prep)
            data = data[:-3,:]

            #### chunk
            if cond == 'FR_CV':
                data_stretch = compute_chunk_FR_CV(sujet, data, srate)

            if cond == 'AC':
                ac_starts = get_ac_starts(sujet)
                data_stretch = compute_chunk_AC(data, ac_starts, srate)

            if cond == 'SNIFF':
                sniff_starts = get_sniff_starts(sujet)
                data_stretch = compute_chunk_SNIFF(data, sniff_starts, srate)

            data_stretch_allcond[band_prep][cond] = data_stretch

    return data_stretch_allcond
            
        

def plot_ERP(sujet, data_stretch_allcond, lm_data, electrode_recording_type):

    print('ERP PLOT')

    os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))

    cond_to_plot = ['AC', 'FR_CV', 'SNIFF']

    prms = get_params(sujet, electrode_recording_type)
    srate = prms['srate']
    df_loca = get_loca_df(sujet, electrode_recording_type)

    #nchan_i, nchan = 0, prms['chan_list'][:-3][0]
    for nchan_i, nchan in enumerate(prms['chan_list'][:-3]):

        if nchan == 'nasal':
            chan_loca = 'nasal'    
        else:
            chan_loca = df_loca['ROI'][df_loca['name'] == nchan].values[0]

        #band_prep_i, band_prep = 0, 'lf'
        for band_prep_i, band_prep in enumerate(band_prep_list):

            fig, axs = plt.subplots(nrows=len(cond_to_plot))

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}')
            if electrode_recording_type == 'bipolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}_bi')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #cond_i, cond = 1, 'SNIFF'
            for cond_i, cond in enumerate(cond_to_plot):

                data_stretch = data_stretch_allcond[band_prep][cond]

                ax = axs[cond_i]

                if cond in ['FR_CV', 'SNIFF']:
                    ax.set_title(f"{cond} : {data_stretch.shape[0]} / r2, pval : {lm_data[band_prep][cond]['coeff'][0,nchan_i]}, {lm_data[band_prep][cond]['coeff'][1,nchan_i]}", fontweight='bold')

                if cond == 'AC':
                    ax.set_title(f"{cond} : {data_stretch.shape[0]}", fontweight='bold')

                if cond == 'AC':
                    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                    time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

                if cond in ['FR_CV', 'SNIFF']:
                    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                    time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

                ax.plot(time_vec, data_stretch.mean(axis=0)[nchan_i,:], color='b')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:], color='k', linestyle='--')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:]*-1, color='k', linestyle='--')

                if cond in ['FR_CV', 'SNIFF']:
                    ax.invert_yaxis()

                max_plot = np.stack((data_stretch.mean(axis=0)[nchan_i,:], data_stretch.std(axis=0)[nchan_i,:])).max()

                if cond == 'AC':
                    ax.vlines([0, 12], ymin=max_plot*-1, ymax=max_plot, colors='g')

                if cond in ['FR_CV', 'SNIFF']:
                    ax.vlines(0, ymin=max_plot*-1, ymax=max_plot, colors='g')

                    time_vec_lm = np.arange(SNIFF_lm_time[0], SNIFF_lm_time[-1], 1/srate)

                    ax.plot(time_vec_lm, lm_data[band_prep][cond]['Y_pred'][nchan_i,:], color='r', linewidth=3)

            # plt.show()

            #### save
            if electrode_recording_type == 'monopolaire':
                fig.savefig(f'{sujet}_{nchan}_{chan_loca}_{band_prep}.jpeg', dpi=150)
            if electrode_recording_type == 'bipolaire':
                fig.savefig(f'{sujet}_{nchan}_{chan_loca}_{band_prep}_bi.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[-1]
        for sujet in sujet_list:

            print(sujet, electrode_recording_type)
            
            data_stretch_allcond = compute_ERP(sujet, electrode_recording_type) 
            lm_data = compute_lm_on_ERP(sujet, data_stretch_allcond, electrode_recording_type)
            plot_ERP(sujet, data_stretch_allcond, lm_data, electrode_recording_type)



