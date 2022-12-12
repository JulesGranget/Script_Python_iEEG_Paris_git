

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False









########################################
######## COMPUTE FUNCTIONS ########
########################################



def compute_chunk_AC(sujet, data, ac_starts, srate, electrode_recording_type):

    #### chunk
    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    data_stretch = np.zeros((len(ac_starts), data.shape[0], int(stretch_point_TF_ac)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            data_stretch[start_i, nchan, :] = x[t_start: t_stop]

    return data_stretch




#tf = tf_allchan.copy()
def compute_chunk_SNIFF(sujet, data, sniff_starts, srate, electrode_recording_type):

    #### chunk
    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    data_stretch = np.zeros((len(sniff_starts), data.shape[0], int(stretch_point_TF_sniff)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_stretch[start_i, nchan, :] = x[t_start: t_stop]

    return data_stretch









################################
######## RESPI ANALYSIS ########
################################


def compute_ERP(sujet, electrode_recording_type):

    print('ERP PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)
    prms = get_params(sujet, electrode_recording_type)

    data_stretch_allcond = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        data_stretch_allcond[band_prep] = {}

        #cond = 'FR_CV'
        for cond in ['FR_CV', 'AC', 'SNIFF']:

            #### select data without aux chan
            data = load_data(sujet, cond, electrode_recording_type, band_prep=band_prep)
            data = data[:-3,:]

            #### stretch or chunk
            if cond == 'FR_CV':
                data_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, data, prms['srate'])[0]

            if cond == 'AC':
                ac_starts = get_ac_starts(sujet)
                data_stretch = compute_chunk_AC(sujet, data, ac_starts, prms['srate'], electrode_recording_type)

            if cond == 'SNIFF':
                sniff_starts = get_sniff_starts(sujet)
                data_stretch = compute_chunk_SNIFF(sujet, data, sniff_starts, prms['srate'], electrode_recording_type)

            data_stretch_allcond[band_prep][cond] = data_stretch

    return data_stretch_allcond
            
        

def plot_ERP(sujet, data_stretch_allcond, electrode_recording_type):

    print('ERP PLOT')

    os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))

    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)

    #nchan_i, nchan = len(prms['chan_list'][:-3])-1, prms['chan_list'][:-3][-1]
    for nchan_i, nchan in enumerate(prms['chan_list'][:-3]):

        if nchan == 'nasal':
            chan_loca = 'nasal'    
        else:
            chan_loca = df_loca['ROI'][df_loca['name'] == nchan].values[0]

        #band_prep_i, band_prep = 0, 'lf'
        for band_prep_i, band_prep in enumerate(band_prep_list):

            fig, axs = plt.subplots(nrows=3)

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}')
            if electrode_recording_type == 'bipolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}_bi')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #cond_i, cond = 0, 'FR_CV'
            for cond_i, cond in enumerate(['FR_CV', 'AC', 'SNIFF']):

                data_stretch = data_stretch_allcond[band_prep][cond]

                ax = axs[cond_i]
                ax.set_title(f'{cond} : {data_stretch.shape[0]}', fontweight='bold')

                if cond == 'FR_CV':
                    time_vec = np.arange(stretch_point_TF)

                if cond == 'AC':
                    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                    time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

                if cond == 'SNIFF':
                    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                    time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

                ax.plot(time_vec, data_stretch.mean(axis=0)[nchan_i,:], color='b')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:], color='k', linestyle='--')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:]*-1, color='k', linestyle='--')

                max_plot = np.stack((data_stretch.mean(axis=0)[nchan_i,:], data_stretch.std(axis=0)[nchan_i,:])).max()

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=max_plot*-1, ymax=max_plot, colors='g')
                if cond == 'AC':
                    ax.vlines([0, 10], ymin=max_plot*-1, ymax=max_plot, colors='g')
                if cond == 'SNIFF':
                    ax.vlines(0, ymin=max_plot*-1, ymax=max_plot, colors='g')

            #plt.show()

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

    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            print(sujet, electrode_recording_type)
            data_stretch_allcond = compute_ERP(sujet, electrode_recording_type) 
            plot_ERP(sujet, data_stretch_allcond, electrode_recording_type)



