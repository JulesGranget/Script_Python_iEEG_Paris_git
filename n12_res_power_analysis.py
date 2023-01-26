
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







################################
######## LOAD TF & ITPC ########
################################


def compute_TF_ITPC(sujet, prms, electrode_recording_type):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if electrode_recording_type == 'monopolaire':
                if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_stretch.pkl')):
                    print('ALREADY COMPUTED')
                    continue
            if electrode_recording_type == 'bipolaire':
                if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_stretch_bi.pkl')):
                    print('ALREADY COMPUTED')
                    continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if electrode_recording_type == 'monopolaire':
                if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'allcond_{sujet}_itpc_stretch.pkl')):
                    print('ALREADY COMPUTED')
                    continue
            if electrode_recording_type == 'bipolaire':
                if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'allcond_{sujet}_itpc_stretch_bi.pkl')):
                    print('ALREADY COMPUTED')
                    continue

        #### load file with reducing to one TF
        tf_stretch_allcond = {}

        #band_prep = 'lf'
        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            #### chose nfrex
            if band_prep == 'lf':
                nfrex = nfrex_lf
            elif band_prep == 'hf':
                nfrex = nfrex_hf

            #cond = 'FR_CV'
            for cond in conditions_compute_TF:

                tf_stretch_allcond[band_prep][cond] = {}

                #### impose good order in dict
                for band, freq in freq_band_dict[band_prep].items():

                    if cond == 'FR_CV':
                        _stretch_point_cond = stretch_point_TF

                    elif cond == 'AC':
                        _stretch_point_cond = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

                    elif cond == 'SNIFF':
                        _stretch_point_cond = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                    
                    tf_stretch_allcond[band_prep][cond][band] = np.zeros(( len(prms['chan_list_ieeg']), nfrex, _stretch_point_cond ))

                #### load file
                for band, freq in freq_band_dict[band_prep].items():
                    
                    for file_i in os.listdir(): 
                        if electrode_recording_type == 'monopolaire':
                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and file_i.find('bi') == -1:
                                file_to_load = file_i
                            else:
                                continue
                        if electrode_recording_type == 'bipolaire':
                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and file_i.find('bi') != -1:
                                file_to_load = file_i
                            else:
                                continue

                    tf_stretch_allcond[band_prep][cond][band] += np.load(file_to_load)

        #### verif
        for band_prep in band_prep_list:
            for cond in conditions_compute_TF:
                for band, freq in freq_band_dict[band_prep].items():
                    if len(tf_stretch_allcond[band_prep][cond][band]) != len(prms['chan_list_ieeg']):
                        raise ValueError(f'reducing incorrect : {band_prep} {cond} {band}')
               
        #### save
        if electrode_recording_type == 'monopolaire':
            if tf_mode == 'TF':
                with open(f'allcond_{sujet}_tf_stretch.pkl', 'wb') as f:
                    pickle.dump(tf_stretch_allcond, f)
            elif tf_mode == 'ITPC':
                with open(f'allcond_{sujet}_itpc_stretch.pkl', 'wb') as f:
                    pickle.dump(tf_stretch_allcond, f)
        if electrode_recording_type == 'bipolaire':
            if tf_mode == 'TF':
                with open(f'allcond_{sujet}_tf_stretch_bi.pkl', 'wb') as f:
                    pickle.dump(tf_stretch_allcond, f)
            elif tf_mode == 'ITPC':
                with open(f'allcond_{sujet}_itpc_stretch_bi.pkl', 'wb') as f:
                    pickle.dump(tf_stretch_allcond, f)

    print('done')





def compute_TF_AL(sujet, prms, electrode_recording_type):

    print('######## LOAD TF ########')
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_AL_stretch.pkl')):
            print('ALREADY COMPUTED')
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_AL_stretch_bi.pkl')):
            print('ALREADY COMPUTED')
            return

    #### identify n_session
    cond = 'AL'

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    n_session = len([file_i for file_i in os.listdir() if file_i.find(cond) != -1 and file_i.find('bi') == -1])
            
    #### load file with reducing to one TF
    tf_stretch_allcond = {}

    #band_prep = 'lf'
    for band_prep in band_prep_list:

        tf_stretch_allcond[band_prep] = {}

        #### chose nfrex
        if band_prep == 'lf':
            nfrex = nfrex_lf
        elif band_prep == 'hf':
            nfrex = nfrex_hf

        #### impose good order in dict
        for band, freq in freq_band_dict[band_prep].items():
            
            tf_stretch_allcond[band_prep][band] = np.zeros(( len(prms['chan_list_ieeg']), nfrex, resampled_points_AL ))

        #### load file
        for band, freq in freq_band_dict[band_prep].items():

            for session_i in range(n_session):
            
                if electrode_recording_type == 'monopolaire':
                    for file_i in os.listdir(): 
                        if file_i.find(f'{freq[0]}_{freq[1]}_{cond}_{str(session_i+1)}') != -1 and file_i.find('bi') == -1:
                            file_to_load = file_i
                        else:
                            continue
                if electrode_recording_type == 'bipolaire':
                    for file_i in os.listdir(): 
                        if file_i.find(f'{freq[0]}_{freq[1]}_{cond}_{str(session_i+1)}') != -1 and file_i.find('bi') != -1:
                            file_to_load = file_i
                        else:
                            continue

                tf_stretch_allcond[band_prep][band] += np.load(file_to_load)

        #### mean
        for band, freq in freq_band_dict[band_prep].items():

            tf_stretch_allcond[band_prep][band] /= n_session

    #### verif
    for band_prep in band_prep_list:
        for cond in conditions_compute_TF:
            for band, freq in freq_band_dict[band_prep].items():
                if len(tf_stretch_allcond[band_prep][band]) != len(prms['chan_list_ieeg']):
                    raise ValueError(f'reducing incorrect : {band_prep} {band}')
            
    #### save
    if electrode_recording_type == 'monopolaire':
        with open(f'allcond_{sujet}_tf_AL_stretch.pkl', 'wb') as f:
            pickle.dump(tf_stretch_allcond, f)
    if electrode_recording_type == 'bipolaire':
        with open(f'allcond_{sujet}_tf_AL_stretch_bi.pkl', 'wb') as f:
            pickle.dump(tf_stretch_allcond, f)

    print('done')




########################################
######## PLOT & SAVE TF & ITPC ########
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






#n_chan, tf_mode, band_prep = 0, 'TF', 'lf'
def save_TF_ITPC_n_chan(sujet, n_chan, tf_mode, band_prep):

    #### load prms
    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)

    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))
    
    chan_name = prms['chan_list_ieeg'][n_chan]
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    freq_band = freq_band_dict[band_prep]

    #### scale
    vmaxs = {}
    vmins = {}
    for cond in conditions_compute_TF:

        scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode, electrode_recording_type)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

            del data

        median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

        vmin = np.median(scales['median_val']) - median_diff
        vmax = np.median(scales['median_val']) + median_diff

        vmaxs[cond] = vmax
        vmins[cond] = vmin

    #### plot
    fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(conditions_compute_TF))
    if electrode_recording_type == 'monopolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')
    if electrode_recording_type == 'bipolaire':
        plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}_bi')

    fig.set_figheight(10)
    fig.set_figwidth(10)

    #### for plotting l_gamma down
    if band_prep == 'hf':
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(conditions_compute_TF):

        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode, electrode_recording_type)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions_compute_TF) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            #### generate time vec
            if cond == 'FR_CV':
                time_vec = np.arange(stretch_point_TF)

            if cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

            if cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

            # ax.pcolormesh(time_vec, frex, data, vmin=vmins[cond], vmax=vmaxs[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
            # ax.pcolormesh(time_vec, frex, data, shading='gouraud', cmap=plt.get_cmap('seismic'))
            # ax.pcolormesh(time_vec, frex, zscore(data), shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.pcolormesh(time_vec, frex, rscore(data), vmin=-rscore(data).max(), vmax=rscore(data).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            if cond == 'FR_CV':
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            if cond == 'AC':
                ax.vlines([0, 10], ymin=freq[0], ymax=freq[1], colors='g')
            if cond == 'SNIFF':
                ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')

            del data

    #plt.show()

    #### save
    if electrode_recording_type == 'monopolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    if electrode_recording_type == 'bipolaire':
        fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}_bi.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()







########################################
######## COMPILATION FUNCTION ########
########################################



def compilation_compute_TF_ITPC(sujet, electrode_recording_type):

    prms = get_params(sujet, electrode_recording_type)

    compute_TF_ITPC(sujet, prms, electrode_recording_type)
    compute_TF_AL(sujet, prms, electrode_recording_type)
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')
        
        #band_prep = 'lf'
        for band_prep in band_prep_list: 

            print(band_prep)

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(sujet, n_chan, tf_mode, band_prep) for n_chan, tf_mode, band_prep in zip(range(len(prms['chan_list_ieeg'])), [tf_mode]*len(prms['chan_list_ieeg']), [band_prep]*len(prms['chan_list_ieeg'])))

    print('done')






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #electrode_recording_type = 'bipolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            print(sujet, electrode_recording_type)

            #### TF & ITPC
            compilation_compute_TF_ITPC(sujet, electrode_recording_type)
            # execute_function_in_slurm_bash_mem_choice('n12_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet, electrode_recording_type], '30G')



        