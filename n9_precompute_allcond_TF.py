
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

        if tf_mode == 'ITPC':
            continue
    
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
                #band, freq = 'theta', [2, 10]
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
                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and file_i.find('STATS') == -1 and file_i.find('bi') == -1:
                                file_to_load = file_i
                            else:
                                continue
                        if electrode_recording_type == 'bipolaire':
                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and file_i.find('STATS') == -1 and file_i.find('bi') != -1:
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
######## COMPILATION FUNCTION ########
########################################



def compilation_compute_TF_ITPC(sujet, electrode_recording_type):

    prms = get_params(sujet, electrode_recording_type)

    compute_TF_ITPC(sujet, prms, electrode_recording_type)
    compute_TF_AL(sujet, prms, electrode_recording_type)
    


################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            print(sujet, electrode_recording_type)

            #### TF & ITPC
            # compilation_compute_TF_ITPC(sujet, electrode_recording_type)
            execute_function_in_slurm_bash_mem_choice('n9_precompute_allcond_TF', 'compilation_compute_TF_ITPC', [sujet, electrode_recording_type], '30G')



        