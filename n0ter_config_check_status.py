

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False



################################
######## CHECK STATUS ######## 
################################

def inspect_files(file_to_inspect):

    lidt_dir_i = os.listdir()

    verif = []
    verif_names = []

    for file_i in file_to_inspect:
        if file_i not in lidt_dir_i:
            verif.append(1)
            verif_names.append(file_i)

    if len(verif) != 0:
        return False
    else:
        return True




def check_status(sujet):

    check_status_dict = {}

    check_status_dict['precompute'] = {}

    for precompute_i in ['baselines', 'surrogates', 'TF', 'fc']:

        if precompute_i == 'baselines':

            os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

            file_to_inspect = []

            for band_prep in band_prep_list:

                for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):

                    file_to_inspect.append(f'{sujet}_{band}_baselines.npy')

            token = inspect_files(file_to_inspect)

            check_status_dict['precompute'][precompute_i] = token


        if precompute_i == 'surrogates':

            os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

            file_to_inspect = []

            for cond in ['FR_CV']:

                for band_prep in band_prep_list:

                    file_to_inspect.append(f'{sujet}_{cond}_cyclefreq_{band_prep}.npy')
                    file_to_inspect.append(f'{sujet}_{cond}_Coh.npy')
                    
            token = inspect_files(file_to_inspect)

            check_status_dict['precompute'][precompute_i] = token


        if precompute_i == 'TF':

            file_to_inspect = []

            for cond in conditions_compute_TF:

                for band_prep in band_prep_list:

                    freq_band = freq_band_dict[band_prep] 

                    for band, freq in freq_band.items():

                        file_to_inspect.append(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy')

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            token_TF = inspect_files(file_to_inspect)

            file_to_inspect = []

            for cond in conditions_compute_TF:

                for band_prep in band_prep_list:

                    freq_band = freq_band_dict[band_prep] 

                    for band, freq in freq_band.items():

                        file_to_inspect.append(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy')

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            token_ITPC = inspect_files(file_to_inspect)

            check_status_dict['precompute'][precompute_i] = token_TF and token_ITPC


        if precompute_i == 'fc':

            os.chdir(os.path.join(path_precompute, sujet, 'FC'))

            file_to_inspect = []

            for band_prep_i, band_prep in enumerate(band_prep_list):

                for band, freq in freq_band_dict_FC[band_prep].items():

                    if band == 'whole':
                        continue

                    for cond_i, cond in enumerate(['FR_CV']) :

                        file_to_inspect.append(f'{sujet}_ISPC_{band}_{cond}.npy')
                        file_to_inspect.append(f'{sujet}_PLI_{band}_{cond}.npy')
                    
            token = inspect_files(file_to_inspect)

            check_status_dict['precompute'][precompute_i] = token



    check_status_dict['results'] = {}

    for results_i in ['sniffs', 'fc', 'TF', 'Pxx', 'AL']:

        if results_i == 'sniffs':

            os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))
            
            if len(os.listdir()) != 0:
                token = True
            else:
                token = False

            check_status_dict['results'][results_i] = token


        if results_i == 'fc':
            
            os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'figures'))

            if len(os.listdir()) != 0:
                token = True
            else:
                token = False

            check_status_dict['results'][results_i] = token


        if results_i == 'TF':
            
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))

            if len(os.listdir()) > 1:
                token = True
            else:
                token = False

            check_status_dict['results'][results_i] = token



        if results_i == 'Pxx':
            
            os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

            if len(os.listdir()) != 0:
                token = True
            else:
                token = False

            check_status_dict['results'][results_i] = token


        if results_i == 'AL':
            
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'AL'))

            if len(os.listdir()) != 0:
                token = True
            else:
                token = False

            check_status_dict['results'][results_i] = token

        
    return check_status_dict




if __name__ == '__main__':


    check_status_dict = check_status(sujet)
    print(sujet)
    print(check_status_dict)