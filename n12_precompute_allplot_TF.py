
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import seaborn as sns

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## ALLPLOT ANATOMY ######## 
########################################

def get_all_ROI_and_Lobes_name():

    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')
    
    #### fill dict with anat names
    anat_loca_dict = {}
    anat_lobe_dict = {}
    anat_loca_list = nomenclature_df['Our correspondances'].values
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(anat_loca_list)):
        anat_loca_dict[anat_loca_list[i]] = {'TF' : {}, 'ITPC' : {}}
        anat_lobe_dict[anat_lobe_list_non_sorted[i]] = {'TF' : {}, 'ITPC' : {}}

    return anat_loca_dict, anat_lobe_dict




########################################
######## PREP ALLPLOT ANALYSIS ########
########################################



def get_ROI_Lobes_list_and_Plots():

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = list(nomenclature_df['Our correspondances'].values)
    lobe_list = []
    [lobe_list.append(lobe_i) for lobe_i in nomenclature_df['Lobes'].values if (lobe_i in lobe_list) == False]

    #### fill dict with anat names
    ROI_dict = {}
    ROI_dict_plots = {}
    lobe_dict = {}
    lobe_dict_plots = {}
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(ROI_list)):
        ROI_dict[ROI_list[i]] = 0
        ROI_dict_plots[ROI_list[i]] = []
        lobe_dict[anat_lobe_list_non_sorted[i]] = 0
        lobe_dict_plots[anat_lobe_list_non_sorted[i]] = []

    #### initiate for cond
    sujet_for_cond = []

    #### search for ROI & lobe that have been counted

    sujet_list_selected = sujet_list

    #sujet_i = sujet_list_selected[0]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict[ROI_tmp] = ROI_dict[ROI_tmp] + 1
            lobe_dict[lobe_tmp] = lobe_dict[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots









################################
######## TF & ITPC ########
################################


# ROI_to_process = 'insula post'
def get_TF_and_ITPC_for_ROI(ROI_to_process, cond, srate):

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))

    #### identify stretch point
    if cond == 'FR_CV':
        stretch_point = stretch_point_TF
    if cond == 'AC':
        stretch_point = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    if cond == 'SNIFF':
        stretch_point = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)

    #### identify if proccessed
    if (ROI_to_process in ROI_to_include) == False:
        return

    # if ROI_to_include.index(ROI_to_process)/len(ROI_to_include) % .2 <= .01:
    #     print('{:.2f}'.format(ROI_to_include.index(ROI_to_process)/len(ROI_to_include)))
    print(ROI_to_process)

    #### plot to compute
    plot_to_process = ROI_dict_plots[ROI_to_process]

    #### identify sujet that participate
    sujet_that_participate = []
    for plot_sujet_i, plot_plot_i in plot_to_process:
        if plot_sujet_i in sujet_that_participate:
            continue
        else:
            sujet_that_participate.append(plot_sujet_i)

    #### generate dict for loading TF
    dict_TF_for_ROI_to_process = {}
    dict_ITPC_for_ROI_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_ITPC_for_ROI_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_ITPC_for_ROI_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    # plot_to_process_i = plot_to_process[0]    
    for plot_to_process_i in plot_to_process:
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)

        #### identify plot name
        plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        len_recorded.append(load_data_sujet(sujet_tmp, 'lf', cond, 0)[plot_tmp_i,:].shape[0]/srate/60)

        #### load TF
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))
        for band, freq in dict_freq_band.items():
                
            TF_load = np.load(sujet_tmp + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '.npy')

            dict_TF_for_ROI_to_process[band] = (dict_TF_for_ROI_to_process[band] + TF_load[plot_tmp_i,:,:])

        #### load ITPC
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'ITPC'))
        for band, freq in dict_freq_band.items():
                
            ITPC_load = np.load(sujet_tmp + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '.npy')

            dict_ITPC_for_ROI_to_process[band] = (dict_ITPC_for_ROI_to_process[band] + ITPC_load[plot_tmp_i,:,:])

    #### mean
    for band, freq in dict_freq_band.items():
        dict_TF_for_ROI_to_process[band] /= len(plot_to_process)
        dict_ITPC_for_ROI_to_process[band] /= len(plot_to_process)
    
    #### fill for allband allcond plotting
    ROI_i_tmp = list(ROI_list_allband.keys()).index(ROI_to_process)

    return dict_ITPC_for_ROI_to_process, dict_TF_for_ROI_to_process







# Lobe_to_process = 'Occipital'
def get_TF_and_ITPC_for_Lobe(Lobe_to_process, cond, srate):

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

    #### identify stretch point
    if cond == 'FR_CV':
        stretch_point = stretch_point_TF
    if cond == 'AC':
        stretch_point = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    if cond == 'SNIFF':
        stretch_point = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)

    #### identify if proccessed
    if (Lobe_to_process in lobe_to_include) != True:
        return

    print(Lobe_to_process)

    #### plot to compute
    plot_to_process = lobe_dict_plots[Lobe_to_process]

    #### identify sujet that participate
    sujet_that_participate = []
    for plot_sujet_i, plot_plot_i in plot_to_process:
        if plot_sujet_i in sujet_that_participate:
            continue
        else:
            sujet_that_participate.append(plot_sujet_i)

    #### generate dict for loading TF
    dict_TF_for_Lobe_to_process = {}
    dict_ITPC_for_Lobe_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    # plot_to_process_i = plot_to_process[0]    
    for plot_to_process_i in plot_to_process:
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects)

        #### identify plot name in trc
        if sujet_tmp[:3] != 'pat':
            list_mod, list_trc = modify_name(chan_list_ieeg)
            plot_tmp = list_trc[list_mod.index(plot_tmp_mod)]
        else:
            plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        len_recorded.append(load_data_sujet(sujet_tmp, 'lf', cond, 0)[plot_tmp_i,:].shape[0]/srate/60)

        #### load TF
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))
        for band, freq in dict_freq_band.items():
                
            TF_load = np.load(sujet_tmp + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '.npy')

            dict_TF_for_Lobe_to_process[band] = (dict_TF_for_Lobe_to_process[band] + TF_load[plot_tmp_i,:,:])

        #### load ITPC
        os.chdir(os.path.join(path_precompute, sujet_tmp, 'ITPC'))
        for band, freq in dict_freq_band.items():
                
            ITPC_load = np.load(sujet_tmp + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '.npy')

            dict_ITPC_for_Lobe_to_process[band] = (dict_ITPC_for_Lobe_to_process[band] + ITPC_load[plot_tmp_i,:,:])   

    #### mean
    for band, freq in dict_freq_band.items():
        dict_TF_for_Lobe_to_process[band] /= len(plot_to_process)
        dict_ITPC_for_Lobe_to_process[band] /= len(plot_to_process)


    return dict_ITPC_for_Lobe_to_process, dict_TF_for_Lobe_to_process

















################################
######## COMPILATION ########
################################



def compilation_allplot_analysis(cond):

    #### verify computation
    os.chdir(os.path.join(path_precompute, 'allplot'))
    band = list(freq_band_list[0].keys())[0]
    if os.path.exists(f'ROI_{cond}_{band}_allband.npy') and os.path.exists(f'Lobes_{cond}_{band}_allband.npy'):
        return

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

    #### check srate
    for sujet_i, sujet in enumerate(sujet_list):
        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
        if sujet_i == 0:
            srate_check = srate
        else:
            if srate == srate_check:
                continue
            else:
                raise ValueError('Not the same srate in all sujets')

    srate = srate_check

    #### identify stretch point
    if cond == 'FR_CV':
        stretch_point = stretch_point_TF
    if cond == 'AC':
        stretch_point = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    if cond == 'SNIFF':
        stretch_point = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    
    #### compute TF & ITPC for ROI
    print('#### TF and ITPC for ROI ####')
    res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_TF_and_ITPC_for_ROI)(ROI_to_process, cond, srate) for ROI_to_process in ROI_to_include)

    #### extract & save
    os.chdir(os.path.join(path_precompute, 'allplot'))
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        for band in list(freq_band_dict.keys()):

            ROI_band = np.zeros((len(ROI_to_include), 2, nfrex_hf, stretch_point))

            for ROI_to_process_i, ROI_to_process in enumerate(ROI_to_include):

                ROI_band[ROI_to_process_i, 0, :, :] = res[ROI_to_process_i][0][band]
                ROI_band[ROI_to_process_i, 1, :, :] = res[ROI_to_process_i][1][band]

            np.save(f'ROI_{cond}_{band}_allband.npy', ROI_band)


    #### compute TF & ITPC for Lobes
    print('#### TF and ITPC for Lobe ####')
    res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_TF_and_ITPC_for_Lobe)(Lobe_to_process, cond, srate) for Lobe_to_process in lobe_to_include)

    #### extract
    os.chdir(os.path.join(path_precompute, 'allplot'))
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        for band in list(freq_band_dict.keys()):

            Lobe_band = np.zeros((len(lobe_to_include), 2, nfrex_hf, stretch_point))

            for Lobe_to_process_i, Lobe_to_process in enumerate(lobe_to_include):

                Lobe_band[Lobe_to_process_i, 0, :, :] = res[Lobe_to_process_i][0][band]
                Lobe_band[Lobe_to_process_i, 1, :, :] = res[Lobe_to_process_i][1][band]

            np.save(f'Lobes_{cond}_{band}_allband.npy', Lobe_band)

    print('done')

    

        








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #cond = 'FR_CV'
    for cond in conditions_compute_TF:

        # compilation_allplot_analysis(cond)
        execute_function_in_slurm_bash('n12_precompute_allplot_TF', 'compilation_allplot_analysis', [cond])
    



