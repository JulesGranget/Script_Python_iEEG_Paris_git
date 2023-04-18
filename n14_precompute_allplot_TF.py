


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

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



def get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type):

    if cond == 'AL_long':
        cond = 'AL'

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
######## TF & ITPC ########
################################





# Lobe_to_process = 'Occipital'
def get_TF_and_ITPC_for_Lobe(Lobe_to_process, cond, electrode_recording_type):

    #### load srate
    srate = get_params(sujet_list[0], electrode_recording_type)['srate']

    #### load anat
    ROI_list_allband, Lobe_list_allband = get_all_ROI_and_Lobes_name()
    len_ROI, len_Lobes = len(list(ROI_list_allband.keys())), len(list(Lobe_list_allband.keys()))
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type)

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
    # dict_ITPC_for_Lobe_to_process = {}
    dict_freq_band = {}
    #freq_band_i, freq_band_dict = 0, freq_band_list[0]
    for freq_band_i, freq_band_dict in enumerate(freq_band_list):
        if freq_band_i == 0:
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex, stretch_point))
                # dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_lf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

        else :
            for band_i in list(freq_band_dict.keys()):
                dict_TF_for_Lobe_to_process[band_i] = np.zeros((nfrex, stretch_point))
                # dict_ITPC_for_Lobe_to_process[band_i] = np.zeros((nfrex_hf, stretch_point))
                dict_freq_band[band_i] = freq_band_list[freq_band_i][band_i]

    #### initiate len recorded
    len_recorded = []
    
    #### compute TF
    # plot_to_process_i = plot_to_process[0]    
    for plot_to_process_i in plot_to_process:
        
        sujet_tmp = plot_to_process_i[0]
        plot_tmp_mod = plot_to_process_i[1]

        #### load subject params
        chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
        chan_list_modified, chan_list_keep = modify_name(chan_list_ieeg)

        #### identify plot name in trc
        if sujet_tmp[:3] != 'pat':
            list_mod, list_trc = modify_name(chan_list_ieeg)
            plot_tmp = list_trc[list_mod.index(plot_tmp_mod)]
        else:
            plot_tmp = plot_tmp_mod

        plot_tmp_i = chan_list_ieeg.index(plot_tmp)

        #### add length recorded
        len_recorded.append(load_data(sujet_tmp, cond, electrode_recording_type)[plot_tmp_i,:].shape[0]/srate/60)

        os.chdir(os.path.join(path_precompute, sujet_tmp, 'TF'))

        #### identify trial number
        band, freq = list(dict_freq_band.items())[0]

        if electrode_recording_type == 'monopolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('bi') == -1])
        if electrode_recording_type == 'bipolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1 and i.find('bi') != -1])

        #### load TF and mean trial
        for band, freq in dict_freq_band.items():
    
            for trial_i in range(n_trials):
                
                if trial_i == 0:

                    if electrode_recording_type == 'monopolaire':
                        TF_load = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy')
                    if electrode_recording_type == 'bipolaire':
                        TF_load = np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy')

                else:

                    if electrode_recording_type == 'monopolaire':
                        TF_load += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy')
                    if electrode_recording_type == 'bipolaire':
                        TF_load += np.load(f'{sujet_tmp}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy')
            
            #### average trials TF and normalize
            TF_load /= n_trials
            TF_load_zscore = zscore_mat(TF_load[plot_tmp_i,:,:])

            dict_TF_for_Lobe_to_process[band] = (dict_TF_for_Lobe_to_process[band] + TF_load_zscore)

        #### load ITPC and mean trial
        # os.chdir(os.path.join(path_precompute, sujet_tmp, 'ITPC'))
        # for band, freq in dict_freq_band.items():
    
        #     for trial_i in range(n_trials):
                
        #         if trial_i == 0:

        #             if electrode_recording_type == 'monopolaire':
        #                 ITPC_load = np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy')
        #             if electrode_recording_type == 'bipolaire':
        #                 ITPC_load = np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy')

        #         else:

        #             if electrode_recording_type == 'monopolaire':
        #                 ITPC_load += np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}.npy')
        #             if electrode_recording_type == 'bipolaire':
        #                 ITPC_load += np.load(f'{sujet_tmp}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{trial_i+1}_bi.npy')
            
        #     #### average trials ITPC
        #     ITPC_load /= n_trials
        #     ITPC_load_zscore = zscore_mat(ITPC_load[plot_tmp_i,:,:])

        #     dict_ITPC_for_Lobe_to_process[band] = (dict_ITPC_for_Lobe_to_process[band] + ITPC_load_zscore)

    #### mean
    for band, freq in dict_freq_band.items():
        dict_TF_for_Lobe_to_process[band] /= len(plot_to_process)
        # dict_ITPC_for_Lobe_to_process[band] /= len(plot_to_process)

    #### if ITPC not computed
    dict_ITPC_for_Lobe_to_process = {}

    return dict_ITPC_for_Lobe_to_process, dict_TF_for_Lobe_to_process










################################
######## COMPILATION ########
################################



def compilation_allplot_analysis(cond, electrode_recording_type):

    #### verify computation
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(f'allsujet_{cond}_ROI.nc'):
            print(f'ALREADY COMPUTED {cond}')
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(f'allsujet_{cond}_ROI_bi.nc'):
            print(f'ALREADY COMPUTED {cond}')
            return

    print(cond)

    #### load anat
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type)
        
    #### identify stretch point
    if cond == 'FR_CV':
        stretch_point = stretch_point_TF
    if cond == 'AC':
        stretch_point = stretch_point_TF_ac_resample
    if cond == 'SNIFF':
        stretch_point = stretch_point_TF_sniff_resampled
    if cond in ['AL', 'AL_long']:
        stretch_point = resampled_points_AL

    #### generate xr
    os.chdir(path_memmap)
    ROI_data_xr = np.memmap(f'allsujet_{cond}_ROI_reduction.dat', dtype=np.float32, mode='w+', shape=(len(ROI_to_include), nfrex, stretch_point))
    
    #### compute TF & ITPC for ROI
    #ROI_to_process = ROI_to_include[1]
    for ROI_to_process in ROI_to_include:

        print(ROI_to_process)

        if cond == 'AL_long':
            tf_allplot = np.zeros((len(ROI_dict_plots[ROI_to_process]),AL_n,nfrex,stretch_point), dtype=np.float32)
        else:
            tf_allplot = np.zeros((len(ROI_dict_plots[ROI_to_process]),nfrex,stretch_point), dtype=np.float32)

        #site_i, (sujet, site) = 0, ROI_dict_plots[ROI_to_process][0]
        for site_i, (sujet, site) in enumerate(ROI_dict_plots[ROI_to_process]):

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

            if cond == 'AL_long':
                for session_i in range(AL_n):
                    if electrode_recording_type == 'monopolaire':
                        tf_allplot[site_i,session_i,:,:] = np.load(f'{sujet}_tf_AL_{session_i+1}.npy')[chan_list_ieeg.index(site),:,:]
                    else:
                        tf_allplot[site_i,session_i,:,:] = np.load(f'{sujet}_tf_AL_{session_i+1}_bi.npy')[chan_list_ieeg.index(site),:,:]

            else:
                if electrode_recording_type == 'monopolaire':
                    tf_allplot[site_i,:,:] = np.median(np.load(f'{sujet}_tf_{cond}.npy')[chan_list_ieeg.index(site),:,:,:], axis=0)
                else:
                    tf_allplot[site_i,:,:] = np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[chan_list_ieeg.index(site),:,:,:], axis=0)

        if cond == 'AL_long':
            ROI_data_xr[ROI_to_include.index(ROI_to_process),:,:] = np.median(np.median(tf_allplot, axis=1), axis=0)
        else:
            ROI_data_xr[ROI_to_include.index(ROI_to_process),:,:] = np.median(tf_allplot, axis=0)
        
        del tf_allplot

        #### verif
        if debug:

            vmin, vmax = np.percentile(tf_allplot[1,:,:].reshape(-1), tf_plot_percentile_scale), np.percentile(tf_allplot[1,:,:].reshape(-1), 100-tf_plot_percentile_scale)
            plt.pcolormesh(tf_allplot[1,:,:], vmin=vmin, vmax=vmax)
            plt.show()

    print('#### TF and ITPC for ROI ####')

    #### extract & save
    os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))
    dict_xr = {'roi' : ROI_to_include, 'nfrex' : np.arange(0, nfrex), 'times' : np.arange(0, stretch_point)}
    xr_export = xr.DataArray(ROI_data_xr, coords=dict_xr.values(), dims=dict_xr.keys())
    if electrode_recording_type == 'monopolaire':
        xr_export.to_netcdf(f'allsujet_{cond}_ROI.nc')
    else:
        xr_export.to_netcdf(f'allsujet_{cond}_ROI_bi.nc')

    os.chdir(path_memmap)
    os.remove(f'allsujet_{cond}_ROI_reduction.dat')

    return


        








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #cond = 'AL_long'
        for cond in ['FR_CV', 'SNIFF', 'AC', 'AL', 'AL_long']:

            # compilation_allplot_analysis(cond, electrode_recording_type)
            execute_function_in_slurm_bash_mem_choice('n14_precompute_allplot_TF', 'compilation_allplot_analysis', [cond, electrode_recording_type], '20G')
        



 