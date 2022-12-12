
import os
import numpy as np
import matplotlib.pyplot as plt
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



def get_ROI_Lobes_list_and_Plots():

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = list(np.unique(nomenclature_df['Our correspondances'].values))
    lobe_list = list(np.unique(nomenclature_df['Lobes'].values))

    #### fill dict with anat names
    ROI_dict = {}
    ROI_dict_plots = {}

    for ROI_i in ROI_list:
        ROI_dict[ROI_i] = 0
        ROI_dict_plots[ROI_i] = []

    lobe_dict = {}
    lobe_dict_plots = {}

    for lobe_i in lobe_list:
        lobe_dict[lobe_i] = 0
        lobe_dict_plots[lobe_i] = []

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









########################################
######## COMPUTE TF FOR COND ######## 
########################################




def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


#struct_name, cond, mat_type, anat_type = ROI_name, 'FR_CV', 'TF', 'ROI'
def open_TForITPC_data(struct_name, cond, mat_type, anat_type, electrode_recording_type):
    
    #### open file
    os.chdir(os.path.join(path_precompute, 'allplot'))
    
    listdir = os.listdir()
    file_to_open = []

    if electrode_recording_type == 'monopolaire':
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') == -1]
    if electrode_recording_type == 'bipolaire':
        [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1 and file_i.find('bi') != -1]

    #### extract band names
    band_names = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]

    #### load matrix
    xr_TF = xr.open_dataarray(file_to_open[0])
    try:
        struct_xr = xr_TF.loc[struct_name, :, mat_type, :, :]
    except:
        print(f'{struct_name} {cond} not found')
        return 0, 0

    #### identify plot number
    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, electrode_recording_type)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    n_count = 0
    
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if electrode_recording_type == 'monopolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        if electrode_recording_type == 'bipolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        if anat_type == 'ROI':
            n_count += np.sum(plot_loca_df['localisation_corrected'] == struct_name)
        if anat_type == 'Lobes':
            n_count += np.sum(plot_loca_df['lobes_corrected'] == struct_name)

    return struct_xr, n_count







#ROI_name, mat_type = 'amygdala', 'TF'
def compute_for_one_ROI_allcond(ROI_name, mat_type, cond_to_compute, srate, electrode_recording_type):

    print(ROI_name)

    #### params
    anat_type = 'ROI'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(ROI_name, cond, mat_type, anat_type, electrode_recording_type)

    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'ROI'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'ROI'))

    #### plot
    # band_prep_i, band_prep = 1, 'hf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_compute))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(ROI_name)
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{ROI_name} bi')

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_compute):

            #### generate time vec
            if cond == 'FR_CV':
                time_vec = np.arange(stretch_point_TF)

            if cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

            if cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                ax = axs[r, c]
                if r == 0 :
                    ax.set_title(f' {cond} : {TF_count_i}')
                if c == 0:
                    ax.set_ylabel(band)
                    
                ax.pcolormesh(time_vec, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
                if cond == 'AC':
                    ax.vlines([0, 10], ymin=freq[0], ymax=freq[1], colors='g')
                if cond == 'SNIFF':
                    ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')

        #plt.show()
                    
        #### save
        if electrode_recording_type == 'monopolaire':
            fig.savefig(f'{ROI_name}_allcond_{band_prep}.jpeg', dpi=150)
        if electrode_recording_type == 'bipolaire':
            fig.savefig(f'{ROI_name}_allcond_{band_prep}_bi.jpeg', dpi=150)
        
        plt.close('all')









#Lobe_name, mat_type = 'Temporal', 'TF'
def compute_for_one_Lobe_allcond(Lobe_name, mat_type, cond_to_compute, srate, electrode_recording_type):

    print(Lobe_name)

    #### params
    anat_type = 'Lobes'

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

    allcond_TF = {}
    allcond_count = {}
    #cond = 'FR_CV'
    for cond in cond_to_compute:

        allcond_TF[cond], allcond_count[cond] = open_TForITPC_data(Lobe_name, cond, mat_type, anat_type, electrode_recording_type)

    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'TF', 'Lobes'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'ITPC', 'Lobes'))

    #### plot
    # band_prep_i, band_prep = 1, 'hf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### extract band to plot
        freq_band = freq_band_dict[band_prep]

        #### initiate fig
        fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(cond_to_compute))

        fig.set_figheight(10)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        if band_prep == 'hf':
            keys_list_reversed = list(freq_band.keys())
            keys_list_reversed.reverse()
            freq_band_reversed = {}
            for key_i in keys_list_reversed:
                freq_band_reversed[key_i] = freq_band[key_i]
            freq_band = freq_band_reversed

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(Lobe_name)
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{Lobe_name} bi')

        #cond_i, cond = 0, 'FR_CV'
        for c, cond in enumerate(cond_to_compute):

            #### generate time vec
            if cond == 'FR_CV':
                time_vec = np.arange(stretch_point_TF)

            if cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

            if cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for r, (band, freq) in enumerate(list(freq_band.items())) :

                TF_i = allcond_TF[cond].loc[band, :, :].data
                TF_count_i = allcond_count[cond]
                frex = np.linspace(freq[0], freq[1], TF_i.shape[0])
                
                ax = axs[r, c]

                if r == 0 :
                    ax.set_title(f' {cond} : {TF_count_i}')
                if c == 0:
                    ax.set_ylabel(band)

                ax.pcolormesh(time_vec, frex, robust_zscore(TF_i), vmin=-robust_zscore(TF_i).max(), vmax=robust_zscore(TF_i).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
                if cond == 'AC':
                    ax.vlines([0, 10], ymin=freq[0], ymax=freq[1], colors='g')
                if cond == 'SNIFF':
                    ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')

        #plt.show()
                    
        #### save
        if electrode_recording_type == 'monopolaire':
            fig.savefig(f'{Lobe_name} _allcond_{band_prep}.jpeg', dpi=150)
        if electrode_recording_type == 'bipolaire':
            fig.savefig(f'{Lobe_name} _allcond_{band_prep}_bi.jpeg', dpi=150)

        plt.close('all')






################################
######## COMPILATION ########
################################

def compilation_slurm(anat_type, mat_type, electrode_recording_type):

    print(f'#### {anat_type} {mat_type} ####')

    #### verify srate for all sujet
    if np.unique(np.array([get_params(sujet, electrode_recording_type)['srate'] for sujet in sujet_list])).shape[0] == 1:
        srate = get_params(sujet_list[0], electrode_recording_type)['srate']

    cond_to_compute = conditions_compute_TF

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

    if anat_type == 'ROI':

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, mat_type, cond_to_compute, srate, electrode_recording_type) for ROI_i in ROI_to_include)
        # for ROI_i in ROI_to_include:
        #     compute_for_one_ROI_allcond(ROI_i, mat_type, cond_to_compute, srate)

    if anat_type == 'Lobes':

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type, cond_to_compute, srate, electrode_recording_type) for Lobe_i in lobe_to_include)
        # for Lobe_i in lobe_to_include:
        #     compute_for_one_Lobe_allcond(Lobe_i, mat_type, cond_to_compute, srate)




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #anat_type = 'ROI'
        for anat_type in ['ROI', 'Lobes']:
        
            #mat_type = 'TF'
            for mat_type in ['TF', 'ITPC']:
                
                # compilation_slurm(anat_type, mat_type, electrode_recording_type)
                execute_function_in_slurm_bash('n16_res_allplot_TF', 'compilation_slurm', [anat_type, mat_type, electrode_recording_type])




