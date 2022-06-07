
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

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









########################################
######## COMPUTE TF FOR COND ######## 
########################################




#i_to_extract, cond, mat_type, anat_type = ROI_i, 'FR_CV', 'TF', 'ROI'
def open_TForITPC_data(i_to_extract, cond, mat_type, anat_type):

    #### prepare index
    if mat_type == 'ITPC':
        mat_type_i = 0
    if mat_type == 'TF':
        mat_type_i = 1
    
    #### open file
    os.chdir(os.path.join(path_precompute, 'allplot'))
    
    listdir = os.listdir()
    file_to_open = []
    [file_to_open.append(file_i) for file_i in listdir if file_i.find(cond) != -1 and file_i.find(anat_type) != -1]

    #### load matrix
    TF_mat = []
    for file_i in file_to_open:
        mat = np.load(file_i)
        mat_plot_i = mat[i_to_extract, mat_type_i, :, :]
        TF_mat.append(mat_plot_i)

    del mat, mat_plot_i

    #### verif
    #fig, axs = plt.subplots(ncols=len(TF_mat))
    #for i in range(len(TF_mat)):
    #    ax = axs[i]
    #    ax.pcolormesh(TF_mat[i])
    #plt.show()
                
    return TF_mat








#ROI_name, mat_type = 'amygdala', 'TF'
def compute_for_one_ROI_allcond(ROI_name, mat_type):

    print(ROI_name)

    #### params
    anat_type = 'ROI'
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_list[0])

    #### get index 
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()
    ROI_dict_to_open = ROI_dict_plots[ROI_name]
    ROI_i = ROI_to_include.index(ROI_name)

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

      
    #### load mat
    ROI_mat_dict = {}
    for cond in conditions_compute_TF:

        ROI_mat_dict[cond] = {}

        TF_mat = open_TForITPC_data(ROI_i, cond, mat_type, anat_type)
                
        for band_i, band in enumerate(band_names):

            ROI_mat_dict[cond][band] = TF_mat[band_i]



    #### compute number of plot for each cond
    cond_count = {}
    for cond in conditions_compute_TF:
        cond_count[cond] = len(ROI_dict_to_open)
    
    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'TF', 'ROI'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'ITPC', 'ROI'))

    #### find scale
    scales_lf = {}

    for cond in conditions_compute_TF:

        scales_lf[cond] = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        # i, (band, freq) = 0, ('theta', [2 ,10])

        for band in band_names[:4] :

            if band == 'whole':
                continue

            data = ROI_mat_dict[cond][band]

            scales_lf[cond]['vmin_val'] = np.append(scales_lf[cond]['vmin_val'], np.min(data))
            scales_lf[cond]['vmax_val'] = np.append(scales_lf[cond]['vmax_val'], np.max(data))
            scales_lf[cond]['median_val'] = np.append(scales_lf[cond]['median_val'], np.median(data))

        median_diff = np.max( [np.abs(np.min(scales_lf[cond]['vmin_val']) - np.median(scales_lf[cond]['median_val'])), np.abs(np.max(scales_lf[cond]['vmax_val']) - np.median(scales_lf[cond]['median_val'])) ])

    values_scales_lf = {}
    
    for cond in conditions_compute_TF:

        values_scales_lf[cond] = {}
    
        values_scales_lf[cond]['vmin'] = np.median(scales_lf[cond]['median_val']) - median_diff
        values_scales_lf[cond]['vmax'] = np.median(scales_lf[cond]['median_val']) + median_diff


    scales_hf = {} 
               
    for cond in conditions_compute_TF:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        scales_hf[cond] = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        for band in band_names[4:] :

            if band == 'l_gamma':
                continue

            data = ROI_mat_dict[cond][band]

            scales_hf[cond]['vmin_val'] = np.append(scales_hf[cond]['vmin_val'], np.min(data))
            scales_hf[cond]['vmax_val'] = np.append(scales_hf[cond]['vmax_val'], np.max(data))
            scales_hf[cond]['median_val'] = np.append(scales_hf[cond]['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_hf[cond]['vmin_val']) - np.median(scales_hf[cond]['median_val'])), np.abs(np.max(scales_hf[cond]['vmax_val']) - np.median(scales_hf[cond]['median_val']))])

    values_scales_hf = {}
    
    for cond in conditions_compute_TF:

        values_scales_hf[cond] = {}
    
        values_scales_hf[cond]['vmin'] = np.median(scales_hf[cond]['median_val']) - median_diff
        values_scales_hf[cond]['vmax'] = np.median(scales_hf[cond]['median_val']) + median_diff

    del scales_lf, scales_hf

    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        if band_prep == 'lf':
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions_compute_TF))
            band_to_plot = band_names[:4]
        if band_prep == 'hf':
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions_compute_TF))
            band_to_plot = band_names[4:]

        plt.suptitle(ROI_name)
        dict_freq_to_plot = freq_band_list[band_prep_i]

        for cond_i, cond in enumerate(conditions_compute_TF):

            if cond == 'FR_CV':
                stretch_point = stretch_point_TF
                time = range(stretch_point)
            if cond == 'AC':
                stretch_point = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                time = np.linspace(t_start_AC, t_stop_AC, stretch_point)
            if cond == 'SNIFF':
                stretch_point = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                time = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for i, (band, freq) in enumerate(list(dict_freq_to_plot.items())) :

                if band_prep == 'hf' and band in band_names[:4]:
                    continue
                if band_prep == 'lf' and band in band_names[4:]:
                    continue

                data = ROI_mat_dict[cond][band]

                frex = np.linspace(freq[0], freq[1], data.shape[0])
                
            
                ax = axs[i, cond_i]
                if i == 0 :
                    ax.set_title(cond + f' : {cond_count[cond]}')
                if cond_i == 0:
                    ax.set_ylabel(band)
                if band_prep == 'lf':
                    ax.pcolormesh(time, frex, data, vmin=values_scales_lf[cond]['vmin'], vmax=values_scales_lf[cond]['vmax'], shading='gouraud', cmap=plt.get_cmap('seismic'))
                elif band_prep == 'hf' and band == 'l_gamma':
                    ax.pcolormesh(time, frex, data, vmin=values_scales_hf[cond]['vmin'], vmax=values_scales_hf[cond]['vmax'], shading='gouraud', cmap=plt.get_cmap('seismic'))
                else:
                    ax.pcolormesh(time, frex, data, vmin=np.median(data), vmax=data.max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

                if cond == 'AC' or cond == 'SNIFF':
                    ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')
                #plt.show()
                    
        #### save
        if band_prep == 'lf':
            fig.savefig(ROI_name + '_all_lf.jpeg', dpi=150)
        if band_prep == 'hf':
            fig.savefig(ROI_name + '_all_hf.jpeg', dpi=150)
        plt.close()









#Lobe_name, mat_type = 'Insula', 'TF'
def compute_for_one_Lobe_allcond(Lobe_name, mat_type):

    print(Lobe_name)

    #### params
    anat_type = 'Lobes'
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_list[0])

    #### get index 
    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()
    Lobe_dict_to_open = lobe_dict_plots[Lobe_name]
    Lobe_i = lobe_to_include.index(Lobe_name)

    #### extract band names
    band_names = []
    freq_values = []
    for band_freq_i in freq_band_list:
        [band_names.append(band_name_i) for band_name_i in list(band_freq_i.keys())]
        [freq_values.append(freq_values_i) for freq_values_i in list(band_freq_i.values())]

      
    #### load mat
    Lobe_mat_dict = {}
    for cond in conditions_compute_TF:

        Lobe_mat_dict[cond] = {}

        TF_mat = open_TForITPC_data(Lobe_i, cond, mat_type, anat_type)
                
        for band_i, band in enumerate(band_names):

            Lobe_mat_dict[cond][band] = TF_mat[band_i]

    del TF_mat

    #### compute number of plot for each cond
    cond_count = {}
    for cond in conditions_compute_TF:
        cond_count[cond] = len(Lobe_dict_to_open)
    
    #### plot & save
    if mat_type == 'TF':
        os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Lobes'))
    if mat_type == 'ITPC':
        os.chdir(os.path.join(path_results, 'allplot', 'ITPC', 'Lobes'))

    #### find scale
    scales_lf = {}

    for cond in conditions_compute_TF:

        scales_lf[cond] = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        # i, (band, freq) = 0, ('theta', [2 ,10])

        for band in band_names[:4] :

            if band == 'whole':
                continue

            data = Lobe_mat_dict[cond][band]

            scales_lf[cond]['vmin_val'] = np.append(scales_lf[cond]['vmin_val'], np.min(data))
            scales_lf[cond]['vmax_val'] = np.append(scales_lf[cond]['vmax_val'], np.max(data))
            scales_lf[cond]['median_val'] = np.append(scales_lf[cond]['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_lf[cond]['vmin_val']) - np.median(scales_lf[cond]['median_val'])), np.abs(np.max(scales_lf[cond]['vmax_val']) - np.median(scales_lf[cond]['median_val']))])

    values_scales_lf = {}
    
    for cond in conditions_compute_TF:

        values_scales_lf[cond] = {}
    
        values_scales_lf[cond]['vmin'] = np.median(scales_lf[cond]['median_val']) - median_diff
        values_scales_lf[cond]['vmax'] = np.median(scales_lf[cond]['median_val']) + median_diff


    scales_hf = {} 
               
    for cond in conditions_compute_TF:
        # i, (band, freq) = 0, ('theta', [2 ,10])

        scales_hf[cond] = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        for band in band_names[4:] :

            if band == 'l_gamma':
                continue

            data = Lobe_mat_dict[cond][band]

            scales_hf[cond]['vmin_val'] = np.append(scales_hf[cond]['vmin_val'], np.min(data))
            scales_hf[cond]['vmax_val'] = np.append(scales_hf[cond]['vmax_val'], np.max(data))
            scales_hf[cond]['median_val'] = np.append(scales_hf[cond]['median_val'], np.median(data))

        median_diff = np.max([np.abs(np.min(scales_hf[cond]['vmin_val']) - np.median(scales_hf[cond]['median_val'])), np.abs(np.max(scales_hf[cond]['vmax_val']) - np.median(scales_hf[cond]['median_val']))])

    values_scales_hf = {}
    
    for cond in conditions_compute_TF:

        values_scales_hf[cond] = {}
    
        values_scales_hf[cond]['vmin'] = np.median(scales_lf[cond]['median_val']) - median_diff
        values_scales_hf[cond]['vmax'] = np.median(scales_lf[cond]['median_val']) + median_diff

    del scales_lf, scales_hf

    # band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        if band_prep == 'lf':
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions_compute_TF))
            band_to_plot = band_names[:4]
        if band_prep == 'hf':
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions_compute_TF))
            band_to_plot = band_names[4:]

        plt.suptitle(Lobe_name)
        dict_freq_to_plot = freq_band_list[band_prep_i]

        for cond_i, cond in enumerate(conditions_compute_TF):

            if cond == 'FR_CV':
                stretch_point = stretch_point_TF
                time = range(stretch_point)
            if cond == 'AC':
                stretch_point = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                time = np.linspace(t_start_AC, t_stop_AC, stretch_point)
            if cond == 'SNIFF':
                stretch_point = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                time = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point)
                        
            # i, (band, freq) = 0, ('theta', [2 ,10])
            for i, (band, freq) in enumerate(list(dict_freq_to_plot.items())) :

                if band_prep == 'hf' and band in band_names[:4]:
                    continue
                if band_prep == 'lf' and band in band_names[4:]:
                    continue

                data = Lobe_mat_dict[cond][band]

                frex = np.linspace(freq[0], freq[1], data.shape[0])
                
            
                ax = axs[i, cond_i]
                if i == 0 :
                    ax.set_title(cond + f' : {cond_count[cond]}')
                if cond_i == 0:
                    ax.set_ylabel(band)
                if band_prep == 'lf':
                    ax.pcolormesh(time, frex, data, vmin=values_scales_lf[cond]['vmin'], vmax=values_scales_lf[cond]['vmax'], shading='gouraud', cmap=plt.get_cmap('seismic'))
                elif band_prep == 'hf' and band == 'l_gamma':
                    ax.pcolormesh(time, frex, data, vmin=values_scales_hf[cond]['vmin'], vmax=values_scales_hf[cond]['vmax'], shading='gouraud', cmap=plt.get_cmap('seismic'))
                else:
                    ax.pcolormesh(time, frex, data, vmin=np.median(data), vmax=data.max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point, ymin=freq[0], ymax=freq[1], colors='g')

                if cond == 'AC' or cond == 'SNIFF':
                    ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')
                #plt.show()
                    
        #### save
        if band_prep == 'lf':
            fig.savefig(Lobe_name + '_all_lf.jpeg', dpi=150)
        if band_prep == 'hf':
            fig.savefig(Lobe_name + '_all_hf.jpeg', dpi=150)
        plt.close()





################################
######## COMPILATION ########
################################

def compilation_slurm(anat_type, mat_type):

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots()

    if anat_type == 'ROI':

        #joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_ROI_allcond)(ROI_i, mat_type) for ROI_i in ROI_to_include)
        for ROI_i in ROI_to_include:
            compute_for_one_ROI_allcond(ROI_i, mat_type)

    if anat_type == 'Lobe':

        #joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_for_one_Lobe_allcond)(Lobe_i, mat_type) for Lobe_i in lobe_to_include)
        for Lobe_i in lobe_to_include:
            compute_for_one_Lobe_allcond(Lobe_i, mat_type)




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #anat_type = 'ROI'
    for anat_type in ['ROI', 'Lobe']:
    
        #mat_type = 'TF'
        for mat_type in ['TF', 'ITPC']:
            
            #compilation_slurm(anat_type, mat_type)
            execute_function_in_slurm_bash('n13_res_allplot_TF', 'compilation_slurm', [anat_type, mat_type])




