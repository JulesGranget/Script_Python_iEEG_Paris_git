
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr
import joblib
import mne_connectivity
import copy

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False




################################
######## CLEAN DATA ########
################################



def clean_data(allband_data, allpairs):

    #### identify pairs to clean
    mask_keep = []

    for pair_i, pair in enumerate(allpairs):

        if pair.split('-')[0] in ROI_for_DFC_plot and pair.split('-')[-1] in ROI_for_DFC_plot:

            mask_keep.append(True)

        else:

            mask_keep.append(False)

    mask_keep = np.array(mask_keep)

    #### clean pairs
    allpairs = allpairs[mask_keep]

    if debug:

        allpairs[~mask_keep]

    #### clean data
    #band_i = 'beta'
    for band_i in allband_data:

        for cond in conditions:

            allband_data[band_i][cond] = allband_data[band_i][cond][:, mask_keep, :, :]

    return allband_data, allpairs



    





########################################
######## ANALYSIS FUNCTIONS ########
########################################




def get_pair_unique_and_roi_unique(pairs):

    #### pairs unique
    pair_unique = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] == pair_i.split('-')[-1]:
            continue
        if f"{pair_i.split('-')[-1]}-{pair_i.split('-')[0]}" in pair_unique:
            continue
        if pair_i not in pair_unique:
            pair_unique.append(pair_i)

    pair_unique = np.array(pair_unique)

    #### get roi in data
    roi_in_data = []

    for pair_i in np.unique(pairs):
        if pair_i.split('-')[0] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[0])

        if pair_i.split('-')[-1] not in roi_in_data:
            roi_in_data.append(pair_i.split('-')[-1])

    roi_in_data = np.array(roi_in_data)

    return pair_unique, roi_in_data





def get_phase_extraction():

    phase_extraction_time = {}

    for cond in conditions:

        if cond == 'FR_CV':
            phase_extraction_time[cond] = {}
            phase_extraction_time[cond]['whole'] = [0, stretch_point_TF]

        if cond == 'AL':
            phase_extraction_time[cond] = {}
            phase_extraction_time[cond]['resp_evnmt_1'] = [0, int(n_points_AL_interpolation/n_phase_extraction_AL)]
            phase_extraction_time[cond]['resp_evnmt_2'] = [int(n_points_AL_interpolation/n_phase_extraction_AL), int(n_points_AL_interpolation/n_phase_extraction_AL)*2]
            phase_extraction_time[cond]['whole'] = [0, n_points_AL_interpolation]

        if cond == 'AC':
            stretch_point_TF_ac = int(np.abs(t_start_AC)*dw_srate_fc_AC +  t_stop_AC*dw_srate_fc_AC)
            time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
            time_vec_i = np.arange(time_vec.shape[0])

            phase_extraction_time[cond] = {}
            phase_extraction_time[cond]['pre'] = [time_vec_i[time_vec <= AC_extract_pre[-1]][0], time_vec_i[time_vec <= AC_extract_pre[-1]][-1]]
            phase_extraction_time[cond]['resp_evnmt_1'] = [time_vec_i[(time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[-1])][0], time_vec_i[(time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[-1])][-1]]
            phase_extraction_time[cond]['resp_evnmt_2'] = [time_vec_i[(time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[-1])][0], time_vec_i[(time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[-1])][-1]]
            phase_extraction_time[cond]['post'] = [time_vec_i[time_vec >= AC_extract_post[0]][0], time_vec_i[time_vec >= AC_extract_post[0]][-1]]
            phase_extraction_time[cond]['whole'] = [0, time_vec_i[time_vec >= AC_extract_post[0]][-1]]

        if cond == 'SNIFF':
            stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
            time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
            time_vec_i = np.arange(time_vec.shape[0])

            phase_extraction_time[cond] = {}
            phase_extraction_time[cond]['pre'] = [time_vec_i[time_vec <= sniff_extract_pre[-1]][0], time_vec_i[time_vec <= sniff_extract_pre[-1]][-1]]
            phase_extraction_time[cond]['resp_evnmt'] = [time_vec_i[(time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[-1])][0], time_vec_i[(time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[-1])][-1]]
            phase_extraction_time[cond]['post'] = [time_vec_i[time_vec >= sniff_extract_post[0]][0], time_vec_i[time_vec >= sniff_extract_post[0]][-1]]
            phase_extraction_time[cond]['whole'] = [0, time_vec_i[time_vec >= sniff_extract_post[0]][-1]]

    return phase_extraction_time








#dfc_data = allband_data[band][cond][cf_metric_i,:,:].data
def dfc_pairs_to_mat(dfc_data, allpairs, cond, phase):

    extraction_time = get_phase_extraction()[cond][phase]

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

    mat_dfc = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[allpairs == pair_to_find, :, :]
            x_rev = dfc_data[allpairs == pair_to_find_rev, :, :]

            x_mean_pair = np.median(np.concatenate((x, x_rev), axis=0), axis=0)

            x_mean_pair_time = np.median(x_mean_pair, axis=0)

            #### chunk
            x_mean_pair_time_chunk = x_mean_pair_time[extraction_time[0]:extraction_time[-1]]

            #### extract value dfc
            mat_dfc[x_i, y_i] = np.median(x_mean_pair_time_chunk, axis=0)

            if debug:
                plt.pcolormesh(x_mean_pair)
                plt.show()

                plt.plot(x_mean_pair_time)
                plt.show()

                plt.plot(x_mean_pair_time_chunk)
                plt.show()

    if debug:

        plt.matshow(mat_dfc)
        plt.show()

    return mat_dfc




def plot_all_verif(allband_data, allpairs, cond_to_compute, band_prep, BL_normalization):

    os.chdir(os.path.join(path_results, sujet, 'DFC', 'verif'))

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
        #pair_i = 200
        for pair_i in range(allpairs.shape[0]):

            if pair_i % 200 == 0:

                fig, axs = plt.subplots(ncols=len(cond_to_compute), nrows=len(freq_band_dict_FC_function[band_prep]), figsize=(15,15))

                if electrode_recording_type == 'monopolaire':
                    plt.suptitle(f'{cf_metric}_pair{pair_i}, mean nfrex, norm : {BL_normalization}', color='k')
                else:
                    plt.suptitle(f'{cf_metric}_pair{pair_i}_bi, mean nfrex, norm : {BL_normalization}', color='k')

                #band = 'theta'
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                    #cond = 'AC'
                    for c, cond in enumerate(cond_to_compute):
                    
                        ax = axs[r,c]

                        fc_to_plot = allband_data[band][cond][cf_metric_i,pair_i,:,:]

                        ax.plot(fc_to_plot.mean(axis=0), label='mean')
                        ax.plot(fc_to_plot.mean(axis=0) + fc_to_plot.std(axis=0), color='r', label='1SD')
                        ax.plot(fc_to_plot.mean(axis=0) - fc_to_plot.std(axis=0), color='r', label='1SD')
                        ax.plot([np.percentile(fc_to_plot, 10)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='10p')
                        ax.plot([np.percentile(fc_to_plot, 25)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='25p')
                        ax.plot([np.percentile(fc_to_plot, 40)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='40p')
                        ax.plot([np.percentile(fc_to_plot, 60)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='60p')
                        ax.plot([np.percentile(fc_to_plot, 75)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='75p')
                        ax.plot([np.percentile(fc_to_plot, 90)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='90p')

                        if r == 0:
                            ax.set_title(f'{cond}')
                        if c == 0:
                            ax.set_ylabel(f'{band}')
                        # plt.show()

                plt.savefig(f'cf_pair{pair_i}_{cf_metric}_{band_prep}_norm_{BL_normalization}.png')
                plt.close('all')

    #### select pairs to plot
    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

    for pair_i, pair in enumerate(pair_unique):

        if pair_i % 10 == 0:

            pair_to_find = pair
            pair_to_find_rev = f"{pair.split('-')[-1]}-{pair.split('-')[0]}"

            pair_i_list = np.array([])

            pair_i_list = np.append(pair_i_list, np.where(allpairs == pair_to_find)[0])
            pair_i_list = np.append(pair_i_list, np.where(allpairs == pair_to_find_rev)[0])

            pair_i_list = pair_i_list.astype('int')

            #cf_metric_i, cf_metric = 0, 'ispc'
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                fig, axs = plt.subplots(ncols=len(cond_to_compute), nrows=len(freq_band_dict_FC_function[band_prep]), figsize=(15,15))

                #band = 'theta'
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                    #cond = 'RD_SV'
                    for c, cond in enumerate(cond_to_compute):

                        fc_to_plot = allband_data[band][cond][cf_metric_i,pair_i_list,:,:].mean(axis=1)
                        
                        ax = axs[r,c]
                        
                        ax.plot(fc_to_plot.mean(axis=0), label='mean')
                        ax.plot(fc_to_plot.mean(axis=0) + fc_to_plot.std(axis=0), color='r', label='1SD')
                        ax.plot(fc_to_plot.mean(axis=0) - fc_to_plot.std(axis=0), color='r', label='1SD')
                        ax.plot([np.percentile(fc_to_plot, 10)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='10p')
                        ax.plot([np.percentile(fc_to_plot, 25)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='25p')
                        ax.plot([np.percentile(fc_to_plot, 40)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='40p')
                        ax.plot([np.percentile(fc_to_plot, 60)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='60p')
                        ax.plot([np.percentile(fc_to_plot, 75)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='75p')
                        ax.plot([np.percentile(fc_to_plot, 90)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='90p')

                        if r == 0:
                            ax.set_title(f'{cond}')
                        if c == 0:
                            ax.set_ylabel(f'{band}')

                if electrode_recording_type == 'monopolaire':
                    plt.suptitle(f'{cf_metric} {pair_to_find} mean pair, count : {fc_to_plot.shape[0]}, norm : {BL_normalization}', color='k')
                else:
                    plt.suptitle(f'{cf_metric} {pair_to_find} mean pair, count : {fc_to_plot.shape[0]}_bi, norm : {BL_normalization}', color='k')

                ax.legend()

                # plt.show()

                plt.savefig(f'cf_mean_allpair{pair_i}_{cf_metric}_{band_prep}_norm_{BL_normalization}.png')
                plt.close('all')
                    
    #### export mat count pairs
    mat_count_pairs = generate_count_pairs_mat(allpairs)

    fig, ax = plt.subplots(figsize=(15,15))

    ax.matshow(mat_count_pairs)

    for (i, j), z in np.ndenumerate(mat_count_pairs):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax.set_yticks(np.arange(roi_in_data.shape[0]))
    ax.set_yticklabels(roi_in_data)

    # plt.show()

    fig.savefig(f'{sujet}_MAT_COUNT.png')
    plt.close('all')

     







#dfc_data, pairs = allband_data[band][cond][phase][cf_metric_i,:,:], allpairs
def generate_count_pairs_mat(pairs):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)

    mat_count_pairs = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = pairs[pairs == pair_to_find]
            x_rev = pairs[pairs == pair_to_find_rev]

            x_tot_pair = x.shape[0] + x_rev.shape[0]

            mat_count_pairs[x_i, y_i] = x_tot_pair

    return mat_count_pairs
    
















################################
######## SAVE FIG ########
################################



def process_fc_res(sujet, electrode_recording_type, BL_normalization, plot_verif, plot_circle_dfc=False, plot_thresh=False):

    print(f'######## DFC BL_norm : {BL_normalization} ########')

    cond_to_plot = ['AC', 'SNIFF', 'AL']

    #### LOAD DATA ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') == -1)]

    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].values

    band_prep = 'wb'

    #### phase list
    phase_list_allcond = {}
    for cond in conditions:

        if cond == 'FR_CV':
            phase_list_allcond[cond] = ['whole']

        if cond == 'AC':
            phase_list_allcond[cond] = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

        if cond == 'SNIFF':
            phase_list_allcond[cond] = ['pre', 'resp_evnmt', 'post']

        if cond == 'AL':
            phase_list_allcond[cond] = ['resp_evnmt_1', 'resp_evnmt_2']

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    allband_data = {}

    #band = 'theta'
    for band in freq_band_dict_FC_function[band_prep]:

        allband_data[band] = {}
        #cond = 'AL'
        for cond in conditions:

            if electrode_recording_type == 'monopolaire':
                file_to_load = f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc'
            else:
                file_to_load = f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc'
            
            allband_data[band][cond] = xr.open_dataarray(file_to_load)
            allpairs = xr.open_dataarray(file_to_load)['pairs'].data

    #### clean data
    allband_data, allpairs = clean_data(allband_data, allpairs)
    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

    if debug:

        for band in freq_band_dict_FC_function[band_prep]:

            for cond in conditions:
                    
                plt.pcolormesh(allband_data[band][cond][0, 0, :, :])

                plt.title(f"{band},{cond}")
                plt.show()

    #### plot verif
    if plot_verif:

        plot_all_verif(allband_data, allpairs, conditions, band_prep, BL_normalization)
            
    #### mean and chunk phase
    allband_dfc_phase = {}
    #band = 'theta'
    for band in freq_band_dict_FC_function[band_prep]:

        print(band)

        allband_dfc_phase[band] = {}

        #cond = 'AL'
        for cond in conditions:

            allband_dfc_phase[band][cond] = {}

            #phase = 'whole'
            for phase_i, phase in enumerate(phase_list_allcond[cond]):

                mat_fc_i = np.zeros((2, len(roi_in_data), len(roi_in_data)))

                #cf_metric_i, cf_metric = 0, 'ispc'
                for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
                        
                    mat_fc_i[cf_metric_i, :, :] = dfc_pairs_to_mat(allband_data[band][cond][cf_metric_i,:,:].values, allpairs, cond, phase)

                allband_dfc_phase[band][cond][phase] = mat_fc_i

    if debug:

        for band in freq_band_dict_FC_function[band_prep]:
            
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                for cond in conditions:

                    for phase_i, phase in enumerate(phase_list_allcond[cond]):

                        fig, ax = plt.subplots()
                        vmin = allband_dfc_phase[band][cond][phase][cf_metric_i,:,:][allband_dfc_phase[band][cond][phase][cf_metric_i,:,:] != 0].min()
                        vmax = allband_dfc_phase[band][cond][phase][cf_metric_i,:,:][allband_dfc_phase[band][cond][phase][cf_metric_i,:,:] != 0].max()
                        cax = ax.matshow(allband_dfc_phase[band][cond][phase][cf_metric_i,:,:], vmin=vmin, vmax=vmax)
                        fig.colorbar(cax)
                        ax.set_xticks(np.arange(roi_in_data.shape[0]))
                        ax.set_xticklabels(roi_in_data)
                        ax.set_yticks(np.arange(roi_in_data.shape[0]))
                        ax.set_yticklabels(roi_in_data)
                        plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                                ha="left", va="center",rotation_mode="anchor")
                        ax.set_title(f'{cf_metric}, {band}')
                        fig.tight_layout()
                        plt.show()

    del allband_data

    #### normalization
    if BL_normalization:
        
        for band in freq_band_dict_FC_function[band_prep]:
            #cond = 'RD_SV'
            for cond in conditions:

                if cond == 'FR_CV':
                    continue

                #phase = 'whole'
                for phase_i, phase in enumerate(phase_list_allcond[cond]):
                    #cf_metric_i, cf_metric = 0, 'ispc'

                    allband_dfc_phase[band][cond][phase] = allband_dfc_phase[band][cond][phase] - allband_dfc_phase[band]['FR_CV']['whole']

    if debug:

        for band in freq_band_dict_FC_function[band_prep]:
            
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                for cond in conditions:

                    for phase_i, phase in enumerate(phase_list_allcond[cond]):

                        fig, ax = plt.subplots()
                        vmin = allband_dfc_phase[band][cond][phase][cf_metric_i,:,:][allband_dfc_phase[band][cond][phase][cf_metric_i,:,:] != 0].min()
                        vmax = allband_dfc_phase[band][cond][phase][cf_metric_i,:,:][allband_dfc_phase[band][cond][phase][cf_metric_i,:,:] != 0].max()
                        cax = ax.matshow(allband_dfc_phase[band][cond][phase][cf_metric_i,:,:], vmin=vmin, vmax=vmax)
                        fig.colorbar(cax)
                        ax.set_xticks(np.arange(roi_in_data.shape[0]))
                        ax.set_xticklabels(roi_in_data)
                        ax.set_yticks(np.arange(roi_in_data.shape[0]))
                        ax.set_yticklabels(roi_in_data)
                        plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                                ha="left", va="center",rotation_mode="anchor")
                        ax.set_title(f'{cf_metric}, {band}')
                        fig.tight_layout()
                        plt.show()

    #### identify scales
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            scales_abs[mat_type][band] = {}

            max_list = np.array(())

            #cond = 'RD_SV'
            for cond in cond_to_plot:

                scales_abs[mat_type][band][cond] = {}

                for phase_i, phase in enumerate(phase_list_allcond[cond]):

                    max_list = np.append(max_list, np.abs(allband_dfc_phase[band][cond][phase][mat_type_i,:,:].min()))
                    max_list = np.append(max_list, allband_dfc_phase[band][cond][phase][mat_type_i,:,:].max())

                scales_abs[mat_type][band][cond]['max'] = max_list.max()

                if BL_normalization:
                    scales_abs[mat_type][band][cond]['min'] = -max_list.max()
                else:
                    scales_abs[mat_type][band][cond]['min'] = 0

    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(allband_dfc_phase)

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            for cond in cond_to_plot:

                for phase_i, phase in enumerate(phase_list_allcond[cond]):

                    thresh_up = np.percentile(allband_dfc_phase[band][cond][phase][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allband_dfc_phase[band][cond][phase][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean[band][cond][phase][mat_type_i,:,:].shape[1]):
                        for y in range(mat_dfc_clean[band][cond][phase][mat_type_i,:,:].shape[1]):
                            if mat_type_i == 0:
                                if mat_dfc_clean[band][cond][phase][mat_type_i,x,y] < thresh_up:
                                    mat_dfc_clean[band][cond][phase][mat_type_i,x,y] = 0
                            else:
                                if (mat_dfc_clean[band][cond][phase][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][cond][phase][mat_type_i,x,y] > thresh_down):
                                    mat_dfc_clean[band][cond][phase][mat_type_i,x,y] = 0




    ######## PLOT ########


    #### go to results
    os.chdir(os.path.join(path_results, sujet, 'DFC', 'summary'))

    if BL_normalization:
        plot_color = cm.seismic
    else:
        plot_color = cm.YlGn

    #### RAW

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        ######## NO THRESH ########

        #cond = 'AL'
        for cond in cond_to_plot:

            #### mat plot raw
            phase_list = phase_list_allcond[cond]

            fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=len(phase_list), figsize=(15,15))

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{cond} {mat_type}')
            else:
                plt.suptitle(f'{cond} {mat_type} bi')
            
            for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                for c, phase in enumerate(phase_list):

                    ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{phase}')
                    
                    cax = ax.matshow(allband_dfc_phase[band][cond][phase][mat_type_i,:,:], vmin=scales_abs[mat_type][band][cond]['min'], 
                                        vmax=scales_abs[mat_type][band][cond]['max'], cmap=plot_color)

                    if c == len(phase_list)-1:
                        fig.colorbar(cax, ax=ax)

                    ax.set_xticks(np.arange(roi_in_data.shape[0]))
                    ax.set_xticklabels(roi_in_data)
                    ax.set_yticks(np.arange(roi_in_data.shape[0]))
                    ax.set_yticklabels(roi_in_data)
                    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45, ha="left", va="center",rotation_mode="anchor")
                    
            # plt.show()

            if electrode_recording_type == 'monopolaire':
                if BL_normalization:
                    fig.savefig(f'MAT_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_{mat_type}_{cond}_{band_prep}.png')
            else:
                if BL_normalization:
                    fig.savefig(f'MAT_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_bi_{mat_type}_{cond}_{band_prep}.png')
            
            plt.close('all')

            #### circle plot RAW
                
            if plot_circle_dfc:
                
                nrows, ncols = len(freq_band_dict_FC_function[band_prep]), len(phase_list_allcond[cond]['phase_list'])
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                    for c, phase in enumerate(phase_list):

                        mne_connectivity.viz.plot_connectivity_circle(allband_dfc_phase[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                    title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                    vmin=0, vmax=scales_abs[mat_type][band], colormap=plot_color, facecolor='w', 
                                                    textcolor='k')

                if electrode_recording_type == 'monopolaire':
                    plt.suptitle(f'{cond}_{mat_type}', color='k')
                else:
                    plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
                
                fig.set_figheight(10)
                fig.set_figwidth(12)
                # fig.show()

                if electrode_recording_type == 'monopolaire':
                    if BL_normalization:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                else:
                    if BL_normalization:
                        fig.savefig(f'CIRCLE_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_bi_{mat_type}_{cond}_{band_prep}.png')

                plt.close('all')

        ######## THRESH ########

        if plot_thresh:

            #cond = 'RD_SV'
            for cond in cond_to_plot:

                #### mat plot raw

                phase_list = phase_list_allcond[cond]['phase_list']

                if BL_normalization:
                    phase_list = [phase_i for phase_i in phase_list if phase_i != 'pre']

                fig, axs = plt.subplots(nrows=len(freq_band_dict_FC_function[band_prep]), ncols=len(phase_list), figsize=(15,15))

                if electrode_recording_type == 'monopolaire':
                    plt.suptitle(f'{cond} {mat_type}')
                else:
                    plt.suptitle(f'{cond} {mat_type} bi')
                
                for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                    for c, phase in enumerate(phase_list):

                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(band)
                        if r == 0:
                            ax.set_title(f'{phase}')
                        
                        cax = ax.matshow(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], vmin=scales_abs[mat_type][band][cond]['min'], 
                                            vmax=scales_abs[mat_type][band][cond]['max'], cmap=plot_color)

                        fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(roi_in_data.shape[0]))
                        ax.set_yticklabels(roi_in_data)
                # plt.show()

                if electrode_recording_type == 'monopolaire':
                    if BL_normalization:
                        fig.savefig(f'THRESH_MAT_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'THRESH_MAT_{mat_type}_{cond}_{band_prep}.png')
                else:
                    if BL_normalization:
                        fig.savefig(f'THRESH_MAT_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'THRESH_MAT_bi_{mat_type}_{cond}_{band_prep}.png')
                
                plt.close('all')

                #### circle plot RAW
                    
                if plot_circle_dfc:
                    
                    nrows, ncols = len(freq_band_dict_FC_function[band_prep]), len(phase_list_allcond[cond]['phase_list'])
                    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                    for r, band in enumerate(freq_band_dict_FC_function[band_prep]):

                        for c, phase in enumerate(phase_list):

                            mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[band][cond][phase][mat_type_i,:,:], node_names=roi_in_data, n_lines=None, 
                                                        title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                        vmin=0, vmax=scales_abs[mat_type][band], colormap=plot_color, facecolor='w', 
                                                        textcolor='k')

                    if electrode_recording_type == 'monopolaire':
                        plt.suptitle(f'{cond}_{mat_type}', color='k')
                    else:
                        plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
                    
                    fig.set_figheight(10)
                    fig.set_figwidth(12)
                    # fig.show()

                    if electrode_recording_type == 'monopolaire':
                        if BL_normalization:
                            fig.savefig(f'THRESH_CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                        else:
                            fig.savefig(f'THRESH_CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                    else:
                        if BL_normalization:
                            fig.savefig(f'THRESH_CIRCLE_bi_{mat_type}_{cond}_norm_{band_prep}.png')
                        else:
                            fig.savefig(f'THRESH_CIRCLE_bi_{mat_type}_{cond}_{band_prep}.png')

                    plt.close('all')










################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    plot_verif = False
    BL_normalization = True

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #electrode_recording_type = 'monopolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            # process_fc_res(sujet, electrode_recording_type, BL_normalization, plot_verif)
            execute_function_in_slurm_bash('n13_res_DFC', 'process_fc_res', [sujet, electrode_recording_type, BL_normalization, plot_verif])

