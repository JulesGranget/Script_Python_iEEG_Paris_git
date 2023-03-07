
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr
import joblib
import mne_connectivity
import copy
import pingouin as pg

from n0quater_stats import * 
from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False




################################
######## CLEAN DATA ########
################################



def clean_data(allband_data, allpairs, cond_to_compute):

    #### identify pairs to clean
    mask_keep = []

    for pair_i, pair in enumerate(allpairs):

        if pair.find('WM') == -1 and pair.find('ventricule') == -1 and pair.find('choroide plexus') == -1:

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

        for cond in cond_to_compute:

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

    for cond in cond_FC_DFC:

        if cond == 'FR_CV':
            continue

        if cond == 'AL':
            phase_extraction_time[cond] = {}
            phase_extraction_time[cond]['pre'] = [0, stretch_point_TF]
            phase_extraction_time[cond]['resp_evnmt_1'] = [0, int(n_points_AL_interpolation/n_phase_extraction_AL)]
            phase_extraction_time[cond]['resp_evnmt_2'] = [int(n_points_AL_interpolation/n_phase_extraction_AL), int(n_points_AL_interpolation/n_phase_extraction_AL)*2]
            phase_extraction_time[cond]['resp_evnmt_3'] = [int(n_points_AL_interpolation/n_phase_extraction_AL)*2, n_points_AL_interpolation]
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








#dfc_data = allband_data[band][cond][cf_metric_i,:,:].values
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
    




########################
######## STATS ########
########################

def get_stats_for_phase(allband_data, cond_to_compute, phase_list_allcond, band_prep, allpairs):

    #band = 'theta'
    for band in freq_band_dict_FC_function[band_prep]:

        #cond = cond_to_compute[0]
        for cond in cond_to_compute:

            #cf_metric_i, cf_metric = 0, 'ispc'
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']): 

                #### extract pre
                if cond =='AL':

                    dfc_pre = allband_data[band]['FR_CV'].loc[cf_metric, :, :, :]

                else:

                    dfc_pre = allband_data[band][cond].loc[cf_metric, :, :, :]

                pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

                mat_dfc_stats = np.zeros((len(phase_list_allcond[cond]['phase_list_diff']), len(roi_in_data), len(roi_in_data)))

                #pair_i, pair_name = 0, pair_unique[0]
                for pair_i, pair_name in enumerate(pair_unique):

                    print_advancement(pair_i, pair_unique.shape[0], steps=[25, 50, 75])

                    pair_to_find = pair_name
                    pair_to_find_rev = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
                    
                    x = dfc_pre[allpairs == pair_to_find, :, :]

                    if (allpairs == pair_to_find_rev).sum() != 0:
                        x_rev = dfc_pre[allpairs == pair_to_find_rev, :, :]

                        x = xr.concat((x, x_rev), dim='pairs')

                    x_mean_pair = x.median(axis=1)

                    #### chunk pre
                    extraction_time = get_phase_extraction()[cond]['pre']
                    x_stats_pre = x_mean_pair[:,extraction_time[0]:extraction_time[-1]].values.reshape(-1)

                    dfc_post = allband_data[band][cond].loc[cf_metric, :, :, :]

                    #phase_i, phase_name = 1, 'resp_evnmt_1'
                    for phase_i, phase_name in enumerate(phase_list_allcond[cond]['phase_list']):

                        if phase_name == 'pre':
                            continue

                        pair_to_find = pair_name
                        pair_to_find_rev = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
                        
                        x = dfc_post[allpairs == pair_to_find, :, :]

                        if (allpairs == pair_to_find_rev).sum() != 0:
                            x_rev = dfc_post[allpairs == pair_to_find_rev, :, :]

                            x = xr.concat((x, x_rev), dim='pairs')

                        x_mean_pair = x.median(axis=1)

                        #### chunk post
                        extraction_time = get_phase_extraction()[cond][phase_name]
                        x_stats_post = x_mean_pair[:,extraction_time[0]:extraction_time[-1]].values.reshape(-1)

                        if debug:

                            min = np.array([x_stats_pre.shape[0], x_stats_post.shape[0]]).min()
                            x_stats_pre_rdm = x_stats_pre[np.random.randint(low=0, high=x_stats_pre.shape[0], size=min)]
                            x_stats_post_rdm = x_stats_post[np.random.randint(low=0, high=x_stats_post.shape[0], size=min)]

                            bins = np.linspace(0, 1, 500)

                            plt.hist(x_stats_pre_rdm, bins=bins)
                            plt.hist(x_stats_post_rdm, bins=bins)
                            plt.show()

                        #### stats
                        if x_stats_pre.shape[0] != x_stats_post.shape[0]:
                            min = np.array([x_stats_pre.shape[0], x_stats_post.shape[0]]).min()
                            x_stats_pre_rdm = x_stats_pre[np.random.randint(low=0, high=x_stats_pre.shape[0], size=min)]
                            x_stats_post_rdm = x_stats_post[np.random.randint(low=0, high=x_stats_post.shape[0], size=min)]
                            pval_i = pg.ttest(x_stats_pre_rdm, x_stats_post_rdm, paired=True, alternative='two-sided')['cohen-d'].values[0]

                        else:
                            
                            pval_i = pg.ttest(x_stats_pre, x_stats_post, paired=True, alternative='two-sided')['cohen-d'].values[0]

                        
                        x_i = np.where(roi_in_data == pair_name.split('-')[0])[0][0]
                        y_i = np.where(roi_in_data == pair_name.split('-')[-1])[0][0]

                        mat_dfc_stats[phase_i-1, x_i, y_i] = pval_i
                        mat_dfc_stats[phase_i-1, y_i, x_i] = pval_i

                    
                if debug:

                    plt.matshow(mat_dfc_stats[0,:,:])
                    plt.show()













################################
######## SAVE FIG ########
################################



def process_fc_res(sujet, electrode_recording_type, FR_CV_normalized=False, plot_circle_dfc=False, plot_verif=False):

    print(f'######## DFC ########')

    cond_to_compute = ['AC', 'SNIFF', 'AL']
    cond_to_load = ['FR_CV', 'AC', 'SNIFF', 'AL']

    #### LOAD DATA ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    if electrode_recording_type == 'monopolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') == -1)]
    else:
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') != -1)]

    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].values

    #### phase list
    phase_list_allcond = {}
    for cond in cond_to_compute:

        phase_list_allcond[cond] = {}

        if cond == 'AC':
            phase_list_allcond[cond] = {
            'phase_list' : ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post'],
            'phase_list_diff' : ['pre-resp_evnmt_1', 'pre-resp_evnmt_2', 'resp_evnmt_1-resp_evnmt_2', 'pre-post']
            }

        if cond == 'SNIFF':
            phase_list_allcond[cond] = {
            'phase_list' : ['pre', 'resp_evnmt', 'post'],
            'phase_list_diff' : ['pre-resp_evnmt', 'pre-post']
            }

        if cond == 'AL':
            phase_list_allcond[cond] = {
            'phase_list' : ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'resp_evnmt_3'],
            'phase_list_diff' : ['pre-resp_evnmt_1', 'pre-resp_evnmt_2', 'pre-resp_evnmt_3']
            }

    #band_prep = 'lf'
    for band_prep in band_prep_list:

        #### load data 
        allband_data = {}
        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            allband_data[band] = {}
            #cond = 'AL'
            for cond in cond_to_load:

                if electrode_recording_type == 'monopolaire':
                    file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1 and i.find('bi') == -1)]
                else:
                    file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1 and i.find('bi') != -1)]
                
                allband_data[band][cond] = xr.open_dataarray(file_to_load[0])
                allpairs = xr.open_dataarray(file_to_load[0])['pairs'].values

        #### clean data
        allband_data, allpairs = clean_data(allband_data, allpairs, cond_to_load)
        pair_unique, roi_in_data = get_pair_unique_and_roi_unique(allpairs)

        if debug:

            for band in freq_band_dict_FC_function[band_prep]:

                for cond in cond_to_compute:
                        
                    plt.pcolormesh(allband_data[band][cond][0, 0, :, :])
    
                    plt.title(f"{band},{cond}")
                    plt.show()

                
        #### mean and chunk phase
        allband_dfc_phase = {}
        #band = 'theta'
        for band in freq_band_dict_FC_function[band_prep]:

            print(band)

            allband_dfc_phase[band] = {}

            #cond = 'AL'
            for cond in cond_to_compute:

                allband_dfc_phase[band][cond] = {}

                #phase = 'whole'
                for phase_i, phase in enumerate(phase_list_allcond[cond]['phase_list']):

                    mat_fc_i = np.zeros((2, len(roi_in_data), len(roi_in_data)))

                    #cf_metric_i, cf_metric = 0, 'ispc'
                    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                        if cond == 'AL' and phase == 'pre':

                            mat_fc_i[cf_metric_i, :, :] = dfc_pairs_to_mat(allband_data[band]['FR_CV'][cf_metric_i,:,:].values, allpairs, cond, phase)

                        else:
                            
                            mat_fc_i[cf_metric_i, :, :] = dfc_pairs_to_mat(allband_data[band][cond][cf_metric_i,:,:].values, allpairs, cond, phase)

                    allband_dfc_phase[band][cond][phase] = mat_fc_i

        
        get_stats_for_phase(allband_dfc_phase)







################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #electrode_recording_type = 'monopolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            execute_function_in_slurm_bash('n13_res_FC', 'process_fc_res', [sujet, electrode_recording_type])

        