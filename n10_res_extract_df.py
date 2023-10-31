

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import networkx as nx
import xarray as xr

import pickle
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False









################################
######## TF & ITPC ########
################################



def chunk_AL(sujet, cond, freq, electrode_recording_type):

    source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))
    time_AL = pd.read_excel(f'{sujet}_count_session.xlsx')
    time_AL = time_AL.iloc[:,-3:]

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    AL_chunks = {}

    for phase in ['pre', 'resp_evnmt', 'post']:

        AL_chunks_i = []

        for AL_i in range(time_AL.columns.shape[0]):

            if electrode_recording_type == 'monopolaire':
                data = np.load(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(AL_i+1)}.npy')
            if electrode_recording_type == 'bipolaire':
                data = np.load(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(AL_i+1)}_bi.npy')   

            time_vec = np.linspace(0, time_AL.iloc[0, AL_i], resampled_points_AL)

            if phase == 'pre':
                select_time_vec = time_vec < AL_extract_time
            elif phase == 'resp_evnmt':
                time_half_AL = time_AL.iloc[0, AL_i] / 2
                select_time_vec = (time_vec > time_half_AL) & (time_vec < (time_half_AL + AL_extract_time))
            elif phase == 'post':
                select_time_vec = (time_vec < time_AL.iloc[0, AL_i]) & (time_vec > (time_AL.iloc[0, AL_i] - AL_extract_time))

            #### resample
            f = scipy.interpolate.interp1d(np.linspace(0, 1, select_time_vec.sum()), data[:, select_time_vec], kind='linear') # exist different type of kind
            data_chunk = f(np.linspace(0, 1, n_points_AL_chunk))

            AL_chunks_i.append(data_chunk)

        AL_chunks[phase] = np.stack((AL_chunks_i[0], AL_chunks_i[1], AL_chunks_i[2])).mean(axis=0)

    os.chdir(source)

    return AL_chunks
           

    





def export_TF_in_df(sujet, electrode_recording_type):

    #### verif computation
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF.xlsx')):
            print('TF : ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF_bi.xlsx')):
            print('TF : ALREADY COMPUTED', flush=True)
            return

    #### load prms
    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)
    band_prep = 'wb'
    
    #### identify chan params
    chan_list = prms['chan_list_ieeg']

    #### prepare df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'band', 'phase', 'Pxx'])

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    #### fill df
    #chan_i, chan_name = 0, chan_list[0]
    for chan_i, chan_name in enumerate(chan_list):

        print(chan_name, flush=True)

        print_advancement(chan_i, len(chan_list), steps=[25, 50, 75])

        ROI_i = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
        Lobe_i = df_loca['lobes'][df_loca['name'] == chan_name].values[0]

        if chan_name.find('p') or chan_name.find("'"):
            side_i = 'l'
        else:
            side_i = 'r' 

        #cond = 'SNIFF'
        for cond in conditions:
            #band, freq = 'theta', [4, 8]
            for band, freq in freq_band_dict_df_extraction[band_prep].items():  

                #### load
                if electrode_recording_type == 'monopolaire':
                    data = np.median(np.load(f'{sujet}_tf_{cond}.npy')[chan_i,:,:,:], axis=0)
                else:
                    data = np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[chan_i,:,:,:], axis=0)

                #### sel freq
                mask_frex_band = (frex >= freq[0]) & (frex <= freq[-1])
                Pxx = data[mask_frex_band,:]

                if cond == 'AL':

                    phase_list = ['re_1', 're_2']

                    Pxx_re_1 = np.median(Pxx[:,:int(Pxx.shape[-1]/2)])
                    Pxx_re_2 = np.median(Pxx[:,int(Pxx.shape[-1]/2):])

                    Pxx_list = [Pxx_re_1, Pxx_re_2]

                if cond == 'FR_CV':

                    phase_list = ['FR_CV_whole', 'FR_CV_inspi', 'FR_CV_expi']

                    Pxx_whole = np.median(Pxx)
                    Pxx_inspi = np.median(Pxx[:,stretch_point_I[0]:stretch_point_I[1]])
                    Pxx_expi = np.median(Pxx[:,stretch_point_E[0]:stretch_point_E[1]])

                    Pxx_list = [Pxx_whole, Pxx_inspi, Pxx_expi]

                if cond == 'SNIFF':

                    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                    time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff_resampled)

                    Pxx_prepre = np.median(Pxx[:,(time_vec >= sniff_extract_prepre[0]) & (time_vec <= sniff_extract_prepre[1])])
                    Pxx_pre = np.median(Pxx[:,(time_vec >= sniff_extract_pre[0]) & (time_vec <= sniff_extract_pre[1])])
                    Pxx_resp_evnmt = np.median(Pxx[:,(time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[1])])
                    Pxx_post = np.median(Pxx[:,(time_vec >= sniff_extract_post[0]) & (time_vec <= sniff_extract_post[1])])

                    phase_list = ['SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post']
                    Pxx_list = [Pxx_prepre, Pxx_pre, Pxx_resp_evnmt, Pxx_post]
                
                if cond == 'AC':

                    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                    time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac_resample)

                    Pxx_pre_1 = np.median(Pxx[:,(time_vec >= AC_extract_pre_1[0]) & (time_vec <= AC_extract_pre_1[1])])
                    Pxx_pre_2 = np.median(Pxx[:,(time_vec >= AC_extract_pre_2[0]) & (time_vec <= AC_extract_pre_2[1])])
                    Pxx_pre_3 = np.median(Pxx[:,(time_vec >= AC_extract_pre_3[0]) & (time_vec <= AC_extract_pre_3[1])])
                    Pxx_pre_4 = np.median(Pxx[:,(time_vec >= AC_extract_pre_4[0]) & (time_vec <= AC_extract_pre_4[1])])
                    Pxx_resp_evnmt_1 = np.median(Pxx[:,(time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[1])])
                    Pxx_resp_evnmt_2 = np.median(Pxx[:,(time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[1])])
                    Pxx_resp_evnmt_3 = np.median(Pxx[:,(time_vec >= AC_extract_resp_evnmt_3[0]) & (time_vec <= AC_extract_resp_evnmt_3[1])])
                    Pxx_resp_evnmt_4 = np.median(Pxx[:,(time_vec >= AC_extract_resp_evnmt_4[0]) & (time_vec <= AC_extract_resp_evnmt_4[1])])
                    Pxx_post_1 = np.median(Pxx[:,(time_vec >= AC_extract_post_1[0]) & (time_vec <= AC_extract_post_1[1])])
                    Pxx_post_2 = np.median(Pxx[:,(time_vec >= AC_extract_post_2[0]) & (time_vec <= AC_extract_post_2[1])])
                    Pxx_post_3 = np.median(Pxx[:,(time_vec >= AC_extract_post_3[0]) & (time_vec <= AC_extract_post_3[1])])
                    Pxx_post_4 = np.median(Pxx[:,(time_vec >= AC_extract_post_4[0]) & (time_vec <= AC_extract_post_4[1])])

                    phase_list = ['AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 'AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04', 'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
                    Pxx_list = [Pxx_pre_1, Pxx_pre_2, Pxx_pre_3, Pxx_pre_4, Pxx_resp_evnmt_1, Pxx_resp_evnmt_2, Pxx_resp_evnmt_3, Pxx_resp_evnmt_4, Pxx_post_1, Pxx_post_2, Pxx_post_3, Pxx_post_4]

                data_export_i =   {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'chan' : [chan_name]*len(phase_list), 
                                    'ROI' : [ROI_i]*len(phase_list), 'Lobe' : [Lobe_i]*len(phase_list), 'side' : [side_i]*len(phase_list), 
                                    'band' : [band]*len(phase_list), 'phase' : phase_list, 'Pxx' : Pxx_list}
                
                df_export_i = pd.DataFrame.from_dict(data_export_i)
                
                df_export = pd.concat([df_export, df_export_i])

    #### save
    df_export.index = np.arange(df_export.shape[0])
    
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if electrode_recording_type == 'monopolaire':
        df_export.to_excel(f'{sujet}_df_TF.xlsx')
    if electrode_recording_type == 'bipolaire':
        df_export.to_excel(f'{sujet}_df_TF_bi.xlsx')






def export_ITPC_in_df(sujet, prms, electrode_recording_type):

    #### verif computation
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC.xlsx')):
            print('ITPC : ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC_bi.xlsx')):
            print('ITPC : ALREADY COMPUTED', flush=True)
            return

    #### load prms
    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)
    
    #### identify chan params
    chan_list = prms['chan_list_ieeg']

    #### prepare df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'chan', 'ROI', 'Lobe', 'side', 'band', 'phase', 'Pxx'])

    #### fill df
    #chan_i, chan_name = 0, chan_list[0]
    for chan_i, chan_name in enumerate(chan_list):

        print_advancement(chan_i, len(chan_list), steps=[25, 50, 75])

        ROI_i = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
        Lobe_i = df_loca['lobes'][df_loca['name'] == chan_name].values[0]

        if chan_name.find('p') or chan_name.find("'"):
            side_i = 'l'
        else:
            side_i = 'r' 

        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #cond = 'AL'
            for cond in prms['conditions']:
                #band, freq = 'theta', [2, 10]
                for band, freq in freq_band_dict[band_prep].items():

                    # if cond != 'AL':
                    #     data = get_tf_itpc_stretch_allcond(sujet, 'ITPC', electrode_recording_type)[band_prep][cond][band][chan_i, :, :]

                    Pxx = np.mean(band, axis=0)

                    if cond == 'FR_CV':

                        Pxx_whole = Pxx.mean()
                        Pxx_inspi = np.mean(Pxx[stretch_point_I[0]:stretch_point_I[1]])
                        Pxx_expi = np.mean(Pxx[stretch_point_E[0]:stretch_point_E[1]])
                        Pxx_IE = np.mean(Pxx[stretch_point_IE[0]:stretch_point_IE[1]])
                        Pxx_EI = np.mean(Pxx[stretch_point_EI[0]:]) + np.mean(Pxx[:stretch_point_EI[1]])

                        phase_list = ['FR_CV_whole', 'FR_CV_inspi', 'FR_CV_expi', 'FR_CV_IE', 'FR_CV_EI']
                        Pxx_list = [Pxx_whole, Pxx_inspi, Pxx_expi, Pxx_IE, Pxx_EI]

                    if cond == 'SNIFF':

                        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                        time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

                        Pxx_pre = np.mean(Pxx[(time_vec >= sniff_extract_pre[0]) & (time_vec <= sniff_extract_pre[1])])
                        Pxx_resp_evnmt = np.mean(Pxx[(time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[1])])
                        Pxx_post = np.mean(Pxx[(time_vec >= sniff_extract_post[0]) & (time_vec <= sniff_extract_post[1])])

                        phase_list = ['SNIFF_pre', 'SNIFF_resp_evnmt', 'SNIFF_post']
                        Pxx_list = [Pxx_pre, Pxx_resp_evnmt, Pxx_post]
                    
                    if cond == 'AC':

                        stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                        time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

                        Pxx_pre = np.mean(Pxx[(time_vec >= AC_extract_pre[0]) & (time_vec <= AC_extract_pre[1])])
                        Pxx_resp_evnmt_1 = np.mean(Pxx[(time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[1])])
                        Pxx_resp_evnmt_2 = np.mean(Pxx[(time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[1])])
                        Pxx_post = np.mean(Pxx[(time_vec >= AC_extract_post[0]) & (time_vec <= AC_extract_post[1])])

                        phase_list = ['AC_pre', 'AC_resp_evnmt_1', 'AC_resp_evnmt_2', 'AC_post']
                        Pxx_list = [Pxx_pre, Pxx_resp_evnmt_1, Pxx_resp_evnmt_2, Pxx_post]
                    
                    if cond == 'AL':

                        AL_chunks = chunk_AL(sujet, cond, freq, electrode_recording_type)

                        Pxx_pre = np.mean(AL_chunks['pre'])
                        Pxx_resp_evnmt = np.mean(AL_chunks['resp_evnmt'])
                        Pxx_post = np.mean(AL_chunks['post'])

                        phase_list = ['AL_pre', 'AL_resp_evnmt', 'AL_post']
                        Pxx_list = [Pxx_pre, Pxx_resp_evnmt, Pxx_post]

                    data_export_i =   {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'chan' : [chan_name]*len(phase_list), 
                                        'ROI' : [ROI_i]*len(phase_list), 'Lobe' : [Lobe_i]*len(phase_list), 'side' : [side_i]*len(phase_list), 
                                        'band' : [band]*len(phase_list), 'phase' : phase_list, 'Pxx' : Pxx_list}
                    
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if electrode_recording_type == 'monopolaire':
        df_export.to_excel(f'{sujet}_df_ITPC.xlsx')
    if electrode_recording_type == 'bipolaire':
        df_export.to_excel(f'{sujet}_df_ITPC_bi.xlsx')












########################################
######## GET FR_CV BASELINES ########
########################################



def clean_data_WM_ventricule_FR_CV(baselines, roi_in_data):

    if 'WM' not in roi_in_data and 'ventricule' not in roi_in_data:

        return baselines, roi_in_data

    else:
        #ROI_to_clean = 'WM'
        for ROI_to_clean in ['WM', 'ventricule']:

            if ROI_to_clean not in roi_in_data:
                continue

            ROI_to_clean_i = np.where(roi_in_data == ROI_to_clean)[0][0]

            #band_i = 'beta'
            for band_i in baselines:

                mat_pre = baselines[band_i]

                mat_post_1st_deletion = np.delete(mat_pre, ROI_to_clean_i, 1)
                mat_post = np.delete(mat_post_1st_deletion, ROI_to_clean_i, 2)

                if debug:
                    plt.matshow(mat_pre[0,:,:])
                    plt.matshow(mat_post[0,:,:])
                    plt.show()

                baselines[band_i] = mat_post

                roi_in_data = roi_in_data[roi_in_data != [ROI_to_clean]]
        
        return baselines, roi_in_data









########################################
######## COMPUTE GRAPH METRICS ########
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





#dfc_data = xr_graph.loc[cf_metric, :, :, sniff_extract_pre[0]:sniff_extract_pre[1]].data
def dfc_pairs_to_mat(dfc_data, pairs, compute_mode='median', rscore_computation=False):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
    
    #### mean across pairs
    dfc_mean_pair = np.zeros(( pair_unique.shape[0], nfrex_dfc, dfc_data.shape[-1] ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[pairs == pair_to_find, :, :]
            x_rev = dfc_data[pairs == pair_to_find_rev, :, :]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            #### identify pair name mean
            try:
                pair_position = np.where(pair_unique == pair_to_find)[0][0]
            except:
                pair_position = np.where(pair_unique == pair_to_find_rev)[0][0]

            dfc_mean_pair[pair_position, :, :] = x_mean

    #### mean pairs to mat
    #### fill mat
    mat_dfc = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_mean_pair[pair_unique == pair_to_find, :, :]
            x_rev = dfc_mean_pair[pair_unique == pair_to_find_rev, :, :]

            x_mean = np.median(np.vstack([x, x_rev]), axis=0)

            if rscore_computation:
                x_mean_rscore = rscore_mat(x_mean)
            else:
                x_mean_rscore = x_mean

            if compute_mode == 'mean':
                val_to_place = x_mean_rscore.mean(axis=1).mean(0)
            if compute_mode == 'median':
                val_to_place = np.median(x_mean_rscore,)
            if compute_mode == 'trapz':
                val_to_place = np.trapz(x_mean_rscore, axis=1).mean(0)

            mat_dfc[x_i, y_i] = val_to_place

    return mat_dfc





#mat = dfc_pre
#roi_names = roi_in_data.copy()
def cln_ROI(mat, roi_names):

    if 'WM' not in roi_names and 'ventricule' not in roi_names:

        return mat

    else:

        #ROI_to_clean = 'ventricule'
        for ROI_to_clean in ['WM', 'ventricule']:

            try:
                ROI_to_clean_i = np.where(roi_names == ROI_to_clean)[0][0]
            except:
                continue

            mat_post_1st_deletion = np.delete(mat, ROI_to_clean_i, axis=0)
            mat_post = np.delete(mat_post_1st_deletion, ROI_to_clean_i, axis=1)

            mat = mat_post.copy()

            roi_names = roi_names[roi_names != [ROI_to_clean]]
        
    return mat_post











def compute_graph_metric_dfc(sujet, electrode_recording_type):

    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    band_prep = 'wb'

    #### verif computation
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC.xlsx')):
            print('DFC : ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC_bi.xlsx')):
            print('DFC : ALREADY COMPUTED', flush=True)
            return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])

    #### compute
    #cond = 'FR_CV'
    for cond in conditions:

        #band, freq = 'beta', [12,40]
        for band, freq in freq_band_dict_FC_function[band_prep].items():

            #cf_metric = 'ispc'
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                if electrode_recording_type == 'monopolaire':
                    xr_graph = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                if electrode_recording_type == 'bipolaire':
                    xr_graph = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')

                pairs = xr_graph['pairs'].data
                pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
                cf_metric_i = np.where(xr_graph['mat_type'].data == cf_metric)[0]

                if cond == 'FR_CV':

                    dfc_whole = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, :].data, pairs)

                    phase_list = ['whole']
                    mat_cf = [cln_ROI(dfc_whole, roi_in_data)]

                if cond == 'SNIFF':

                    dfc_pre = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, sniff_extract_pre[0]:sniff_extract_pre[1]].data, pairs)
                    dfc_resp_evnmt = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, sniff_extract_resp_evnmt[0]:sniff_extract_resp_evnmt[1]].data, pairs)
                    dfc_post = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, sniff_extract_post[0]:sniff_extract_post[1]].data, pairs)

                    phase_list = ['pre', 'resp_evnt', 'post']
                    mat_cf = [cln_ROI(dfc_pre, roi_in_data), cln_ROI(dfc_resp_evnmt, roi_in_data), cln_ROI(dfc_post, roi_in_data)]
                
                if cond == 'AC':

                    dfc_pre = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, AC_extract_pre[0]:AC_extract_pre[1]].data, pairs)
                    dfc_resp_evnmt_1 = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, AC_extract_resp_evnmt_1[0]:AC_extract_resp_evnmt_1[1]].data, pairs)
                    dfc_resp_evnmt_2 = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, AC_extract_resp_evnmt_2[0]:AC_extract_resp_evnmt_2[1]].data, pairs)
                    dfc_post = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, AC_extract_post[0]:AC_extract_post[1]].data, pairs)

                    phase_list = ['pre', 'resp_evnt_1', 'resp_evnt_2', 'post']
                    mat_cf = [cln_ROI(dfc_pre, roi_in_data), cln_ROI(dfc_resp_evnmt_1, roi_in_data), cln_ROI(dfc_resp_evnmt_2, roi_in_data), cln_ROI(dfc_post, roi_in_data)]

                if cond == 'AL':

                    dfc_resp_evnmt_1 = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, :int(xr_graph.shape[-1]/2)].data, pairs)
                    dfc_resp_evnmt_2 = dfc_pairs_to_mat(xr_graph.loc[cf_metric, :, :, int(xr_graph.shape[-1]/2):].data, pairs)

                    phase_list = ['resp_evnt_1', 'resp_evnt_2']
                    mat_cf = [cln_ROI(dfc_resp_evnmt_1, roi_in_data), cln_ROI(dfc_resp_evnmt_2, roi_in_data)]

                if debug:
                    plt.plot(xr_graph[cf_metric_i, 0, :].data.reshape(-1))
                    plt.show()

                    plt.matshow(mat_cf[0])
                    plt.matshow(mat_cf[1])
                    plt.show()

                #### Build graph
                #dfc_phase_i, dfc_phase = 0, 'pre'
                for dfc_phase_i, dfc_phase in enumerate(phase_list):

                    mat = mat_cf[dfc_phase_i]
                    mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                    
                    if debug:
                        np.sum(mat_values > np.percentile(mat_values, 90))

                        count, bin, fig = plt.hist(mat_values)
                        plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, percentile_graph_metric), ymin=count.min(), ymax=count.max(), color='r')
                        plt.show()

                    #### apply thresh
                    for chan_i in range(mat.shape[0]):
                        mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

                    #### verify that the graph is fully connected
                    chan_i_to_remove = []
                    for chan_i in range(mat.shape[0]):
                        if np.sum(mat[chan_i,:]) == 0:
                            chan_i_to_remove.append(chan_i)

                    mat_i_mask = [i for i in range(mat.shape[0]) if i not in chan_i_to_remove]

                    if len(chan_i_to_remove) != 0:
                        for row in range(2):
                            if row == 0:
                                mat = mat[mat_i_mask,:]
                            elif row == 1:
                                mat = mat[:,mat_i_mask]

                    if debug:
                        plt.matshow(mat)
                        plt.show()

                    #### generate graph
                    G = nx.from_numpy_array(mat)
                    if debug:
                        list(G.nodes)
                        list(G.edges)
                    
                    nodes_names = {}
                    for node_i, roi_in_data_i in enumerate(mat_i_mask):
                        nodes_names[node_i] = roi_in_data[roi_in_data_i]
                
                    nx.relabel_nodes(G, nodes_names, copy=False)
                    
                    if debug:
                        G.nodes.data()
                        nx.draw(G, with_labels=True)
                        plt.show()

                        pos = nx.circular_layout(G)
                        nx.draw(G, pos=pos, with_labels=True)
                        plt.show()

                    node_degree = {}
                    for node_i, node_name in zip(mat_i_mask, roi_in_data[mat_i_mask]):
                        node_degree[node_name] = G.degree[roi_in_data[node_i]]

                    CPL = nx.average_shortest_path_length(G)
                    GE = nx.global_efficiency(G)
                    SWN = nx.omega(G, niter=5, nrand=10, seed=None)

                    data_export_i =    {'sujet' : [sujet], 'cond' : [cond], 'band' : [band], 'metric' : [cf_metric], 'phase' : [dfc_phase], 
                                    'CPL' : [CPL], 'GE' : [GE], 'SWN' : [SWN]}
                    
                    df_export_i = pd.DataFrame.from_dict(data_export_i)

                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if electrode_recording_type == 'monopolaire':
        df_export.to_excel(f'{sujet}_df_graph_DFC.xlsx')
    if electrode_recording_type == 'bipolaire':
        df_export.to_excel(f'{sujet}_df_graph_DFC_bi.xlsx')






########################################
######## EXTRACT DFC VALUES ########
########################################


def generate_ROI_pairs():

    pairs_of_interest = np.array([])

    for A in ROI_for_DFC_df:

        for B in ROI_for_DFC_df:

            if A == B:
                continue

            pair_i = f'{A}-{B}'

            pairs_of_interest = np.append(pairs_of_interest, pair_i)

    return pairs_of_interest





#dfc_data, pairs = xr_dfc[cf_metric_i, :, :, :].values, xr_dfc['pairs'].data
def dfc_pairs_median(dfc_data, pairs):

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pairs)
    
    #### mean across pairs
    dfc_mean_pair = np.zeros(( pair_unique.shape[0], dfc_data.shape[1], dfc_data.shape[-1] ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[pairs == pair_to_find, :, :]
            x_rev = dfc_data[pairs == pair_to_find_rev, :, :]

            x_mean = np.median(np.vstack([x, x_rev]), axis=0)

            #### identify pair name mean
            try:
                pair_position = np.where(pair_unique == pair_to_find)[0][0]
            except:
                pair_position = np.where(pair_unique == pair_to_find_rev)[0][0]

            dfc_mean_pair[pair_position, :, :] = x_mean

    return dfc_mean_pair, pair_unique













def compute_dfc_values(sujet, electrode_recording_type):

    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    band_prep = 'wb'

    #### verif computation
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC.xlsx')):
            print('DFC values : ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC_bi.xlsx')):
            print('DFC values : ALREADY COMPUTED', flush=True)
            return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'pair', 'value'])

    #### get pairs of interest
    pairs_of_interest = generate_ROI_pairs()

    #### compute
    #cond = 'AC'
    for cond in conditions:
        #band, freq = 'theta', [4,8]
        for band, freq in freq_band_dict_FC_function[band_prep].items():
            #cf_metric_i, cf_metric = 0, 'ispc'
            for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                print(cond, band, cf_metric, flush=True)

                #### extract data
                if electrode_recording_type == 'monopolaire':
                    xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                if electrode_recording_type == 'bipolaire':
                    xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc') 

                #### mean across pairs
                dfc_mean_pair, pairs_unique = dfc_pairs_median(xr_dfc.loc[cf_metric, :, :, :].data, xr_dfc['pairs'].data)

                #### identify pairs of interest
                #pair_i, pair_name = 0, pairs_of_interest[0]
                for pair_i, pair_name in enumerate(pairs_of_interest):

                    pairs_of_interest_i_list = np.where((pairs_unique == pair_name) | (pairs_unique == f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"))[0]
                    
                    if pairs_of_interest_i_list.shape[0] == 0:
                        continue

                    #pair_selected_i = pairs_of_interest_i_list[0]
                    for pair_selected_i in pairs_of_interest_i_list:

                        if cond == 'FR_CV':

                            phase_list = ['whole']

                            value_list = [np.median(dfc_mean_pair[pair_selected_i, :, :])]

                        if cond == 'SNIFF':

                            time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, dfc_mean_pair.shape[-1])

                            select_time_vec_pre = (time_vec >= sniff_extract_pre[0]) & (time_vec <= sniff_extract_pre[1])
                            select_time_vec_resp_evnmt = (time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[1])
                            select_time_vec_post = (time_vec >= sniff_extract_post[0]) & (time_vec <= sniff_extract_post[1])

                            phase_list = ['whole', 'pre', 'resp_evnmt', 'post']

                            value_list =    [np.median(dfc_mean_pair[pair_selected_i, :, :]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_pre]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_resp_evnmt]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_post])
                                        ]
                        
                        if cond == 'AC':

                            time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac_resample)

                            select_time_vec_pre = (time_vec >= AC_extract_pre[0]) & (time_vec <= AC_extract_pre[1])
                            select_time_vec_resp_evnmt_1 = (time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[1])
                            select_time_vec_resp_evnmt_2 = (time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[1])
                            select_time_vec_post = (time_vec >= AC_extract_post[0]) & (time_vec <= AC_extract_post[1])

                            phase_list = ['whole', 'pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

                            value_list =    [np.median(dfc_mean_pair[pair_selected_i, :, :]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_pre]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_resp_evnmt_1]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_resp_evnmt_2]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, select_time_vec_post])
                                        ]

                        if cond == 'AL':

                            phase_list = ['resp_evnmt_1', 'resp_evnmt_2']

                            value_list =    [np.median(dfc_mean_pair[pair_selected_i, :, :int(dfc_mean_pair.shape[-1]/2)]),
                                        np.median(dfc_mean_pair[pair_selected_i, :, int(dfc_mean_pair.shape[-1]/2):]),
                                        ]

                        data_export_i =    {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'band' : [band]*len(phase_list), 
                                'metric' : [cf_metric]*len(phase_list), 'phase' : phase_list, 'pair' : [pairs_unique[pair_selected_i]]*len(phase_list), 
                                'value' : value_list}

                        df_export_i = pd.DataFrame.from_dict(data_export_i)

                        df_export = pd.concat([df_export, df_export_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'df'))
    if electrode_recording_type == 'monopolaire':
        df_export.to_excel(f'{sujet}_df_DFC.xlsx')
    if electrode_recording_type == 'bipolaire':
        df_export.to_excel(f'{sujet}_df_DFC_bi.xlsx')










########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df(sujet, electrode_recording_type):

    print(sujet, flush=True)

    #### export
    print('COMPUTE TF', flush=True)
    export_TF_in_df(sujet, electrode_recording_type)
    
    # print('COMPUTE ITPC')
    # export_ITPC_in_df(sujet, prms, electrode_recording_type)
    
    # print('COMPUTE GRAPH DFC', flush=True)
    # compute_graph_metric_dfc(sujet, electrode_recording_type)
    
    print('COMPUTE DFC VALUES', flush=True)
    compute_dfc_values(sujet, electrode_recording_type)





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #electrode_recording_type = 'bipolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:
                    
            #### export df
            # compilation_export_df(sujet, electrode_recording_type)
            # execute_function_in_slurm_bash_mem_choice('n10_res_extract_df', 'compilation_export_df', [sujet, electrode_recording_type], '20G')

            # export_TF_in_df(sujet, electrode_recording_type)
            execute_function_in_slurm_bash_mem_choice('n10_res_extract_df', 'export_TF_in_df', [sujet, electrode_recording_type], '20G')
        
        
