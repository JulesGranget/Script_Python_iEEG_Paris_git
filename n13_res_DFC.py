

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr
import copy
import mne_connectivity
import pickle


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





################################
######## CLEAN DATA ########
################################



def clean_data_WM_ventricule(allband_data, roi_in_data):

    if 'WM' not in roi_in_data and 'ventricule' not in roi_in_data:

        return allband_data, roi_in_data

    else:
        #ROI_to_clean = 'WM'
        for ROI_to_clean in ['WM', 'ventricule']:

            if ROI_to_clean not in roi_in_data:
                continue

            ROI_to_clean_i = np.where(roi_in_data == ROI_to_clean)[0][0]

            if list(allband_data.keys())[0] in list(freq_band_whole.keys()):

                #band_i = 'beta'
                for band_i in allband_data:

                    mat_pre = allband_data[band_i]

                    mat_post_1st_deletion = np.delete(mat_pre, ROI_to_clean_i, 1)
                    mat_post = np.delete(mat_post_1st_deletion, ROI_to_clean_i, 2)

                    if debug:
                        plt.matshow(mat_pre[0,:,:])
                        plt.matshow(mat_post[0,:,:])
                        plt.show()

                    allband_data[band_i] = mat_post

                    roi_in_data = roi_in_data[roi_in_data != [ROI_to_clean]]

            else:

                #resp_evnmt_i = 'pre'
                for resp_evnmt_i in allband_data:
                    #band_i = 'beta'
                    for band_i in allband_data[resp_evnmt_i]:

                        mat_pre = allband_data[resp_evnmt_i][band_i]

                        mat_post_1st_deletion = np.delete(mat_pre, ROI_to_clean_i, 1)
                        mat_post = np.delete(mat_post_1st_deletion, ROI_to_clean_i, 2)

                        if debug:
                            plt.matshow(mat_pre[0,:,:])
                            plt.matshow(mat_post[0,:,:])
                            plt.show()

                        allband_data[resp_evnmt_i][band_i] = mat_post

                        roi_in_data = roi_in_data[roi_in_data != [ROI_to_clean]]
        
        return allband_data, roi_in_data




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








################################################
######## COMPUTE DATA RESPI PHASE ########
################################################



def chunk_AL(data, sujet, export_type):

    source = os.getcwd()

    #### extract AL time
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    time_AL = pd.read_excel(f'{sujet}_count_session.xlsx')
    time_AL = time_AL.iloc[:,-3:]

    #### chunk
    Al_chunks = []

    #AL_i = 0
    for AL_i in range(time_AL.columns.shape[0]):

        print(AL_i)

        time_vec = np.linspace(0, time_AL.iloc[0, AL_i], n_points_AL_interpolation)

        if export_type == 'pre':
            select_time_vec = time_vec < AL_extract_time
        elif export_type == 'resp_evnmt':
            time_half_AL = time_AL.iloc[0, AL_i] / 2
            select_time_vec = (time_vec > time_half_AL) & (time_vec < (time_half_AL + AL_extract_time))
        elif export_type == 'post':
            select_time_vec = (time_vec < time_AL.iloc[0, AL_i]) & (time_vec > (time_AL.iloc[0, AL_i] - AL_extract_time))

        #### resample
        f = scipy.interpolate.interp1d(np.linspace(0, 1, select_time_vec.sum()), data[AL_i, :, :, :, select_time_vec].data, kind='linear') # exist different type of kind
        data_chunk = f(np.linspace(0, 1, n_points_AL_chunk))

        Al_chunks.append(data_chunk)

    data_chunk_mean = np.stack((Al_chunks[0], Al_chunks[1], Al_chunks[2])).mean(axis=0)

    data_chunk_mean = xr.DataArray(data_chunk_mean, dims=data.dims[1:], coords=[data.coords['mat_type'], data.coords['pairs'], np.arange(data.shape[-2]), np.arange(n_points_AL_chunk)] )

    os.chdir(source)

    return data_chunk_mean
           




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




#dfc_data, pairs = data_chunk.loc[cf_metric,:,:,:].data, data['pairs'].data
def dfc_pairs_to_mat(dfc_data, pairs, compute_mode, rscore_computation):

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

            x_mean = np.vstack([x, x_rev]).mean(axis=0)

            if rscore_computation:
                x_mean_rscore = rscore_mat(x_mean)
            else:
                x_mean_rscore = x_mean

            if compute_mode == 'mean':
                val_to_place = x_mean_rscore.mean(axis=1).mean(0)
            if compute_mode == 'trapz':
                val_to_place = np.trapz(x_mean_rscore, axis=1).mean(0)

            mat_dfc[x_i, y_i] = val_to_place

    return mat_dfc
    





def get_data_for_phase(sujet, cond, electrode_recording_type, roi_in_data, prms, rscore_computation=False):

    print(f'chunk {cond}')

    #### get ROI list
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    if electrode_recording_type == 'monopolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find(cond) != -1 and i.find('bi') == -1)]
    if electrode_recording_type == 'bipolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find(cond) != -1 and i.find('bi') != -1)]
    
    #### load data 
    allband_data = {}

    #### prepare export
    if cond == 'AC':
        export_type_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

    if cond == 'SNIFF': 
        export_type_list = ['pre', 'resp_evnmt', 'post']

    if cond == 'AL':
        export_type_list = ['pre', 'resp_evnmt', 'post']

    #export_type = 'pre'
    for export_type in export_type_list:

        print(f'export type : {export_type}')

        allband_data[export_type] = {}

        #band = 'beta'
        for band in band_name_fc_dfc:

            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

            if electrode_recording_type == 'monopolaire':
                file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1 and i.find('bi') == -1)]
            if electrode_recording_type == 'bipolaire':
                file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1 and i.find('bi') != -1)]
            
            data = xr.open_dataarray(file_to_load[0])

            if cond == 'SNIFF':

                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
                select_time_vec_pre = (time_vec >= sniff_extract_pre[0]) & (time_vec <= sniff_extract_pre[1])
                select_time_vec_resp_evnmt = (time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[1])
                select_time_vec_post = (time_vec >= sniff_extract_post[0]) & (time_vec <= sniff_extract_post[1])
            
            if cond == 'AC':

                stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
                select_time_vec_pre = (time_vec >= AC_extract_pre[0]) & (time_vec <= AC_extract_pre[1])
                select_time_vec_resp_evnmt_1 = (time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[1])
                select_time_vec_resp_evnmt_2 = (time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[1])
                select_time_vec_post = (time_vec >= AC_extract_post[0]) & (time_vec <= AC_extract_post[1])

            #### chunk

            if cond == 'AL':

                data_chunk = chunk_AL(data, sujet, export_type)

            else:
                if export_type == 'pre':
                    data_chunk = data[:, :, :, select_time_vec_pre]
                elif export_type == 'resp_evnmt':
                    data_chunk = data[:, :, :, select_time_vec_resp_evnmt]
                elif export_type == 'resp_evnmt_1':
                    data_chunk = data[:, :, :, select_time_vec_resp_evnmt_1]
                elif export_type == 'resp_evnmt_2':
                    data_chunk = data[:, :, :, select_time_vec_resp_evnmt_2]
                elif export_type == 'post':
                    data_chunk = data[:, :, :, select_time_vec_post]

            #### fill mat
            mat_cf = np.zeros(( data['mat_type'].shape[0], roi_in_data.shape[0], roi_in_data.shape[0] ))

            #cf_metric_i, cf_metric = 0, 'wpli'
            for cf_metric_i, cf_metric in enumerate(data['mat_type'].data):
                mat_cf[cf_metric_i,:,:] = dfc_pairs_to_mat(data_chunk.loc[cf_metric,:,:,:].data, data['pairs'].data, 'mean', rscore_computation)
                #mat_cf[cf_metric_i,:,:] = dfc_pairs_to_mat(data_chunk.loc[cf_metric,:,:,:].data, data['pairs'].data, 'trapz', rscore_computation=True)

            allband_data[export_type][band] = mat_cf

    return allband_data












########################################
######## GET FR_CV BASELINES ########
########################################


def get_FR_CV_baselines(sujet, electrode_recording_type, rscore_computation=False):

    baselines = {}

    #band = 'beta'
    for band in band_name_fc_dfc:

        #### extract data
        os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

        if electrode_recording_type == 'monopolaire':
            file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find('FR_CV') != -1 and i.find('bi') == -1)]
        if electrode_recording_type == 'bipolaire':
            file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find('FR_CV') != -1 and i.find('bi') != -1)]
        
        data = xr.open_dataarray(file_to_load[0])
        pair_unique, roi_in_data = get_pair_unique_and_roi_unique(data['pairs'].data)

        #### fill mat
        mat_cf = np.zeros(( data['mat_type'].shape[0], roi_in_data.shape[0], roi_in_data.shape[0] ))

        #cf_metric_i, cf_metric = 0, 'wpli'
        for cf_metric_i, cf_metric in enumerate(data['mat_type'].data):
            mat_cf[cf_metric_i,:,:] = dfc_pairs_to_mat(data.loc[cf_metric,:,:,:].data, data['pairs'].data, 'mean', rscore_computation)
            #mat_cf[cf_metric_i,:,:] = dfc_pairs_to_mat(data_chunk.loc[cf_metric,:,:,:].data, data['pairs'].data, 'trapz', rscore_computation=True)

        baselines[band] = mat_cf

    #### clean data from WM and ventricule
    baselines, roi_in_data = clean_data_WM_ventricule_FR_CV(baselines, roi_in_data)

    return baselines







################################
######## SAVE FIG ########
################################


def process_dfc_res(sujet, cond, baselines, electrode_recording_type, prms, FR_CV_normalized=True):

    print(f'######## {cond} DFC ########')

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    if electrode_recording_type == 'monopolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') == -1)]
    if electrode_recording_type == 'bipolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') != -1)]

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(xr.open_dataarray(file_to_load[0])['pairs'].data)
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_fc_dfc)

    ######## PHASE ########

    if electrode_recording_type == 'monopolaire' and os.path.exists(f'{sujet}_allphase_{cond}.nc'): 

        print(f'{sujet}_allphase_{cond} : ALREADY COMPUTED')

        allband_data = xr.open_dataarray(f'{sujet}_allphase_{cond}.nc')
        roi_names = allband_data['x'].data

    if electrode_recording_type == 'bipolaire' and os.path.exists(f'{sujet}_allphase_{cond}_bi.nc'):

        print(f'{sujet}_allphase_{cond} : ALREADY COMPUTED')

        allband_data = xr.open_dataarray(f'{sujet}_allphase_{cond}_bi.nc')
        roi_names = allband_data['x'].data

    if electrode_recording_type == 'monopolaire' and os.path.exists(f'{sujet}_allphase_{cond}.nc') == False or electrode_recording_type == 'bipolaire' and os.path.exists(f'{sujet}_allphase_{cond}_bi.nc') == False: 

        #### load data 
        allband_data = get_data_for_phase(sujet, cond, electrode_recording_type, roi_in_data, prms, rscore_computation=False)

        #### clean WM and ventricule
        allband_data, roi_names = clean_data_WM_ventricule(allband_data, roi_in_data)

        #### switch to xarray
        data_xr = np.zeros((len(list(allband_data.keys())), len(list(allband_data['pre'].keys())), allband_data['pre']['beta'].shape[0], allband_data['pre']['beta'].shape[1], allband_data['pre']['beta'].shape[1]))

        for phase_i, phase in enumerate(allband_data):
            
            for band_i, band in enumerate(allband_data[phase]):

                data_xr[phase_i, band_i, :, :, :] = allband_data[phase][band]
        
        dims = ['phase', 'band', 'cf_metrics', 'x', 'y']
        coords = [list(allband_data.keys()), list(allband_data['pre'].keys()), ['ispc', 'wpli'], roi_names, roi_names]
        allband_data = xr.DataArray(data_xr, dims=dims, coords=coords)

        #### save
        if electrode_recording_type == 'monopolaire':
            allband_data.to_netcdf(f'{sujet}_allphase_{cond}.nc')
        if electrode_recording_type == 'bipolaire':
            allband_data.to_netcdf(f'{sujet}_allphase_{cond}_bi.nc')

    #### define diff and phase to plot
    if cond == 'AC':
        phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
        phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

    if cond == 'SNIFF':
        phase_list = ['pre', 'resp_evnmt', 'post']
        phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

    if cond == 'AL':
        phase_list = ['pre', 'resp_evnmt', 'post']
        phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

    n_cols_raw = len(phase_list)
    n_cols_diff = len(phase_list_diff)

    #### normalize
    if FR_CV_normalized:

        allband_data_normalized = {}

        #phase_list_i = phase_list[0]
        for phase_list_i in phase_list:

            allband_data_normalized[phase_list_i] = {}
                
            for band in band_name_fc_dfc:

                allband_data_normalized[phase_list_i][band] = allband_data.loc[phase_list_i, band, :, :, :].data - baselines[band]

        allband_data = allband_data_normalized.copy()

        del allband_data_normalized

    #### substract data
    allband_data_diff = {}

    #phase_diff = 'pre-post'
    for phase_diff in phase_list_diff:

        allband_data_diff[phase_diff] = {}

        phase_diff_A, phase_diff_B = phase_diff.split('-')[0], phase_diff.split('-')[1]
            
        for band in band_name_fc_dfc:

            allband_data_diff[phase_diff][band] = allband_data[phase_diff_A][band] - allband_data[phase_diff_B][band]

    #### put 0 to matrix center
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for phase_list_i in phase_list:
                    
                for band in band_name_fc_dfc:

                    mat_to_clean = allband_data[phase_list_i][band][mat_type_i, :, :]

                    for roi_i in range(mat_to_clean.shape[0]):

                        mat_to_clean[roi_i,roi_i] = 0

                    allband_data[phase_list_i][band][mat_type_i, :, :] = mat_to_clean.copy()

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for phase_diff in phase_list_diff:

            for band in band_name_fc_dfc:

                mat_to_clean = allband_data_diff[phase_diff][band][mat_type_i, :, :]

                for roi_i in range(mat_to_clean.shape[0]):

                    mat_to_clean[roi_i,roi_i] = 0

                allband_data_diff[phase_diff][band][mat_type_i, :, :] = mat_to_clean.copy()

    #### identify scales
    scales = {}

    for phase in phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

            for band in band_name_fc_dfc:

                # mat_scaled = allband_data[phase][band][mat_type_i,:,:][allband_data[phase][band][mat_type_i,:,:] != 0]
                mat_scaled = allband_data[phase][band][mat_type_i,:,:]

                scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
                scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

            scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_name_fc_dfc:

            max_list = np.array(())

            for phase in phase_list:

                max_list = np.append(max_list, allband_data[phase][band][mat_type_i,:,:].max())
                max_list = np.append(max_list, np.abs(allband_data[phase][band][mat_type_i,:,:].min()))

            scales_abs[mat_type][band] = max_list.max()

    #### thresh alldata
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(allband_data)

    for phase in phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(allband_data[phase][band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allband_data[phase][band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[phase][band][mat_type_i,:,:].shape[1]):
                    for y in range(mat_dfc_clean[phase][band][mat_type_i,:,:].shape[1]):
                        if mat_dfc_clean[phase][band][mat_type_i,x,y] < thresh_up:
                            mat_dfc_clean[phase][band][mat_type_i,x,y] = 0
                        

    #### thresh alldata diff
    mat_dfc_clean_diff = copy.deepcopy(allband_data_diff)

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for phase_diff in phase_list_diff:

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(allband_data_diff[phase_diff][band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allband_data_diff[phase_diff][band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean_diff[phase_diff][band][mat_type_i,:,:].shape[1]):
                    for y in range(mat_dfc_clean_diff[phase_diff][band][mat_type_i,:,:].shape[1]):
                        if (mat_dfc_clean_diff[phase_diff][band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean_diff[phase_diff][band][mat_type_i,x,y] > thresh_down):
                            mat_dfc_clean_diff[phase_diff][band][mat_type_i,x,y] = 0

    ######## PLOT ########
    
    os.chdir(os.path.join(path_results, sujet, 'DFC', 'allcond'))

    #mat_type_i, mat_type = 1, 'wpli'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        ######## NO THRESH ########

        #### mat plot raw 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_raw, figsize=(15,15))

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond} {mat_type}')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond} {mat_type} bi')
        
        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list):

                ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')
                
                cax = ax.matshow(allband_data[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                fig.colorbar(cax, ax=ax)

                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_MAT_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'RAW_MAT_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_MAT_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'RAW_MAT_{cond}_{mat_type}_bi.png')
        
        plt.close('all')

        #### mat plot DIFF 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_diff, figsize=(15,15))

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond} {mat_type}')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond} {mat_type} bi')

        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list_diff):

                if n_cols_diff == 1:
                    ax = axs[r]
                else:
                    ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')

                cax = ax.matshow(allband_data_diff[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, cmap=cm.seismic)
                
                fig.colorbar(cax, ax=ax)

                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_MAT_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'DIFF_MAT_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_MAT_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'DIFF_MAT_{cond}_{mat_type}.png')
        
        plt.close('all')


        #### circle plot RAW
        nrows, ncols = n_band, n_cols_raw
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list):

                mne_connectivity.viz.plot_connectivity_circle(allband_data[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond}_{mat_type}', color='k')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
        
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_CIRCLE_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'RAW_CIRCLE_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_CIRCLE_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'RAW_CIRCLE_{cond}_{mat_type}_bi.png')

        plt.close('all')


        #### circle plot DIFF
        nrows, ncols = n_band, n_cols_diff
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list_diff):

                mne_connectivity.viz.plot_connectivity_circle(allband_data_diff[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                            vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond}_{mat_type}', color='k')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond}_{mat_type}_bi', color='k')
        
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_CIRCLE_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'DIFF_CIRCLE_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_CIRCLE_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'DIFF_CIRCLE_{cond}_{mat_type}_bi.png')
        
        plt.close('all')



        ######### THRESH #########

        #### mat plot raw 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_raw, figsize=(15,15))

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond} {mat_type} THRESH')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond} {mat_type} THRESH bi')
        
        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list):
                
                ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')
                
                cax = ax.matshow(mat_dfc_clean[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)

                fig.colorbar(cax, ax=ax)
                
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_MAT_THRESH_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'RAW_MAT_THRESH_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_MAT_THRESH_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'RAW_MAT_THRESH_{cond}_{mat_type}_bi.png')
        
        plt.close('all')

        #### mat plot DIFF 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_diff, figsize=(15,15))

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond} {mat_type} THRESH')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond} {mat_type} THRESH bi')

        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list_diff):
                
                if n_cols_diff == 1:
                    ax = axs[r]
                else:
                    ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')

                cax = ax.matshow(mat_dfc_clean_diff[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, cmap=cm.seismic)
                
                fig.colorbar(cax, ax=ax)

                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_MAT_THRESH_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'DIFF_MAT_THRESH_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_MAT_THRESH_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'DIFF_MAT_THRESH_{cond}_{mat_type}_bi.png')
        
        plt.close('all')


        #### circle plot RAW
        nrows, ncols = n_band, n_cols_raw
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list):

                mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond}_{mat_type} THRESH', color='k')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond}_{mat_type} THRESH bi', color='k')

        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_CIRCLE_THRESH_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'RAW_CIRCLE_THRESH_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'RAW_CIRCLE_THRESH_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'RAW_CIRCLE_THRESH_{cond}_{mat_type}_bi.png')  

        plt.close('all')


        #### circle plot DIFF
        nrows, ncols = n_band, n_cols_diff 
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list_diff):

                mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean_diff[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                            vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        if electrode_recording_type == 'monopolaire':
            plt.suptitle(f'{cond}_{mat_type} THRESH', color='k')
        if electrode_recording_type == 'bipolaire':
            plt.suptitle(f'{cond}_{mat_type} THRESH bi', color='k')
        
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()

        if electrode_recording_type == 'monopolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_CIRCLE_THRESH_{cond}_{mat_type}_norm.png')
            else:
                fig.savefig(f'DIFF_CIRCLE_THRESH_{cond}_{mat_type}.png')
        if electrode_recording_type == 'bipolaire':
            if FR_CV_normalized:
                fig.savefig(f'DIFF_CIRCLE_THRESH_{cond}_{mat_type}_bi_norm.png')
            else:
                fig.savefig(f'DIFF_CIRCLE_THRESH_{cond}_{mat_type}_bi.png')
        
        plt.close('all')















            
################################
######## SUMMARY ########
################################

def process_dfc_res_summary(sujet, baselines, prms, electrode_recording_type, FR_CV_normalized=True):

    print(f'######## SUMMARY DFC ########')

    #### CONNECTIVITY PLOT ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    if electrode_recording_type == 'monopolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') == -1)]
    if electrode_recording_type == 'bipolaire':
        file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') != -1)]

    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_fc_dfc)

    #### LOAD DATA ####

    print('#### LOAD DATA #####')

    os.chdir(os.path.join(path_precompute, 'allplot'))

    allcond_data = {}
    allcond_scales_abs = {}

    #### select conditions
    conditions = [cond for cond in prms['conditions'] if cond != 'FR_CV']

    for cond in conditions:

        #### define diff and phase to plot
        if cond == 'AC':
            phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
            phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

        if cond == 'SNIFF':
            phase_list = ['pre', 'resp_evnmt', 'post']
            phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

        if cond == 'AL':
            phase_list = ['pre', 'resp_evnmt', 'post']
            phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

        #### load data
        if electrode_recording_type == 'monopolaire': 

            allband_data = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs.nc')
            roi_names = allband_data['x'].data

        if electrode_recording_type == 'bipolaire':

            allband_data = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs_bi.nc')
            roi_names = allband_data['x'].data

        #### normalize
        if FR_CV_normalized:

            allband_data_normalized = {}

            #phase_list_i = phase_list[0]
            for phase_list_i in phase_list:

                allband_data_normalized[phase_list_i] = {}
                    
                for band in band_name_fc_dfc:

                    allband_data_normalized[phase_list_i][band] = allband_data.loc[phase_list_i, band, :, :].data - baselines[band]

            allband_data = allband_data_normalized

        #### scale abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_fc_dfc:

                max_list = np.array(())

                for phase in phase_list:

                    max_list = np.append(max_list, allband_data[phase][band][mat_type_i,:,:].max())
                    max_list = np.append(max_list, np.abs(allband_data[phase][band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

        allcond_scales_abs[cond] = scales_abs

        #### conpute diff
        allcond_data_diff_i = {}

        #phase_diff = 'pre-post'
        for phase_diff in phase_list_diff:

            allcond_data_diff_i[phase_diff] = {}

            phase_diff_A, phase_diff_B = phase_diff.split('-')[0], phase_diff.split('-')[1]
                
            for band in band_name_fc_dfc:

                allcond_data_diff_i[phase_diff][band] = allband_data[phase_diff_A][band] - allband_data[phase_diff_B][band]

        #### thresh
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean_i = copy.deepcopy(allcond_data_diff_i)

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for phase_diff in phase_list_diff:

                for band in band_name_fc_dfc:

                    thresh_up = np.percentile(allcond_data_diff_i[phase_diff][band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allcond_data_diff_i[phase_diff][band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean_i[phase_diff][band][mat_type_i,:,:].shape[1]):
                        for y in range(mat_dfc_clean_i[phase_diff][band][mat_type_i,:,:].shape[1]):
                            if (mat_dfc_clean_i[phase_diff][band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean_i[phase_diff][band][mat_type_i,x,y] > thresh_down):
                                mat_dfc_clean_i[phase_diff][band][mat_type_i,x,y] = 0

        #### fill res containers
        allcond_data[cond] = mat_dfc_clean_i

    #### PLOT ####

    print('#### PLOT ####')

    #### adjust scale
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_name_fc_dfc:

            max_list = np.array(())

            for cond in conditions:

                max_list = np.append(max_list, allcond_scales_abs[cond][mat_type][band])

            scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, sujet, 'DFC', 'summary'))

    n_rows = len(band_name_fc_dfc)
    n_cols = len(conditions)

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #phase_diff = phase_list_diff[0]
        for phase_diff in phase_list_diff:

            #### mat
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{mat_type} summary THRESH : {phase_diff}')
            if electrode_recording_type == 'bipolaire':
                plt.suptitle(f'{mat_type} summary THRESH : {phase_diff} bi')
            
            for r, band in enumerate(band_name_fc_dfc):
                for c, cond in enumerate(conditions):

                    if cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') == -1:
                        phase_diff_list = []
                        for phase_diff_i in phase_diff.split('-'):
                            if phase_diff_i.find('resp_evnmt') != -1:
                                phase_diff_list.append('resp_evnmt')
                            else:
                                phase_diff_list.append(phase_diff_i)
                        
                        phase_diff_sel = f'{phase_diff_list[0]}-{phase_diff_list[1]}'

                    elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') != -1:
                        continue

                    elif cond != 'AC' and phase_diff.find('1') == -1 and phase_diff.find('2') != -1:
                        continue

                    else:
                        phase_diff_sel = phase_diff

                    if n_cols == 1:
                        ax = axs[r]    
                    else:
                        ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{cond}')

                    cax = ax.matshow(allcond_data[cond][phase_diff_sel][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, cmap=cm.seismic)
                    
                    fig.colorbar(cax, ax=ax)

                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)
            # plt.show()

            if electrode_recording_type == 'monopolaire':
                if FR_CV_normalized:
                    fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_norm.png')
                else:
                    fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}.png')
            if electrode_recording_type == 'bipolaire':
                if FR_CV_normalized:
                    fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi_norm.png')
                else:
                    fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi.png')
            
            plt.close('all')

            #### circle plot
            fig, axs = plt.subplots(nrows=n_band, ncols=n_cols, subplot_kw=dict(polar=True))

            for r, band in enumerate(band_name_fc_dfc):

                for c, cond in enumerate(conditions):

                    if cond == 'AL' and phase_diff.find('resp') != -1:
                        continue

                    elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') == -1:
                        phase_diff_list = []
                        for phase_diff_i in phase_diff.split('-'):
                            if phase_diff_i.find('resp_evnmt') != -1:
                                phase_diff_list.append('resp_evnmt')
                            else:
                                phase_diff_list.append(phase_diff_i)
                        
                        phase_diff_sel = f'{phase_diff_list[0]}-{phase_diff_list[1]}'

                    elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') != -1:
                        continue

                    elif cond != 'AC' and phase_diff.find('1') == -1 and phase_diff.find('2') != -1:
                        continue

                    else:
                        phase_diff_sel = phase_diff

                    mne_connectivity.viz.plot_connectivity_circle(allcond_data[cond][phase_diff_sel][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=f'{cond} {band}', show=False, padding=7, ax=axs[r, c],
                                                vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{cond}_{mat_type}_THRESH : {phase_diff}', color='k')
            if electrode_recording_type == 'bipolaire':
                plt.suptitle(f'{cond}_{mat_type}_THRESH : {phase_diff} bi', color='k')
            
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()

            if electrode_recording_type == 'monopolaire':
                if FR_CV_normalized:
                    fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_norm.png')
                else:
                    fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}.png')
            if electrode_recording_type == 'bipolaire':
                if FR_CV_normalized:
                    fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi_norm.png')
                else:
                    fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi.png')
            
            plt.close('all')










################################
######## COMPILATION ########
################################


def res_DFC_compilation(sujet, electrode_recording_type):

    print(sujet)

    prms = get_params(sujet, electrode_recording_type)

    baselines = get_FR_CV_baselines(sujet, electrode_recording_type, rscore_computation=False)

    #### allcond
    #cond = 'AL'
    for cond in cond_FC_DFC:

        if cond == 'FR_CV':

            continue

        else:

            process_dfc_res(sujet, cond, baselines, electrode_recording_type, prms, FR_CV_normalized=True)

    #### summary
    process_dfc_res_summary(sujet, baselines, prms, electrode_recording_type, FR_CV_normalized=True)






################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[1]
        for sujet in sujet_list:

            res_DFC_compilation(sujet, electrode_recording_type)
            # execute_function_in_slurm_bash_mem_choice('n13_res_DFC', 'res_DFC_compilation', [sujet, electrode_recording_type], '15G')





    