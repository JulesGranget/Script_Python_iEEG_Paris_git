

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import copy
import mne_connectivity

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################################
######## COMPUTE DATA RESPI PHASE ########
################################################


#data_dfc, pairs, roi_in_data = data.loc[AL_i+1, cf_metric, :, select_time_vec], pairs, roi_in_data
def from_dfc_to_mat_conn_mean(data_dfc, pairs, roi_in_data):

    #### fill mat
    mat_cf = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = data_dfc[pairs == pair_to_find]
            x_rev = data_dfc[pairs == pair_to_find_rev]

            x_mean = np.vstack([x, x_rev]).mean(axis=0)
            val_to_place = np.trapz(x_mean)

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf





################################
######## PRECOMPUTE MAT ########
################################



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





def precompute_dfc_mat_AL_allplot(cond, electrode_recording_type):

    band_prep = 'wb'

    #### initiate containers
    mat_pairs_allplot = {}
    
    #cf_metric = 'ispc'
    for cf_metric in ['ispc', 'wpli']:

        mat_pairs_allplot[cf_metric] = {}

        for band, freq in freq_band_dict_FC_function[band_prep].items():

            print(cf_metric, band)

            #sujet = sujet_list[0]
            for sujet_i, sujet in enumerate(sujet_list):

                #### extract data
                os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

                if electrode_recording_type == 'monopolaire':
                    xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                if electrode_recording_type == 'bipolaire':
                    xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                
                #### concat pairs name
                if sujet_i == 0:
                    pairs_allplot = xr_dfc['pairs'].data
                else:
                    pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                #### extract data and concat
                mat = xr_dfc.loc[cf_metric,:,:,:].data

                del xr_dfc

                if sujet_i == 0:
                    mat_values_allplot = mat
                else:
                    mat_values_allplot = np.concatenate((mat_values_allplot, mat), axis=0)

                del mat

                #### mean
                pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)

                dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], mat_values_allplot.shape[1], mat_values_allplot.shape[-1] ))

                #x_i, x_name = 0, roi_in_data[0]
                for x_i, x_name in enumerate(roi_in_data_allplot):
                    #y_i, y_name = 2, roi_in_data[2]
                    for y_i, y_name in enumerate(roi_in_data_allplot):
                        if x_name == y_name:
                            continue

                        pair_to_find = f'{x_name}-{y_name}'
                        pair_to_find_rev = f'{y_name}-{x_name}'
                        
                        x = mat_values_allplot[pairs_allplot == pair_to_find, :, :]
                        x_rev = mat_values_allplot[pairs_allplot == pair_to_find_rev, :, :]

                        x_mean = np.vstack([x, x_rev]).mean(axis=0)

                        if np.isnan(x_mean).sum() > 1:
                            continue

                        #### identify pair name mean
                        try:
                            pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                        except:
                            pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                        dfc_mean_pair[pair_position, :, :] = x_mean

                mat_pairs_allplot[cf_metric][band] = dfc_mean_pair

            del mat_values_allplot
                    
    #### identify missing pairs with clean ROI
    for ROI_to_clean in ['WM', 'ventricule']:
        if ROI_to_clean in roi_in_data_allplot:
            roi_in_data_allplot = np.delete(roi_in_data_allplot, roi_in_data_allplot==ROI_to_clean)

    mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
    for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
        for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

            if ROI_row_name == ROI_col_name:
                mat_verif[ROI_row_i, ROI_col_i] = 1
                continue

            pair_name = f'{ROI_row_name}-{ROI_col_name}'
            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

            if pair_name in pair_unique_allplot:
                mat_verif[ROI_row_i, ROI_col_i] = 1
            if pair_name_rev in pair_unique_allplot:
                mat_verif[ROI_row_i, ROI_col_i] = 1

    if debug:
        plt.matshow(mat_verif)
        plt.show()

    #### export missig matrix
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))
    plt.matshow(mat_verif)
    plt.savefig('missing_mat.png')
    plt.close('all')

    #### clean pairs
    pairs_to_keep = []
    for pair_i, pair in enumerate(pair_unique_allplot):
        if pair.split('-')[0] not in ['WM', 'ventricule'] and pair.split('-')[1] not in ['WM', 'ventricule']:
            pairs_to_keep.append(pair_i)

    for AL_i in range(3):
        for cf_metric in ['ispc', 'wpli']:
            for band in mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric].keys():
                mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band] = mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band][pairs_to_keep, :, :].copy()

    pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

    #### save
    os.chdir(os.path.join(path_precompute, 'allplot'))
    data_dfc_allpairs = np.zeros((3, 2, len(band_name_fc_dfc), pair_unique_allplot.shape[0], mat_pairs_allplot['AL_1']['ispc']['beta'].shape[1], mat_pairs_allplot['AL_1']['ispc']['beta'].shape[2]))
    
    for AL_i in range(3):
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric].keys()):
                data_dfc_allpairs[AL_i, cf_metric_i, band_i, :, :, :] = mat_pairs_allplot[f'AL_{AL_i+1}'][cf_metric][band]

    dims = ['AL_num', 'cf_metric', 'band', 'pairs', 'nfrex', 'time']
    coords = [range(3), ['ispc', 'wpli'], band_name_fc_dfc, pair_unique_allplot, range(nfrex_dfc), range(mat_pairs_allplot['AL_1']['ispc']['beta'].shape[2])]
    xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

    if electrode_recording_type == 'monopolaire':
        xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
    if electrode_recording_type == 'bipolaire':
        xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot







def precompute_dfc_mat_allplot(cond, electrode_recording_type):

    os.chdir(os.path.join(path_precompute, 'allplot', 'DFC'))

    if electrode_recording_type == 'monopolaire' and os.path.exists(f'allcond_dfc_{cond}_allpairs.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

        return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot

    elif electrode_recording_type == 'bipolaire' and os.path.exists(f'allcond_dfc_{cond}_allpairs_bi.nc'):
        
        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs_bi.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

        return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot

    elif cond == 'AL':

        xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot = precompute_dfc_mat_AL_allplot(cond, electrode_recording_type)

        return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot

    else:

        #### initiate containers
        mat_pairs_allplot = {}

        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:

            mat_pairs_allplot[cf_metric] = {}

            for band, freq in freq_band_dict_FC_function[band_prep].items():

                print(cf_metric, band)

                #sujet = sujet_list[0]
                for sujet_i, sujet in enumerate(sujet_list):

                    #### extract data
                    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

                    if electrode_recording_type == 'monopolaire':
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                    if electrode_recording_type == 'bipolaire':
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                    
                    #### concat pairs name
                    if sujet_i == 0:
                        pairs_allplot = xr_dfc['pairs'].data
                    else:
                        pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                    #### extract data and concat
                    mat = xr_dfc.loc[cf_metric,:,:,:].data

                    del xr_dfc

                    if sujet_i == 0:
                        mat_values_allplot = mat
                    else:
                        mat_values_allplot = np.concatenate((mat_values_allplot, mat), axis=0)

                    del mat

                #### mean
                pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)

                dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], nfrex_dfc, mat_values_allplot.shape[-1] ))

                #x_i, x_name = 0, roi_in_data[0]
                for x_i, x_name in enumerate(roi_in_data_allplot):
                    #y_i, y_name = 2, roi_in_data[2]
                    for y_i, y_name in enumerate(roi_in_data_allplot):
                        if x_name == y_name:
                            continue

                        pair_to_find = f'{x_name}-{y_name}'
                        pair_to_find_rev = f'{y_name}-{x_name}'
                        
                        x = mat_values_allplot[pairs_allplot == pair_to_find, :, :]
                        x_rev = mat_values_allplot[pairs_allplot == pair_to_find_rev, :, :]

                        x_mean = np.vstack([x, x_rev]).mean(axis=0)

                        if np.isnan(x_mean).sum() > 1:
                            continue

                        #### identify pair name mean
                        try:
                            pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                        except:
                            pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                        dfc_mean_pair[pair_position, :, :] = x_mean

                mat_pairs_allplot[cf_metric][band] = dfc_mean_pair

                del mat_values_allplot
                    
        #### identify missing pairs with clean ROI
        for ROI_to_clean in ['WM', 'ventricule']:
            if ROI_to_clean in roi_in_data_allplot:
                roi_in_data_allplot = np.delete(roi_in_data_allplot, roi_in_data_allplot==ROI_to_clean)

        mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
        for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
            #ROI_col_i, ROI_col_name = 1, ROI_list[1]
            for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

                if ROI_row_name == ROI_col_name:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                    continue

                pair_name = f'{ROI_row_name}-{ROI_col_name}'
                pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                if pair_name in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1
                if pair_name_rev in pair_unique_allplot:
                    mat_verif[ROI_row_i, ROI_col_i] = 1

        if debug:
            plt.matshow(mat_verif)
            plt.show()

        #### export missig matrix
        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))
        plt.matshow(mat_verif)
        plt.savefig('missing_mat.png')
        plt.close('all')

        #### clean pairs
        pairs_to_keep = []
        for pair_i, pair in enumerate(pair_unique_allplot):
            if pair.split('-')[0] in ROI_for_DFC_plot and pair.split('-')[1] in ROI_for_DFC_plot:
                pairs_to_keep.append(pair_i)

        for cf_metric in ['ispc', 'wpli']:
            for band in mat_pairs_allplot[cf_metric].keys():
                mat_pairs_allplot[cf_metric][band] = mat_pairs_allplot[cf_metric][band][pairs_to_keep, :, :].copy()

        pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

        #### save
        os.chdir(os.path.join(path_precompute, 'allplot', 'DFC'))
        data_dfc_allpairs = np.zeros((2, len(band_name_fc_dfc), pair_unique_allplot.shape[0], mat_pairs_allplot['ispc']['beta'].shape[1], mat_pairs_allplot['ispc']['beta'].shape[2]))
        
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[cf_metric].keys()):
                data_dfc_allpairs[cf_metric_i, band_i, :, :, :] = mat_pairs_allplot[cf_metric][band]

        dims = ['cf_metric', 'band', 'pairs', 'nfrex', 'time']
        coords = [['ispc', 'wpli'], band_name_fc_dfc, pair_unique_allplot, range(nfrex_dfc), range(mat_pairs_allplot['ispc']['beta'].shape[2])]
        xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

        if electrode_recording_type == 'monopolaire':
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
        if electrode_recording_type == 'bipolaire':
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    return xr_dfc_allpairs, pair_unique_allplot, roi_in_data_allplot














#data, pairs, roi_in_data = xr_dfc, xr_dfc['pairs'], ROI
def graph_AL_chunk(sujet, data, cf_metric, pairs, roi_in_data):

    source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))
    time_AL = pd.read_excel(f'{sujet}_count_session.xlsx')
    time_AL = time_AL.iloc[:,-3:]

    AL_chunks = {}

    for phase in ['pre', 'resp_evnmt', 'post']:

        AL_chunks_i = []

        for AL_i in range(time_AL.columns.shape[0]):

            time_vec = np.linspace(0, time_AL.iloc[0, AL_i], n_points_AL_interpolation)

            if phase == 'pre':
                select_time_vec = time_vec < AL_extract_time
            elif phase == 'resp_evnmt':
                time_half_AL = time_AL.iloc[0, AL_i] / 2
                select_time_vec = (time_vec > time_half_AL) & (time_vec < (time_half_AL + AL_extract_time))
            elif phase == 'post':
                select_time_vec = (time_vec < time_AL.iloc[0, AL_i]) & (time_vec > (time_AL.iloc[0, AL_i] - AL_extract_time))

            AL_chunks_i.append(from_dfc_to_mat_conn_mean(data.loc[AL_i+1, cf_metric, :, select_time_vec].data, pairs, roi_in_data))

        AL_chunks[phase] = np.stack((AL_chunks_i[0], AL_chunks_i[1], AL_chunks_i[2])).mean(axis=0)

    os.chdir(source)

    return AL_chunks
           



#dfc_data, pairs, compute_mode = xr_dfc_allpairs.loc[cf_metric, band, :, :, :].data, pair_unique_allplot, 'mean'
def dfc_pairs_to_mat(dfc_data, pairs, compute_mode):

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

            if np.isnan(x_mean).sum() > 1:
                continue

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

            if np.isnan(x_mean).sum() > 1:
                continue

            # if rscore_computation:
            #     x_mean_rscore = rscore_mat(x_mean)
            # else:
            #     x_mean_rscore = x_mean

            if compute_mode == 'mean':
                val_to_place = x_mean.mean(axis=1).mean(0)
            if compute_mode == 'trapz':
                val_to_place = np.trapz(x_mean, axis=1).mean(0)

            mat_dfc[x_i, y_i] = val_to_place

    return mat_dfc
    








def chunk_AL(mat, sujet, export_type, band, cf_metric):

    source = os.getcwd()

    #### extract AL time
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    time_AL = pd.read_excel(f'{sujet}_count_session.xlsx')
    time_AL = time_AL.iloc[:,-3:]

    #### chunk
    Al_chunks = []

    #AL_i = 0
    for AL_i in range(time_AL.columns.shape[0]):

        time_vec = np.linspace(0, time_AL.iloc[0, AL_i], n_points_AL_interpolation)

        if export_type == 'pre':
            select_time_vec = time_vec < AL_extract_time
        elif export_type == 'resp_evnmt':
            time_half_AL = time_AL.iloc[0, AL_i] / 2
            select_time_vec = (time_vec > time_half_AL) & (time_vec < (time_half_AL + AL_extract_time))
        elif export_type == 'post':
            select_time_vec = (time_vec < time_AL.iloc[0, AL_i]) & (time_vec > (time_AL.iloc[0, AL_i] - AL_extract_time))

        #### resample
        f = scipy.interpolate.interp1d(np.linspace(0, 1, select_time_vec.sum()), mat.loc[AL_i, cf_metric, band, :, :, select_time_vec].data, kind='linear') # exist different type of kind
        data_chunk = f(np.linspace(0, 1, n_points_AL_chunk))

        Al_chunks.append(data_chunk)

    data_chunk_mean = np.stack((Al_chunks[0], Al_chunks[1], Al_chunks[2])).mean(axis=0)

    data_chunk_mean = xr.DataArray(data_chunk_mean, dims=['pairs', 'nfrex', 'time'], coords=[mat.coords['pairs'], np.arange(mat.shape[-2]), np.arange(n_points_AL_chunk)] )

    os.chdir(source)

    return data_chunk_mean
           






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






#mat, pairs, roi_in_data = xr_pairs_allplot, pair_unique_allplot, roi_in_data_allplot
def precompute_dfc_mat_allplot_phase(mat, pairs, cond, baselines, electrode_recording_type, rscore_computation=False):

    band_prep = 'wb'

    #### define diff and phase to plot
    if cond == 'AC':
        phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

    if cond == 'SNIFF':
        phase_list = ['pre', 'resp_evnmt', 'post']

    if cond == 'AL':
        phase_list = ['resp_evnmt_1', 'resp_evnmt_2']

    #### initiate containers
    mat_phase = {}

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

        mat_phase[cf_metric] = {}

        #band, freq = 'beta', [10,40]
        for band, freq in freq_band_dict_FC_function[band_prep].items():

            mat_phase[cf_metric][band] = {}

            #phase = 'pre'
            for phase in phase_list:

                #### chunk
                extraction_time = get_phase_extraction()[cond][phase]

                data_chunk = mat.loc[cf_metric, band, :, :, extraction_time[0]:extraction_time[-1]].values

                #### fill mat
                mat_phase[cf_metric][band][phase] = dfc_pairs_to_mat(data_chunk, pairs, 'mean') - baselines[cf_metric][band]

    return mat_phase















########################################
######## FR_CV BASELINES ########
########################################



def precompute_baselines_allplot(electrode_recording_type, rscore_computation=False):

    cond = 'FR_CV'
    band_prep = 'wb'

    os.chdir(os.path.join(path_precompute, 'allplot', 'DFC'))

    if electrode_recording_type == 'monopolaire' and os.path.exists(f'allcond_dfc_FR_CV_allpairs.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

    elif electrode_recording_type == 'bipolaire' and os.path.exists(f'allcond_dfc_FR_CV_allpairs_bi.nc'):

        xr_dfc_allpairs = xr.open_dataarray(f'allcond_dfc_{cond}_allpairs_bi.nc')
        pair_unique_allplot = xr_dfc_allpairs['pairs']
        pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pair_unique_allplot)

    else:

        #### initiate containers
        mat_pairs_allplot = {}

        #cf_metric = 'ispc'
        for cf_metric in ['ispc', 'wpli']:
            mat_pairs_allplot[cf_metric] = {}
            #band, freq = 'theta', [4,8]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                print(cf_metric, band)

                #### load allpairs
                #sujet_i, sujet = 0, sujet_list[0]
                for sujet_i, sujet in enumerate(sujet_list):

                    #### extract data
                    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

                    if electrode_recording_type == 'monopolaire':
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                    if electrode_recording_type == 'bipolaire':
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')
                    
                    #### concat pairs name
                    if sujet_i == 0:
                        pairs_allplot = xr_dfc['pairs'].data
                    else:
                        pairs_allplot = np.concatenate((pairs_allplot, xr_dfc['pairs'].data), axis=0)

                    #### extract data and concat
                    if sujet_i == 0:
                        allpairs_sujet_i = xr_dfc.loc[cf_metric,:,:,:].data
                    else:
                        allpairs_sujet_i = np.concatenate((allpairs_sujet_i, xr_dfc.loc[cf_metric,:,:,:].data), axis=0)

                    del xr_dfc

                #### mean allpairs
                pair_unique_allplot, roi_in_data_allplot = get_pair_unique_and_roi_unique(pairs_allplot)

                dfc_mean_pair = np.zeros(( pair_unique_allplot.shape[0], nfrex_dfc, stretch_point_TF ))

                #x_i, x_name = 0, roi_in_data[0]
                for x_i, x_name in enumerate(roi_in_data_allplot):
                    #y_i, y_name = 2, roi_in_data[2]
                    for y_i, y_name in enumerate(roi_in_data_allplot):
                        if x_name == y_name:
                            continue

                        pair_to_find = f'{x_name}-{y_name}'
                        pair_to_find_rev = f'{y_name}-{x_name}'
                        
                        x = allpairs_sujet_i[pairs_allplot == pair_to_find, :, :]
                        x_rev = allpairs_sujet_i[pairs_allplot == pair_to_find_rev, :, :]

                        x_mean = np.vstack([x, x_rev]).mean(axis=0)

                        if np.isnan(x_mean).sum() > 1:
                            continue

                        #### identify pair name mean
                        try:
                            pair_position = np.where(pair_unique_allplot == pair_to_find)[0][0]
                        except:
                            pair_position = np.where(pair_unique_allplot == pair_to_find_rev)[0][0]

                        dfc_mean_pair[pair_position, :, :] = x_mean

                mat_pairs_allplot[cf_metric][band] = dfc_mean_pair

                del allpairs_sujet_i        

        if debug :

            mat_verif = np.zeros(( roi_in_data_allplot.shape[0], roi_in_data_allplot.shape[0]))
            for ROI_row_i, ROI_row_name in enumerate(roi_in_data_allplot):
                #ROI_col_i, ROI_col_name = 1, ROI_list[1]
                for ROI_col_i, ROI_col_name in enumerate(roi_in_data_allplot):

                    if ROI_row_name == ROI_col_name:
                        mat_verif[ROI_row_i, ROI_col_i] = 1
                        continue

                    pair_name = f'{ROI_row_name}-{ROI_col_name}'
                    pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                    if pair_name in pair_unique_allplot:
                        mat_verif[ROI_row_i, ROI_col_i] = 1
                    if pair_name_rev in pair_unique_allplot:
                        mat_verif[ROI_row_i, ROI_col_i] = 1

            plt.matshow(mat_verif)
            plt.show()

        #### clean pairs
        pairs_to_keep = []
        for pair_i, pair in enumerate(pair_unique_allplot):
            if pair.split('-')[0] in ROI_for_DFC_plot and pair.split('-')[1] in ROI_for_DFC_plot:
                pairs_to_keep.append(pair_i)

        for cf_metric in ['ispc', 'wpli']:
            for band in freq_band_dict_FC_function[band_prep].keys():
                mat_pairs_allplot[cf_metric][band] = mat_pairs_allplot[cf_metric][band][pairs_to_keep, :, :].copy()

        pair_unique_allplot = pair_unique_allplot[pairs_to_keep]

        #### save
        os.chdir(os.path.join(path_precompute, 'allplot', 'DFC'))
        data_dfc_allpairs = np.zeros((2, len(band_name_fc_dfc), pair_unique_allplot.shape[0], nfrex_dfc, stretch_point_TF))
        
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            for band_i, band in enumerate(mat_pairs_allplot[cf_metric].keys()):
                data_dfc_allpairs[cf_metric_i, band_i, :, :] = mat_pairs_allplot[cf_metric][band]

        dims = ['cf_metric', 'band', 'pairs', 'nfrex', 'time']
        coords = [['ispc', 'wpli'], band_name_fc_dfc, pair_unique_allplot, range(nfrex_dfc), range(mat_pairs_allplot['ispc']['beta'].shape[2])]
        xr_dfc_allpairs = xr.DataArray(data_dfc_allpairs, coords=coords, dims=dims)

        if electrode_recording_type == 'monopolaire':
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs.nc')
        if electrode_recording_type == 'bipolaire':
            xr_dfc_allpairs.to_netcdf(f'allcond_dfc_{cond}_allpairs_bi.nc')

    #### compute dfc mat baselines
    mat_baselines = {}

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

        mat_baselines[cf_metric] = {}

        #band, freq = 'beta', [10,40]
        for band, freq in freq_band_dict_FC_function[band_prep].items():

            mat_baselines[cf_metric][band] = dfc_pairs_to_mat(xr_dfc_allpairs.loc[cf_metric, band, :, :, :].data, pair_unique_allplot, 'mean')

    pair_unique, roi_in_data = get_pair_unique_and_roi_unique(pair_unique_allplot)

    return mat_baselines, roi_in_data










################################
######## SAVE FIG ########
################################



#mat = mat_allplot_reduced
def save_fig_dfc_allplot(cond, mat_phase, roi_in_data_allplot, electrode_recording_type, FR_CV_normalized=True, plot_circle=False, plot_thresh=False):

    roi_names = roi_in_data_allplot

    #### get params
    cf_metrics_list = ['ispc', 'wpli']
    n_band = len(band_name_fc_dfc)

    #### define diff and phase to plot
    if cond == 'AC':
        phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

    if cond == 'SNIFF':
        phase_list = ['pre', 'resp_evnmt', 'post']

    if cond == 'AL':
        phase_list = ['resp_evnmt_1', 'resp_evnmt_2']

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {}

        for band in band_name_fc_dfc:

            scales[mat_type][band] = {}

            max_list = np.array([])
            
            for phase in phase_list:

                max_list = np.append(max_list, np.abs(mat_phase[mat_type][band][phase].min()))
                max_list = np.append(max_list, mat_phase[mat_type][band][phase].max())

            scales[mat_type][band]['vmin'], scales[mat_type][band]['vmax'] = -max_list.max(), max_list.max()

    #### thresh alldata
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(mat_phase)

    for phase in phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(mat_phase[mat_type][band][phase].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(mat_phase[mat_type][band][phase].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[mat_type][band][phase].shape[1]):
                    for y in range(mat_dfc_clean[mat_type][band][phase].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[mat_type][band][phase][x,y] < thresh_up:
                                mat_dfc_clean[mat_type][band][phase][x,y] = 0
                        else:
                            if (mat_dfc_clean[mat_type][band][phase][x,y] < thresh_up) & (mat_dfc_clean[mat_type][band][phase][x,y] > thresh_down):
                                mat_dfc_clean[mat_type][band][phase][x,y] = 0

    ######## PLOT #######

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))

    #### plot    
    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        ######## NO THRESH ########

        #### mat plot raw 
        fig, axs = plt.subplots(nrows=n_band, ncols=len(phase_list), figsize=(15,15))

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
                
                cax = ax.matshow(mat_phase[mat_type][band][phase], vmin=scales[mat_type][band]['vmin'], vmax=scales[mat_type][band]['vmax'], cmap=cm.seismic)

                if c == len(phase_list)-1:
                        fig.colorbar(cax, ax=ax)

                ax.set_xticks(np.arange(roi_names.shape[0]))
                ax.set_xticklabels(roi_names)
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
                plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45, ha="left", va="center",rotation_mode="anchor")

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


        #### circle plot RAW
        if plot_circle:

            nrows, ncols = n_band, len(phase_list)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

            for r, band in enumerate(band_name_fc_dfc):

                for c, phase in enumerate(phase_list):

                    mne_connectivity.viz.plot_connectivity_circle(mat_phase[mat_type][band][phase], node_names=roi_names, n_lines=None, 
                                                title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                vmin=scales[mat_type][band]['vmin'], vmax=scales[mat_type][band]['vmax'], colormap=cm.seismic, facecolor='w', 
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


        ######### THRESH #########

        if plot_thresh:

            #### mat plot raw 
            fig, axs = plt.subplots(nrows=n_band, ncols=len(phase_list), figsize=(15,15))

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
                    
                    cax = ax.matshow(mat_dfc_clean[mat_type][band][phase], vmin=scales[mat_type][band]['vmin'], vmax=scales[mat_type][band]['vmax'], cmap=cm.seismic)

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

            #### circle plot RAW
            if plot_circle:

                nrows, ncols = n_band, len(phase_list)
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                for r, band in enumerate(band_name_fc_dfc):

                    for c, phase in enumerate(phase_list):

                        mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band][phase], node_names=roi_names, n_lines=None, 
                                                    title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                    vmin=scales[mat_type][band]['vmin'], vmax=scales[mat_type][band]['vmax'], colormap=cm.seismic, facecolor='w', 
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













            
################################
######## SUMMARY ########
################################



# def process_dfc_res_summary(cond_to_plot, electrode_recording_type):

#     print(f'######## SUMMARY DFC ########')

#     #### CONNECTIVITY PLOT ####

#     #### get params
#     os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    
#     cf_metrics_list = ['ispc', 'wpli']

#     #### load allcond data 
#     allcond_data = {}
#     allcond_scales_abs = {}
#     allcond_ROI_list = {}

#     for cond in cond_to_plot:

#         #### load data
#         xr_pairs_allplot, pair_unique_allplot, ROI_list = precompute_dfc_mat_allplot(cond, electrode_recording_type)
#         allcond_data_i = precompute_dfc_mat_allplot_phase(xr_pairs_allplot, pair_unique_allplot, cond, baselines, electrode_recording_type, rscore_computation=False)
#         allcond_ROI_list[cond] = ROI_list

#         #### define diff and phase to plot
#         if cond == 'AC':
#             phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
#             phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

#         if cond == 'SNIFF':
#             phase_list = ['pre', 'resp_evnmt', 'post']
#             phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

#         if cond == 'AL':
#             phase_list = ['pre', 'resp_evnmt', 'post']
#             phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

#         #### scale abs
#         scales_abs = {}

#         for mat_type_i, mat_type in enumerate(cf_metrics_list):

#             scales_abs[mat_type] = {}

#             for band in band_name_fc_dfc:

#                 max_list = np.array(())

#                 for phase in phase_list:

#                     max_list = np.append(max_list, allcond_data_i[mat_type][band][phase].max())
#                     max_list = np.append(max_list, np.abs(allcond_data_i[mat_type][band][phase].min()))

#                 scales_abs[mat_type][band] = max_list.max()

#         allcond_scales_abs[cond] = scales_abs

#         #### thresh
#         percentile_thresh_up = 99
#         percentile_thresh_down = 1

#         mat_dfc_clean_i = copy.deepcopy(allcond_data_diff_i)

#         #mat_type_i, mat_type = 0, 'ispc'
#         for mat_type_i, mat_type in enumerate(cf_metrics_list):

#             for phase_diff in phase_list_diff:

#                 for band in band_name_fc_dfc:

#                     thresh_up = np.percentile(allcond_data_diff_i[mat_type][band][phase_diff].reshape(-1), percentile_thresh_up)
#                     thresh_down = np.percentile(allcond_data_diff_i[mat_type][band][phase_diff].reshape(-1), percentile_thresh_down)

#                     for x in range(mat_dfc_clean_i[mat_type][band][phase_diff].shape[1]):
#                         for y in range(mat_dfc_clean_i[mat_type][band][phase_diff].shape[1]):
#                             if (mat_dfc_clean_i[mat_type][band][phase_diff][x,y] < thresh_up) & (mat_dfc_clean_i[mat_type][band][phase_diff][x,y] > thresh_down):
#                                 mat_dfc_clean_i[mat_type][band][phase_diff][x,y] = 0

#         #### fill res containers
#         allcond_data[cond] = mat_dfc_clean_i

#     #### adjust scale
#     scales_abs = {}

#     for mat_type_i, mat_type in enumerate(cf_metrics_list):

#         scales_abs[mat_type] = {}

#         for band in band_name_fc_dfc:

#             max_list = np.array(())

#             for cond in cond_to_plot:

#                 max_list = np.append(max_list, allcond_scales_abs[cond][mat_type][band])

#             scales_abs[mat_type][band] = max_list.max()

#     #### plot
#     os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'summary'))

#     n_rows = len(band_name_fc_dfc)
#     n_cols = len(cond_to_plot)

#     phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

#     #mat_type_i, mat_type = 0, 'ispc'
#     for mat_type_i, mat_type in enumerate(cf_metrics_list):

#         #phase_diff = phase_list_diff[0]
#         for phase_diff in phase_list_diff:

#             #### mat
#             fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))

#             if electrode_recording_type == 'monopolaire':
#                 plt.suptitle(f'{mat_type} summary THRESH : {phase_diff}')
#             if electrode_recording_type == 'bipolaire':
#                 plt.suptitle(f'{mat_type} summary THRESH : {phase_diff} bi')
            
#             for r, band in enumerate(band_name_fc_dfc):
#                 for c, cond in enumerate(conditions):

#                     if cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') == -1:
#                         phase_diff_list = []
#                         for phase_diff_i in phase_diff.split('-'):
#                             if phase_diff_i.find('resp_evnmt') != -1:
#                                 phase_diff_list.append('resp_evnmt')
#                             else:
#                                 phase_diff_list.append(phase_diff_i)
                        
#                         phase_diff_sel = f'{phase_diff_list[0]}-{phase_diff_list[1]}'

#                     elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') != -1:
#                         continue

#                     elif cond != 'AC' and phase_diff.find('1') == -1 and phase_diff.find('2') != -1:
#                         continue

#                     else:
#                         phase_diff_sel = phase_diff

#                     if n_cols == 1:
#                         ax = axs[r]    
#                     else:
#                         ax = axs[r, c]

#                     if c == 0:
#                         ax.set_ylabel(band)
#                     if r == 0:
#                         ax.set_title(f'{cond}')

#                     cax = ax.matshow(allcond_data[cond][mat_type][band][phase_diff_sel], vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, cmap=cm.seismic)
                    
#                     fig.colorbar(cax, ax=ax)

#                     if c == 0:
#                         ax.set_yticks(np.arange(allcond_ROI_list[cond].shape[0]))
#                         ax.set_yticklabels(allcond_ROI_list[cond])
#             # plt.show()

#             if electrode_recording_type == 'monopolaire':
#                 if FR_CV_normalized:
#                     fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_norm.png')
#                 else:
#                     fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}.png')
#             if electrode_recording_type == 'bipolaire':
#                 if FR_CV_normalized:
#                     fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi_norm.png')
#                 else:
#                     fig.savefig(f'MAT_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi.png')
            
#             plt.close('all')

#             #### circle plot
#             fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw=dict(polar=True))

#             for r, band in enumerate(band_name_fc_dfc):

#                 for c, cond in enumerate(conditions):

#                     if cond == 'AL' and phase_diff.find('resp') != -1:
#                         continue

#                     elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') == -1:
#                         phase_diff_list = []
#                         for phase_diff_i in phase_diff.split('-'):
#                             if phase_diff_i.find('resp_evnmt') != -1:
#                                 phase_diff_list.append('resp_evnmt')
#                             else:
#                                 phase_diff_list.append(phase_diff_i)
                        
#                         phase_diff_sel = f'{phase_diff_list[0]}-{phase_diff_list[1]}'

#                     elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') != -1:
#                         continue

#                     elif cond != 'AC' and phase_diff.find('1') == -1 and phase_diff.find('2') != -1:
#                         continue

#                     else:
#                         phase_diff_sel = phase_diff

#                     mne_connectivity.viz.plot_connectivity_circle(allcond_data[cond][phase_diff_sel][band][mat_type_i,:,:], node_names=allcond_ROI_list[cond], n_lines=None, 
#                                                 title=f'{cond} {band}', show=False, padding=7, ax=axs[r, c],
#                                                 vmin=-scales_abs[mat_type][band]/2, vmax=scales_abs[mat_type][band]/2, colormap=cm.seismic, facecolor='w', 
#                                                 textcolor='k')

#             if electrode_recording_type == 'monopolaire':
#                 plt.suptitle(f'{cond}_{mat_type}_THRESH : {phase_diff}', color='k')
#             if electrode_recording_type == 'bipolaire':
#                 plt.suptitle(f'{cond}_{mat_type}_THRESH : {phase_diff} bi', color='k')
            
#             fig.set_figheight(10)
#             fig.set_figwidth(12)
#             # fig.show()

#             if electrode_recording_type == 'monopolaire':
#                 if FR_CV_normalized:
#                     fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_norm.png')
#                 else:
#                     fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}.png')
#             if electrode_recording_type == 'bipolaire':
#                 if FR_CV_normalized:
#                     fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi_norm.png')
#                 else:
#                     fig.savefig(f'CIRCLE_{sujet}_summary_{phase_diff}_TRESH_{mat_type}_bi.png')
            
#             plt.close('all')









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        FR_CV_normalized = True
        band_prep = 'wb'
        cond_to_plot = ['AC', 'SNIFF', 'AL']

        #### get baselines
        baselines, roi_in_data = precompute_baselines_allplot(electrode_recording_type, rscore_computation=False)

        if debug:

            for metric in ['ispc', 'wpli']:
    
                for band in freq_band_dict_FC_function[band_prep].keys():

                    fig, ax = plt.subplots()
                    vmin = baselines[metric][band][baselines[metric][band] != 0].min()
                    vmax = baselines[metric][band][baselines[metric][band] != 0].max()
                    cax = ax.matshow(baselines[metric][band], vmin=vmin, vmax=vmax)
                    fig.colorbar(cax)
                    ax.set_xticks(np.arange(roi_in_data.shape[0]))
                    ax.set_xticklabels(roi_in_data)
                    ax.set_yticks(np.arange(roi_in_data.shape[0]))
                    ax.set_yticklabels(roi_in_data)
                    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
                            ha="left", va="center",rotation_mode="anchor")
                    ax.set_title(f'{metric}, {band}')
                    fig.tight_layout()
                    plt.show()

        #### save fig
        #cond = 'AL'
        for cond in cond_to_plot:
            
            print(cond, electrode_recording_type)
                
            xr_pairs_allplot, pair_unique_allplot, roi_in_data_allplot = precompute_dfc_mat_allplot(cond, electrode_recording_type)
            mat_phase = precompute_dfc_mat_allplot_phase(xr_pairs_allplot, pair_unique_allplot, cond, baselines, electrode_recording_type, rscore_computation=False)

            save_fig_dfc_allplot(cond, mat_phase, roi_in_data_allplot, electrode_recording_type, FR_CV_normalized=True)

        # process_dfc_res_summary(roi_in_data_allplot, cond_to_plot, electrode_recording_type)








