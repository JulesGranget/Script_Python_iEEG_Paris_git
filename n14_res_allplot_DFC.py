

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import copy


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################################
######## COMPUTE DATA RESPI PHASE ########
################################################


#data_dfc, pairs, roi_in_data = data_chunk.loc[cf_metric,:,:].data, data['pairs'].data, roi_in_data
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



def precompute_dfc_mat_allplot(cond):

    #### initiate containers
    mat_allplot = {}
    mat_pairs_allplot = {}

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i)
        if cond in prms_i['conditions']:
            if cond == 'FR_CV':
                continue
            sujet_list_selected.append(sujet_i)

    #cf_metric = 'ispc'
    for cf_metric in ['ispc', 'wpli']:
        mat_allplot[cf_metric] = {}
        mat_pairs_allplot[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot[cf_metric][band] = np.array([])
                    mat_pairs_allplot[cf_metric][band] = np.array([])

                    #sujet = sujet_list_selected[0]
                    for sujet in sujet_list_selected:

                        #### extract data
                        os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                        ROI = xr_dfc['x'].data

                        #### generate mat ROI names
                        for row_i, ROI_row_i in enumerate(ROI):
                            
                            mat_roi_row_i = np.array([])
                            
                            for ROI_col_i in ROI:
                                
                                pair_i = f'{ROI_row_i}-{ROI_col_i}'
                                mat_roi_row_i = np.append(mat_roi_row_i, pair_i)

                            if row_i == 0:
                                mat_roi = mat_roi_row_i
                            else:
                                mat_roi = np.vstack([mat_roi, mat_roi_row_i])

                        #### extract data from mat
                        mat = xr_dfc.loc[cf_metric,:,:].data
                        mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                        mat_pairs_names = mat_roi[np.triu_indices(mat_roi.shape[0], k=1)]

                        #### fill containers
                        mat_allplot[cf_metric][band] = np.append(mat_allplot[cf_metric][band], mat_values)
                        mat_pairs_allplot[cf_metric][band] = np.append(mat_pairs_allplot[cf_metric][band], mat_pairs_names)

    #### reduce vector
    mat_allplot_unique = {}

    pairs_unique = np.unique(mat_pairs_allplot[cf_metric][band])

    for cf_metric in ['ispc', 'wpli']:
        mat_allplot_unique[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot_unique[cf_metric][band] = np.array([])

                    #pair_i = pairs_unique[0]
                    for pair_name_i in pairs_unique:
                        
                        mask_pair_i = np.where(mat_pairs_allplot[cf_metric][band] == pair_name_i)[0]
                        mat_allplot_unique[cf_metric][band] = np.append( mat_allplot_unique[cf_metric][band], np.mean(mat_allplot[cf_metric][band][mask_pair_i]) )

    #### identify missing pairs
    ROI_list = np.array([])

    for pair_i in pairs_unique:

        pair_A, pair_B = pair_i.split('-')[0], pair_i.split('-')[1]

        if pair_A not in ROI_list:
            ROI_list = np.append(ROI_list, pair_A)
        if pair_B not in ROI_list:
            ROI_list = np.append(ROI_list, pair_B)

    mat_verif = np.zeros(( ROI_list.shape[0], ROI_list.shape[0]))
    for ROI_row_i, ROI_row_name in enumerate(ROI_list):
        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
        for ROI_col_i, ROI_col_name in enumerate(ROI_list):

            if ROI_row_name == ROI_col_name:
                mat_verif[ROI_row_i, ROI_col_i] = 1
                continue

            pair_name = f'{ROI_row_name}-{ROI_col_name}'
            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

            if pair_name in pairs_unique:
                mat_verif[ROI_row_i, ROI_col_i] = 1
            if pair_name_rev in pairs_unique:
                mat_verif[ROI_row_i, ROI_col_i] = 1

    if debug:
        plt.matshow(mat_verif)
        plt.show()

    #### export missig matrix
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))
    plt.matshow(mat_verif)
    plt.savefig('missing_mat.png')
    plt.close('all')

    #### generate reduced mat 
    mat_allplot_reduced = {}

    for cf_metric in ['ispc', 'wpli']:
        mat_allplot_reduced[cf_metric] = {}
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [10,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:

                    mat_allplot_reduced[cf_metric][band] = np.zeros(( ROI_list.shape[0], ROI_list.shape[0] ))

                    #ROI_row_i, ROI_row_name = 0, ROI_list[0]
                    for ROI_row_i, ROI_row_name in enumerate(ROI_list):
                        #ROI_col_i, ROI_col_name = 1, ROI_list[1]
                        for ROI_col_i, ROI_col_name in enumerate(ROI_list):

                            if ROI_row_name == ROI_col_name:
                                continue

                            pair_name = f'{ROI_row_name}-{ROI_col_name}'
                            pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                            if np.where(pairs_unique == pair_name)[0].shape[0] == 1:
                                dfc_val = mat_allplot_unique[cf_metric][band][np.where(pairs_unique == pair_name)[0]]
                            else:
                                dfc_val = mat_allplot_unique[cf_metric][band][np.where(pairs_unique == pair_name_rev)[0]]
                            
                            if dfc_val.shape[0] == 0:
                                continue
                            else:
                                mat_allplot_reduced[cf_metric][band][ROI_row_i, ROI_col_i] = dfc_val

                            if debug:
                                plt.matshow(mat_allplot_reduced['ispc']['beta'])
                                plt.show()


    return mat_allplot_reduced, ROI_list







def precompute_dfc_mat_allplot_phase(cond):

    #### define diff and phase to plot
    if cond == 'AC':
        phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
        phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

    if cond == 'SNIFF':
        phase_list = ['pre', 'resp_evnmt', 'post']
        phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

    if cond == 'AL':
        phase_list = ['pre', 'post']
        phase_list_diff = ['pre-post']

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i)
        if cond in prms_i['conditions']:
            if cond == 'FR_CV':
                continue
            sujet_list_selected.append(sujet_i)

    #### initiate containers
    mat_allplot = {}
    mat_pairs_allplot = {}

    #phase = 'pre'
    for phase in phase_list:
        mat_allplot[phase] = {}
        mat_pairs_allplot[phase] = {}
        #cf_metric_i, cf_metric = 0, 'ispc'
        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
            mat_allplot[phase][cf_metric] = {}
            mat_pairs_allplot[phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot[phase][cf_metric][band] = np.array([])
                        mat_pairs_allplot[phase][cf_metric][band] = np.array([])

                        #sujet = sujet_list_selected[0]
                        for sujet in sujet_list_selected:

                            #### extract data
                            prms = get_params(sujet)
                            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
                            xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                            ROI = xr_dfc['x'].data

                            #### generate mat ROI names
                            for row_i, ROI_row_i in enumerate(ROI):
                                
                                mat_roi_row_i = np.array([])
                                
                                for ROI_col_i in ROI:
                                    
                                    pair_i = f'{ROI_row_i}-{ROI_col_i}'
                                    mat_roi_row_i = np.append(mat_roi_row_i, pair_i)

                                if row_i == 0:
                                    mat_roi = mat_roi_row_i
                                else:
                                    mat_roi = np.vstack([mat_roi, mat_roi_row_i])

                            #### generate mat for correct respi phase
                            xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')

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

                            if cond == 'AL':

                                AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                                select_time_vec_pre = np.arange(0, AL_separation_i)
                                select_time_vec_post = np.arange(AL_separation_i, resampled_points_AL)

                            if phase == 'pre':
                                mat = from_dfc_to_mat_conn_mean(xr_dfc[cf_metric_i, :, select_time_vec_pre].data, xr_dfc['pairs'], ROI)
                            elif phase == 'resp_evnmt_1':
                                mat = from_dfc_to_mat_conn_mean(xr_dfc[cf_metric_i, :, select_time_vec_resp_evnmt_1].data, xr_dfc['pairs'], ROI)
                            elif phase == 'resp_evnmt_2':
                                mat = from_dfc_to_mat_conn_mean(xr_dfc[cf_metric_i, :, select_time_vec_resp_evnmt_2].data, xr_dfc['pairs'], ROI)
                            elif phase == 'resp_evnmt':
                                mat = from_dfc_to_mat_conn_mean(xr_dfc[cf_metric_i, :, select_time_vec_resp_evnmt].data, xr_dfc['pairs'], ROI)
                            elif phase == 'post':
                                mat = from_dfc_to_mat_conn_mean(xr_dfc[cf_metric_i, :, select_time_vec_post].data, xr_dfc['pairs'], ROI)

                            #### extract data from mat
                            mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                            mat_pairs_names = mat_roi[np.triu_indices(mat_roi.shape[0], k=1)]

                            #### fill containers
                            mat_allplot[phase][cf_metric][band] = np.append(mat_allplot[phase][cf_metric][band], mat_values)
                            mat_pairs_allplot[phase][cf_metric][band] = np.append(mat_pairs_allplot[phase][cf_metric][band], mat_pairs_names)

    #### reduce vector
    mat_allplot_unique = {}

    pairs_unique = np.unique(mat_pairs_allplot[phase][cf_metric][band])

    for phase in phase_list:

        mat_allplot_unique[phase] = {}

        for cf_metric in ['ispc', 'wpli']:
            mat_allplot_unique[phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot_unique[phase][cf_metric][band] = np.array([])

                        #pair_i = pairs_unique[0]
                        for pair_name_i in pairs_unique:
                            
                            mask_pair_i = np.where(mat_pairs_allplot[phase][cf_metric][band] == pair_name_i)[0]
                            mat_allplot_unique[phase][cf_metric][band] = np.append( mat_allplot_unique[phase][cf_metric][band], np.mean(mat_allplot[phase][cf_metric][band][mask_pair_i]) )

    #### generate ROI list
    ROI_list = np.array([])

    for pair_i in pairs_unique:

        pair_A, pair_B = pair_i.split('-')[0], pair_i.split('-')[1]

        if pair_A not in ROI_list:
            ROI_list = np.append(ROI_list, pair_A)
        if pair_B not in ROI_list:
            ROI_list = np.append(ROI_list, pair_B)

    #### generate reduced mat 
    mat_allplot_reduced = {}

    for phase in phase_list:

        mat_allplot_reduced[phase] = {}

        for cf_metric in ['ispc', 'wpli']:
            mat_allplot_reduced[phase][cf_metric] = {}
            #band_prep = 'lf'
            for band_prep in band_prep_list:
                #band, freq = 'beta', [10,40]
                for band, freq in freq_band_dict_FC_function[band_prep].items():

                    if band in ['beta', 'l_gamma', 'h_gamma']:

                        mat_allplot_reduced[phase][cf_metric][band] = np.zeros(( ROI_list.shape[0], ROI_list.shape[0] ))

                        #ROI_row_i, ROI_row_name = 0, ROI_list[0]
                        for ROI_row_i, ROI_row_name in enumerate(ROI_list):
                            #ROI_col_i, ROI_col_name = 1, ROI_list[1]
                            for ROI_col_i, ROI_col_name in enumerate(ROI_list):

                                if ROI_row_name == ROI_col_name:
                                    continue

                                pair_name = f'{ROI_row_name}-{ROI_col_name}'
                                pair_name_rev = f'{ROI_col_name}-{ROI_row_name}'

                                if np.where(pairs_unique == pair_name)[0].shape[0] == 1:
                                    dfc_val = mat_allplot_unique[phase][cf_metric][band][np.where(pairs_unique == pair_name)[0]]
                                else:
                                    dfc_val = mat_allplot_unique[phase][cf_metric][band][np.where(pairs_unique == pair_name_rev)[0]]
                                
                                if dfc_val.shape[0] == 0:
                                    continue
                                else:
                                    mat_allplot_reduced[phase][cf_metric][band][ROI_row_i, ROI_col_i] = dfc_val

                                if debug:
                                    plt.matshow(mat_allplot_reduced[phase])
                                    plt.show()


    #### compute diff
    mat_allplot_reduced_diff = {}

    #phase_diff = 'pre-post'
    for phase_diff in phase_list_diff:

        mat_allplot_reduced_diff[phase_diff] = {}

        for mat_type_i, mat_type in enumerate(['ispc', 'wpli']):

            mat_allplot_reduced_diff[phase_diff][mat_type] = {}

            phase_diff_A, phase_diff_B = phase_diff.split('-')[0], phase_diff.split('-')[1]
                
            for band in band_name_fc_dfc:

                mat_allplot_reduced_diff[phase_diff][mat_type][band] = mat_allplot_reduced[phase_diff_A][mat_type][band] - mat_allplot_reduced[phase_diff_B][mat_type][band]

    return mat_allplot_reduced, mat_allplot_reduced_diff, ROI_list












################################
######## SAVE FIG ########
################################

#mat = mat_allplot_reduced
def save_fig_dfc(cond, mat, ROI_list):

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_fc_dfc)
    plot_list = ['no_thresh', 'thresh']

    print(f'######## PLOT {cond} ########')
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))

    #### plot
    roi_names = ROI_list
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_name_fc_dfc:

            # mat_scaled = mat[mat_type][band][mat[band][mat_type_i,:,:] != 0]
            mat_scaled = mat[mat_type][band]

            scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_scaled.min())
            scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_scaled.max())

        scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].min(), scales[mat_type]['vmax'].max()

    #### identify scales abs
    scales_abs = {}
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_name_fc_dfc:

            max_list = np.array(())
            max_list = np.append(max_list, mat[mat_type][band].max())
            max_list = np.append(max_list, np.abs(mat[mat_type][band].min()))

            scales_abs[mat_type][band] = max_list.max()

    #### thresh on previous plot
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(mat)

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for band in band_name_fc_dfc:

            thresh_up = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_up)
            thresh_down = np.percentile(mat[mat_type][band].reshape(-1), percentile_thresh_down)

            for x in range(mat_dfc_clean[mat_type][band].shape[1]):
                for y in range(mat_dfc_clean[mat_type][band].shape[1]):
                    if mat_type_i == 0:
                        if mat_dfc_clean[mat_type][band][x,y] < thresh_up:
                            mat_dfc_clean[mat_type][band][x,y] = 0
                    else:
                        if (mat_dfc_clean[mat_type][band][x,y] < thresh_up) & (mat_dfc_clean[mat_type][band][x,y] > thresh_down):
                            mat_dfc_clean[mat_type][band][x,y] = 0

    #mat_type = 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #### mat plot
        fig, axs = plt.subplots(nrows=len(plot_list), ncols=n_band, figsize=(15,15))
        plt.suptitle(mat_type)
        for r, plot_type in enumerate(plot_list):
            for c, band in enumerate(band_name_fc_dfc):
                ax = axs[r, c]
                ax.set_title(f'{band} {plot_type}')
                if r == 0:
                    ax.matshow(mat[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    # ax.matshow(mat[mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                    # ax.matshow(mat[mat_type][band])
                if r == 1:
                    ax.matshow(mat_dfc_clean[mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                if c == 0:
                    ax.set_yticks(np.arange(roi_names.shape[0]))
                    ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'{cond}_MAT_{mat_type}.png')
        plt.close('all')

        #### circle plot
        nrows, ncols = len(plot_list), n_band
        fig = plt.figure()
        _position = 0

        for r, plot_type in enumerate(plot_list):

            for c, band in enumerate(band_name_fc_dfc):

                _position += 1

                if r == 0:
                    # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                    #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
                                                    title=f'{band} {plot_type}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')
                    # mne.viz.plot_connectivity_circle(mat[mat_type][band], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                    #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                if r == 1:
                    mne.viz.plot_connectivity_circle(mat_dfc_clean[mat_type][band], node_names=roi_names, n_lines=None, 
                                                    title=f'{band} {plot_type}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                                    vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                    textcolor='k')

        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{cond}_CIRCLE_{mat_type}.png')
        plt.close('all')






#mat = mat_allplot_reduced
def save_fig_dfc_respi_phase_diff(cond, mat, mat_diff, ROI_list):

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1)]
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_fc_dfc)
    plot_list = ['thresh', 'no_thresh']

    #### define diff and phase to plot
    if cond == 'AC':
        phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
        phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

    if cond == 'SNIFF':
        phase_list = ['pre', 'resp_evnmt', 'post']
        phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

    if cond == 'AL':
        phase_list = ['pre', 'post']
        phase_list_diff = ['pre-post']

    n_cols_raw = len(phase_list)
    n_cols_diff = len(phase_list_diff)

    print(f'######## PLOT {cond} DIFF ########')

    #### identify scales
    scales = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

        for band in band_name_fc_dfc:
            
            for phase in phase_list:

                # mat_scaled = mat[band][mat_type_i,:,:][allband_data[respi_phase_i][band][mat_type_i,:,:] != 0]
                mat_scaled = mat[phase][mat_type][band]

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

                max_list = np.append(max_list, mat[phase][mat_type][band].max())
                max_list = np.append(max_list, np.abs(mat[phase][mat_type][band].min()))

            scales_abs[mat_type][band] = max_list.max()

    #### thresh alldata
    percentile_thresh_up = 99
    percentile_thresh_down = 1

    mat_dfc_clean = copy.deepcopy(mat)

    for phase in phase_list:

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(mat[phase][mat_type][band].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(mat[phase][mat_type][band].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[phase][mat_type][band].shape[1]):
                    for y in range(mat_dfc_clean[phase][mat_type][band].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[phase][mat_type][band][x,y] < thresh_up:
                                mat_dfc_clean[phase][mat_type][band][x,y] = 0
                        else:
                            if (mat_dfc_clean[phase][mat_type][band][x,y] < thresh_up) & (mat_dfc_clean[phase][mat_type][band][x,y] > thresh_down):
                                mat_dfc_clean[phase][mat_type][band][x,y] = 0

    #### thresh alldata diff
    mat_dfc_clean_diff = copy.deepcopy(mat_diff)

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        for phase_diff in phase_list_diff:

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(mat_diff[phase_diff][mat_type][band].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(mat_diff[phase_diff][mat_type][band].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean_diff[phase_diff][mat_type][band].shape[1]):
                    for y in range(mat_dfc_clean_diff[phase_diff][mat_type][band].shape[1]):
                        if (mat_dfc_clean_diff[phase_diff][mat_type][band][x,y] < thresh_up) & (mat_dfc_clean_diff[phase_diff][mat_type][band][x,y] > thresh_down):
                            mat_dfc_clean_diff[phase_diff][mat_type][band][x,y] = 0

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'allcond'))
    roi_names = ROI_list
    mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

    #### plot    
    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        ######## NO THRESH ########

        #### mat plot raw 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_raw, figsize=(15,15))
        plt.suptitle(f'{cond} {mat_type}')
        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list):

                ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')
                
                ax.matshow(mat[phase][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat[phase][mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat[phase][mat_type][band])

                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'{cond}_MAT_RAW_{mat_type}.png')
        plt.close('all')

        #### mat plot DIFF 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_diff, figsize=(15,15))
        plt.suptitle(f'{cond} {mat_type}')
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

                ax.matshow(mat_diff[phase_diff][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'{cond}_MAT_DIFF_{mat_type}.png')
        plt.close('all')


        #### circle plot RAW
        nrows, ncols = n_band, n_cols_raw
        fig = plt.figure()
        _position = 0

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list):

                _position += 1

                # mne.viz.plot_connectivity_circle(mat[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(mat[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{cond}_CIRCLE_RAW_{mat_type}.png')
        plt.close('all')


        #### circle plot DIFF
        nrows, ncols = n_band, n_cols_diff
        fig = plt.figure()
        _position = 0

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list_diff):

                _position += 1

                mne.viz.plot_connectivity_circle(mat_diff[phase_diff][mat_type][band], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        plt.suptitle(f'{cond}_{mat_type}', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{cond}_CIRCLE_DIFF_{mat_type}.png')
        plt.close('all')



        ######### THRESH ########

        #### mat plot raw 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_raw, figsize=(15,15))
        plt.suptitle(f'{cond} {mat_type} THRESH')
        for r, band in enumerate(band_name_fc_dfc):
            for c, phase in enumerate(phase_list):
                
                ax = axs[r, c]

                if c == 0:
                    ax.set_ylabel(band)
                if r == 0:
                    ax.set_title(f'{phase}')
                
                ax.matshow(mat_dfc_clean[phase][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                # ax.matshow(mat_dfc_clean[phase][mat_type][band], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                # ax.matshow(mat_dfc_clean[phase][mat_type][band])

                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'{cond}_MAT_RAW_THRESH_{mat_type}.png')
        plt.close('all')

        #### mat plot DIFF 
        fig, axs = plt.subplots(nrows=n_band, ncols=n_cols_diff, figsize=(15,15))
        plt.suptitle(f'{cond} {mat_type} THRESH')
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

                ax.matshow(mat_dfc_clean_diff[phase_diff][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                
                ax.set_yticks(np.arange(roi_names.shape[0]))
                ax.set_yticklabels(roi_names)
        # plt.show()
        fig.savefig(f'{cond}_MAT_DIFF_THRESH_{mat_type}.png')
        plt.close('all')


        #### circle plot RAW
        nrows, ncols = n_band, n_cols_raw
        fig = plt.figure()
        _position = 0

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list):

                _position += 1

                # mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                # mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                #                                 textcolor='k')
                mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][mat_type][band], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        plt.suptitle(f'{cond}_{mat_type} THRESH', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{cond}_CIRCLE_RAW_THRESH_{mat_type}.png')
        plt.close('all')


        #### circle plot DIFF
        nrows, ncols = n_band, n_cols_diff 
        fig = plt.figure()
        _position = 0

        for r, band in enumerate(band_name_fc_dfc):

            for c, phase in enumerate(phase_list_diff):

                _position += 1

                mne.viz.plot_connectivity_circle(mat_dfc_clean_diff[phase_diff][mat_type][band], node_names=roi_names, n_lines=None, 
                                            title=f'{band} {phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                            vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                            textcolor='k')

        plt.suptitle(f'{cond}_{mat_type} THRESH', color='k')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{cond}_CIRCLE_DIFF_THRESH_{mat_type}.png')
        plt.close('all')







            
################################
######## SUMMARY ########
################################



def process_dfc_res_summary(cond_to_compute):

    print(f'######## SUMMARY DFC ########')

    #### CONNECTIVITY PLOT ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1)]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data

    #### load allcond data 
    allcond_data = {}
    allcond_scales_abs = {}
    allcond_ROI_list = {}

    for cond in cond_to_compute:

        #### load data
        allcond_data_i, allcond_data_diff_i, ROI_list = precompute_dfc_mat_allplot_phase(cond)
        allcond_ROI_list[cond] = ROI_list

        #### define diff and phase to plot
        if cond == 'AC':
            phase_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']
            phase_list_diff = ['pre-resp_evnmt_1', 'pre-post', 'resp_evnmt_1-resp_evnmt_2', 'resp_evnmt_2-post']

        if cond == 'SNIFF':
            phase_list = ['pre', 'resp_evnmt', 'post']
            phase_list_diff = ['pre-resp_evnmt', 'pre-post', 'resp_evnmt-post']

        if cond == 'AL':
            phase_list = ['pre', 'post']
            phase_list_diff = ['pre-post']

        #### scale abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_fc_dfc:

                max_list = np.array(())

                for phase in phase_list:

                    max_list = np.append(max_list, allcond_data_i[phase][mat_type][band].max())
                    max_list = np.append(max_list, np.abs(allcond_data_i[phase][mat_type][band].min()))

                scales_abs[mat_type][band] = max_list.max()

        allcond_scales_abs[cond] = scales_abs

        #### thresh
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean_i = copy.deepcopy(allcond_data_diff_i)

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for phase_diff in phase_list_diff:

                for band in band_name_fc_dfc:

                    thresh_up = np.percentile(allcond_data_diff_i[phase_diff][mat_type][band].reshape(-1), percentile_thresh_up)
                    thresh_down = np.percentile(allcond_data_diff_i[phase_diff][mat_type][band].reshape(-1), percentile_thresh_down)

                    for x in range(mat_dfc_clean_i[phase_diff][mat_type][band].shape[1]):
                        for y in range(mat_dfc_clean_i[phase_diff][mat_type][band].shape[1]):
                            if (mat_dfc_clean_i[phase_diff][mat_type][band][x,y] < thresh_up) & (mat_dfc_clean_i[phase_diff][mat_type][band][x,y] > thresh_down):
                                mat_dfc_clean_i[phase_diff][mat_type][band][x,y] = 0

        #### fill res containers
        allcond_data[cond] = mat_dfc_clean_i

    #### adjust scale
    scales_abs = {}

    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        scales_abs[mat_type] = {}

        for band in band_name_fc_dfc:

            max_list = np.array(())

            for cond in cond_to_compute:

                max_list = np.append(max_list, allcond_scales_abs[cond][mat_type][band])

            scales_abs[mat_type][band] = max_list.max()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'DFC', 'summary'))

    n_rows = len(band_name_fc_dfc)
    n_cols = len(cond_to_compute)

    #mat_type_i, mat_type = 0, 'ispc'
    for mat_type_i, mat_type in enumerate(cf_metrics_list):

        #phase_diff = phase_list_diff[0]
        for phase_diff in phase_list_diff:

            #### mat
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,15))
            plt.suptitle(f'{mat_type} summary THRESH : {phase_diff}')
            for r, band in enumerate(band_name_fc_dfc):
                for c, cond in enumerate(cond_to_compute):

                    if cond == 'AL' and phase_diff.find('resp') != -1:
                        continue

                    if n_cols == 1:
                        ax = axs[r]    
                    else:
                        ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{cond}')

                    ax.matshow(allcond_data[cond][phase_diff][mat_type][band], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    
                    if c == 0:
                        ax.set_yticks(np.arange(allcond_ROI_list[cond].shape[0]))
                        ax.set_yticklabels(allcond_ROI_list[cond])
            # plt.show()
            fig.savefig(f'{sujet}_summary_MAT_{phase_diff}_TRESH_{mat_type}.png')
            plt.close('all')


            #### circle plot
            fig = plt.figure()
            _position = 0

            for r, band in enumerate(band_name_fc_dfc):

                for c, cond in enumerate(cond_to_compute):

                    if cond == 'AL' and phase_diff.find('resp') != -1:
                        _position += 1
                        continue

                    _position += 1

                    mne.viz.plot_connectivity_circle(allcond_data[cond][phase_diff][mat_type][band], node_names=allcond_ROI_list[cond], n_lines=None, 
                                                title=f'{cond} {band}', show=False, padding=7, fig=fig, subplot=(n_rows, n_cols, _position),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')

            plt.suptitle(f'{cond}_{mat_type}_THRESH : {phase_diff}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'{sujet}_summary_CIRCLE_{phase_diff}_TRESH_{mat_type}.png')
            plt.close('all')









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    cond_to_compute = ['AC', 'SNIFF', 'AL']

    #cond = 'AC'
    for cond in cond_to_compute:
            
        mat, ROI_list = precompute_dfc_mat_allplot(cond)
        save_fig_dfc(cond, mat, ROI_list)

        mat, mat_diff, ROI_list = precompute_dfc_mat_allplot_phase(cond)            
        save_fig_dfc_respi_phase_diff(cond, mat, mat_diff, ROI_list)

    process_dfc_res_summary(cond_to_compute)








