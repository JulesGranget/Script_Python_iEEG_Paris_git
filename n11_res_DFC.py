

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
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






def get_data_for_phase(sujet, cond):

    #### get ROI list
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find(cond) != -1)]
    roi_in_data = xr.open_dataarray(file_to_load[0])['x'].data

    #### load data 
    allband_data = {}

    #### prepare export
    if cond == 'AC':
        export_type_list = ['pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

    if cond == 'SNIFF': 
        export_type_list = ['pre', 'resp_evnmt', 'post']

    if cond == 'AL':
        export_type_list = ['pre', 'post']

    #export_type = 'pre'
    for export_type in export_type_list:

        allband_data[export_type] = {}

        #band = 'beta'
        for band in band_name_fc_dfc:

            file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1)]
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

            if cond == 'AL':

                AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                select_time_vec_pre = np.arange(0, AL_separation_i)
                select_time_vec_post = np.arange(AL_separation_i, resampled_points_AL)

            if export_type == 'pre':
                data_chunk = data[:, :, select_time_vec_pre]
            elif export_type == 'resp_evnmt':
                data_chunk = data[:, :, select_time_vec_resp_evnmt]
            elif export_type == 'resp_evnmt_1':
                data_chunk = data[:, :, select_time_vec_resp_evnmt_1]
            elif export_type == 'resp_evnmt_2':
                data_chunk = data[:, :, select_time_vec_resp_evnmt_2]
            elif export_type == 'post':
                data_chunk = data[:, :, select_time_vec_post]

            mat_cf = np.zeros(( data['mat_type'].shape[0], roi_in_data.shape[0], roi_in_data.shape[0] ))

            for cf_metric_i, cf_metric in enumerate(data['mat_type'].data):
                mat_cf[cf_metric_i,:,:] = from_dfc_to_mat_conn_mean(data_chunk.loc[cf_metric,:,:].data, data['pairs'].data, roi_in_data)

            allband_data[export_type][band] = mat_cf

    return allband_data













################################
######## SAVE FIG ########
################################


def process_dfc_res(sujet, cond, export_type):

    print(f'######## {cond} DFC {export_type} ########')

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1)]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    n_band = len(band_name_fc_dfc)
    plot_list = ['no_thresh', 'thresh']

    ######## WHOLE ########

    if export_type == 'whole':
            
        #### load data 
        os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
        allband_data = {}

        for band in band_name_fc_dfc:

            file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band) != -1 and i.find(cond) != -1)]
            allband_data[band] = xr.open_dataarray(file_to_load[0]).data

        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'DFC', 'allcond'))

        #### identify scales
        scales = {}
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales[mat_type] = {'vmin' : np.array([]), 'vmax' : np.array([])}

            for band in band_name_fc_dfc:

                mat_zero_excluded = allband_data[band][mat_type_i,:,:][allband_data[band][mat_type_i,:,:] != 0]

                scales[mat_type]['vmin'] = np.append(scales[mat_type]['vmin'], mat_zero_excluded.min())
                scales[mat_type]['vmax'] = np.append(scales[mat_type]['vmax'], mat_zero_excluded.max())

            scales[mat_type]['vmin'], scales[mat_type]['vmax'] = scales[mat_type]['vmin'].mean(), scales[mat_type]['vmax'].mean()
            
        #### identify scales abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_fc_dfc:

                max_list = np.array(())

                max_list = np.append(max_list, allband_data[band][mat_type_i,:,:].max())
                max_list = np.append(max_list, np.abs(allband_data[band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

        #### thresh on previous plot
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean = copy.deepcopy(allband_data)

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            for band in band_name_fc_dfc:

                thresh_up = np.percentile(allband_data[band][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                thresh_down = np.percentile(allband_data[band][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                for x in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                    for y in range(mat_dfc_clean[band][mat_type_i,:,:].shape[1]):
                        if mat_type_i == 0:
                            if mat_dfc_clean[band][mat_type_i,x,y] < thresh_up:
                                mat_dfc_clean[band][mat_type_i,x,y] = 0
                        else:
                            if (mat_dfc_clean[band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][mat_type_i,x,y] > thresh_down):
                                mat_dfc_clean[band][mat_type_i,x,y] = 0

        #### plot
        mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

        #mat_type_i, mat_type = 0, 'ispc'
        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #### mat plot
            fig, axs = plt.subplots(nrows=len(plot_list), ncols=n_band, figsize=(15,15))
            plt.suptitle(f'{cond} {mat_type}')
            for r, plot_type in enumerate(plot_list):
                for c, band in enumerate(band_name_fc_dfc):
                    ax = axs[r, c]
                    ax.set_title(f'{band} {plot_type}')
                    if r == 0:
                        ax.matshow(allband_data[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                        # ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                        # ax.matshow(allband_data[band][mat_type_i,:,:])
                    if r == 1:
                        ax.matshow(mat_dfc_clean[band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'{cond}_MAT_{sujet}_{mat_type}.png')
            plt.close('all')

            #### circle plot
            nrows, ncols = len(plot_list), n_band
            fig = plt.figure()
            _position = 0

            for r, plot_type in enumerate(plot_list):

                for c, band in enumerate(band_name_fc_dfc):

                    _position += 1

                    if r == 0:
                        # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                        #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                        #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                        #                                 textcolor='k')
                        mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                        title=f'{band} {plot_type}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                                        vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                        textcolor='k')
                        # mne.viz.plot_connectivity_circle(allband_data[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                        #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, c+1),
                        #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                        #                                 textcolor='k')
                    if r == 1:
                        mne.viz.plot_connectivity_circle(mat_dfc_clean[band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                        title=f'{band} {plot_type}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                                        vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                        textcolor='k')

            plt.suptitle(f'{cond}_{mat_type}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'{cond}_CIRCLE_{sujet}_{mat_type}.png')
            plt.close('all')

    ######## PHASE ########

    elif export_type == 'phase':

        #### load data 
        allband_data = get_data_for_phase(sujet, cond)

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

        #### substract data
        allband_data_diff = {}

        #phase_diff = 'pre-post'
        for phase_diff in phase_list_diff:

            allband_data_diff[phase_diff] = {}

            phase_diff_A, phase_diff_B = phase_diff.split('-')[0], phase_diff.split('-')[1]
                
            for band in band_name_fc_dfc:

                allband_data_diff[phase_diff][band] = allband_data[phase_diff_A][band] - allband_data[phase_diff_B][band]

        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'DFC', 'allcond'))

        #### plot
        mat_type_color = {'ispc' : cm.OrRd, 'wpli' : cm.seismic}

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
                            if mat_type_i == 0:
                                if mat_dfc_clean[phase][band][mat_type_i,x,y] < thresh_up:
                                    mat_dfc_clean[phase][band][mat_type_i,x,y] = 0
                            else:
                                if (mat_dfc_clean[phase][band][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[phase][band][mat_type_i,x,y] > thresh_down):
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
        
        #mat_type_i, mat_type = 1, 'wpli'
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
                    
                    ax.matshow(allband_data[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    # ax.matshow(allband_data[phase][band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                    # ax.matshow(allband_data[phase][band][mat_type_i,:,:])

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

                    ax.matshow(allband_data_diff[phase_diff][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    
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

                    # mne.viz.plot_connectivity_circle(allband_data[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    # mne.viz.plot_connectivity_circle(allband_data[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    mne.viz.plot_connectivity_circle(allband_data[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
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

                    mne.viz.plot_connectivity_circle(allband_data_diff[phase_diff][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                                                title=f'{band} {phase}', show=False, padding=7, fig=fig, subplot=(nrows, ncols, _position),
                                                vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], colormap=cm.seismic, facecolor='w', 
                                                textcolor='k')

            plt.suptitle(f'{cond}_{mat_type}', color='k')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            # fig.show()
            fig.savefig(f'{cond}_CIRCLE_DIFF_{mat_type}.png')
            plt.close('all')



            ######### THRESH #########

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
                    
                    ax.matshow(mat_dfc_clean[phase][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    # ax.matshow(mat_dfc_clean[phase][band][mat_type_i,:,:], vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'])
                    # ax.matshow(mat_dfc_clean[phase][band][mat_type_i,:,:])

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

                    ax.matshow(mat_dfc_clean_diff[phase_diff][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    
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

                    # mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 vmin=scales[mat_type]['vmin'], vmax=scales[mat_type]['vmax'], colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    # mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
                    #                                 title=band, show=False, padding=7, fig=fig, subplot=(nrows, ncols, r+c+1),
                    #                                 colormap=mat_type_color[mat_type], facecolor='w', 
                    #                                 textcolor='k')
                    mne.viz.plot_connectivity_circle(mat_dfc_clean[phase][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
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

                    mne.viz.plot_connectivity_circle(mat_dfc_clean_diff[phase_diff][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
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

def process_dfc_res_summary(sujet):

    print(f'######## SUMMARY DFC ########')

    #### CONNECTIVITY PLOT ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    file_to_load = [i for i in os.listdir() if ( i.find('reducedpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1)]
    roi_names = xr.open_dataarray(file_to_load[0])['x'].data
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].data
    prms = get_params(sujet)

    #### load allcond data 
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
            phase_list = ['pre', 'post']
            phase_list_diff = ['pre-post']

        #### load data
        allcond_data_i = get_data_for_phase(sujet, cond)

        #### scale abs
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            for band in band_name_fc_dfc:

                max_list = np.array(())

                for phase in phase_list:

                    max_list = np.append(max_list, allcond_data_i[phase][band][mat_type_i,:,:].max())
                    max_list = np.append(max_list, np.abs(allcond_data_i[phase][band][mat_type_i,:,:].min()))

                scales_abs[mat_type][band] = max_list.max()

        allcond_scales_abs[cond] = scales_abs

        #### conpute diff
        allcond_data_diff_i = {}

        #phase_diff = 'pre-post'
        for phase_diff in phase_list_diff:

            allcond_data_diff_i[phase_diff] = {}

            phase_diff_A, phase_diff_B = phase_diff.split('-')[0], phase_diff.split('-')[1]
                
            for band in band_name_fc_dfc:

                allcond_data_diff_i[phase_diff][band] = allcond_data_i[phase_diff_A][band] - allcond_data_i[phase_diff_B][band]

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
            plt.suptitle(f'{mat_type} summary THRESH : {phase_diff}')
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

                    if n_cols == 1:
                        ax = axs[r]    
                    else:
                        ax = axs[r, c]

                    if c == 0:
                        ax.set_ylabel(band)
                    if r == 0:
                        ax.set_title(f'{cond}')

                    ax.matshow(allcond_data[cond][phase_diff_sel][band][mat_type_i,:,:], vmin=-scales_abs[mat_type][band], vmax=scales_abs[mat_type][band], cmap=cm.seismic)
                    
                    if c == 0:
                        ax.set_yticks(np.arange(roi_names.shape[0]))
                        ax.set_yticklabels(roi_names)
            # plt.show()
            fig.savefig(f'{sujet}_summary_MAT_{phase_diff}_TRESH_{mat_type}.png')
            plt.close('all')

            #### circle plot
            fig = plt.figure()
            _position = 0

            for r, band in enumerate(band_name_fc_dfc):

                for c, cond in enumerate(conditions):

                    if cond == 'AL' and phase_diff.find('resp') != -1:
                        _position += 1
                        continue

                    elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') == -1:
                        phase_diff_list = []
                        for phase_diff_i in phase_diff.split('-'):
                            if phase_diff_i.find('resp_evnmt') != -1:
                                phase_diff_list.append('resp_evnmt')
                            else:
                                phase_diff_list.append(phase_diff_i)
                        
                        phase_diff_sel = f'{phase_diff_list[0]}-{phase_diff_list[1]}'
                        _position += 1

                    elif cond != 'AC' and phase_diff.find('1') != -1 and phase_diff.find('2') != -1:
                        _position += 1
                        continue

                    elif cond != 'AC' and phase_diff.find('1') == -1 and phase_diff.find('2') != -1:
                        _position += 1
                        continue

                    else:
                        phase_diff_sel = phase_diff
                        _position += 1

                    mne.viz.plot_connectivity_circle(allcond_data[cond][phase_diff_sel][band][mat_type_i,:,:], node_names=roi_names, n_lines=None, 
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

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)
        prms = get_params(sujet)

        #### allcond
        #cond = 'AC'
        for cond in prms['conditions']:
            if cond == 'FR_CV':
                continue
            #export_type = 'phase'
            for export_type in ['whole', 'phase']:
                process_dfc_res(sujet, cond, export_type)
        
        #### summary
        process_dfc_res_summary(sujet)





    