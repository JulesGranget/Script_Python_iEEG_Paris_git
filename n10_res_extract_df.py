

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
######## LOAD DATA ########
################################




def get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond




def get_tf_itpc_stretch_allcond(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond




def get_tf_itpc_stretch_allcond_AL(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'allcond_{sujet}_tf_AL_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    if tf_mode == 'ITPC':
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'allcond_{sujet}_itpc_AL_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    os.chdir(source_path)

    return tf_stretch_allcond



def load_surrogates(sujet, respfeatures_allcond, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq_lf' : {}, 'cyclefreq_hf' : {}, 'MVL' : {}}

    for cond in prms['conditions']:

        if len(respfeatures_allcond[cond]) == 1:

            surrogates_allcond['Cxy'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy')]
            surrogates_allcond['cyclefreq_lf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy')]
            surrogates_allcond['cyclefreq_hf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy')]
            surrogates_allcond['MVL'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_MVL_lf.npy')]

        elif len(respfeatures_allcond[cond]) > 1:

            data_load = {'Cxy' : [], 'cyclefreq_lf' : [], 'cyclefreq_hf' : [], 'MVL' : []}

            for session_i in range(len(respfeatures_allcond[cond])):

                data_load['Cxy'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
                data_load['cyclefreq_lf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
                data_load['cyclefreq_hf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
                data_load['MVL'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_lf.npy'))
            
            surrogates_allcond['Cxy'][cond] = data_load['Cxy']
            surrogates_allcond['cyclefreq_lf'][cond] = data_load['cyclefreq_lf']
            surrogates_allcond['cyclefreq_hf'][cond] = data_load['cyclefreq_hf']
            surrogates_allcond['MVL'][cond] = data_load['MVL']


    return surrogates_allcond








################################
######## TF & ITPC ########
################################




def export_TF_in_df(sujet, prms):

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_TF.xlsx')):
        print('TF : ALREADY COMPUTED')
        return

    #### load prms
    prms = get_params(sujet)
    df_loca = get_loca_df(sujet)
    
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

                    if cond == 'AL':
                        data = get_tf_itpc_stretch_allcond_AL(sujet, 'TF')[band_prep][band][chan_i, :, :]
                    else:
                        data = get_tf_itpc_stretch_allcond(sujet, 'TF')[band_prep][cond][band][chan_i, :, :]

                    Pxx = np.mean(data, axis=0)

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

                        AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                        Pxx_pre = np.mean(Pxx[:AL_separation_i])
                        Pxx_post = np.mean(Pxx[AL_separation_i:])

                        phase_list = ['AL_1st_phase', 'AL_2nd_phase']
                        Pxx_list = [Pxx_pre, Pxx_post]

                    data_export_i =   {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'chan' : [chan_name]*len(phase_list), 
                                        'ROI' : [ROI_i]*len(phase_list), 'Lobe' : [Lobe_i]*len(phase_list), 'side' : [side_i]*len(phase_list), 
                                        'band' : [band]*len(phase_list), 'phase' : phase_list, 'Pxx' : Pxx_list}
                    
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_TF.xlsx')







def export_ITPC_in_df(sujet, prms):

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_ITPC.xlsx')):
        print('ITPC : ALREADY COMPUTED')
        return

    #### load prms
    prms = get_params(sujet)
    df_loca = get_loca_df(sujet)
    
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

                if cond == 'AL':
                    continue

                #band, freq = 'theta', [2, 10]
                for band, freq in freq_band_dict[band_prep].items():

                    data = get_tf_itpc_stretch_allcond(sujet, 'ITPC')[band_prep][cond][band][chan_i, :, :]

                    Pxx = np.mean(data, axis=0)

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

                        AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                        Pxx_pre = np.mean(Pxx[:AL_separation_i])
                        Pxx_post = np.mean(Pxx[AL_separation_i:])

                        phase_list = ['AL_1st_phase', 'AL_2nd_phase']
                        Pxx_list = [Pxx_pre, Pxx_post]

                    data_export_i =   {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'chan' : [chan_name]*len(phase_list), 
                                        'ROI' : [ROI_i]*len(phase_list), 'Lobe' : [Lobe_i]*len(phase_list), 'side' : [side_i]*len(phase_list), 
                                        'band' : [band]*len(phase_list), 'phase' : phase_list, 'Pxx' : Pxx_list}
                    
                    df_export_i = pd.DataFrame.from_dict(data_export_i)
                    
                    df_export = pd.concat([df_export, df_export_i])

    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_ITPC.xlsx')








########################################
######## COMPUTE GRAPH METRICS ########
########################################


#mat = dfc_data['inspi']
def from_dfc_to_mat_conn_trpz(mat, pairs, roi_in_data):

    #### mean over pairs
    pairs_unique = np.unique(pairs)

    pairs_unique_mat = np.zeros(( pairs_unique.shape[0], mat.shape[1] ))
    #pair_name_i = pairs_unique[0]
    for pair_name_i, pair_name in enumerate(pairs_unique):
        pairs_to_mean = np.where(pairs == pair_name)[0]
        pairs_unique_mat[pair_name_i, :] = np.mean(mat[pairs_to_mean,:], axis=0)

    #### fill mat
    mat_cf = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue
            val_to_place, pair_count = 0, 0
            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            if np.where(pairs_unique == pair_to_find)[0].shape[0] != 0:
                x = mat[np.where(pairs_unique == pair_to_find)[0]]
                val_to_place += np.trapz(x)
                pair_count += 1
            if np.where(pairs_unique == pair_to_find_rev)[0].shape[0] != 0:
                x = mat[np.where(pairs_unique == pair_to_find_rev)[0]]
                val_to_place += np.trapz(x)
                pair_count += 1
            val_to_place /= pair_count

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf




def compute_graph_metric_dfc(sujet, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_DFC.xlsx')):
        print('DFC : ALREADY COMPUTED')
        return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'CPL', 'GE', 'SWN'])

    #### compute
    #cond = 'AC'
    for cond in prms['conditions']:
        #band_prep = 'hf'
        for band_prep in band_prep_list:
            #band, freq = 'l_gamma', [50,80]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma'] and cond != 'FR_CV':
                    #cf_metric = 'ispc'
                    for cf_metric in ['ispc', 'wpli']:

                        roi_in_data = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')['x'].data
                        xr_graph = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                        pairs = xr_graph['pairs'].data
                        cf_metric_i = np.where(xr_graph['mat_type'].data == cf_metric)[0]

                        if cond == 'SNIFF':

                            dfc_pre = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, sniff_extract_pre[0]:sniff_extract_pre[1]], pairs, roi_in_data)
                            dfc_resp_evnmt = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, sniff_extract_resp_evnmt[0]:sniff_extract_resp_evnmt[1]], pairs, roi_in_data)
                            dfc_post = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, sniff_extract_post[0]:sniff_extract_post[1]], pairs, roi_in_data)

                            phase_list = ['pre', 'resp_evnt', 'post']
                            mat_cf = [dfc_pre, dfc_resp_evnmt, dfc_post]
                        
                        if cond == 'AC':

                            dfc_pre = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, AC_extract_pre[0]:AC_extract_pre[1]], pairs, roi_in_data)
                            dfc_resp_evnmt_1 = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, AC_extract_resp_evnmt_1[0]:AC_extract_resp_evnmt_1[1]], pairs, roi_in_data)
                            dfc_resp_evnmt_2 = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, AC_extract_resp_evnmt_2[0]:AC_extract_resp_evnmt_2[1]], pairs, roi_in_data)
                            dfc_post = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, AC_extract_post[0]:AC_extract_post[1]], pairs, roi_in_data)

                            phase_list = ['pre', 'resp_evnt_1', 'resp_evnt_2', 'post']
                            mat_cf = [dfc_pre, dfc_resp_evnmt_1, dfc_resp_evnmt_2, dfc_post]

                        if cond == 'AL':

                            AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                            dfc_pre = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, :AL_separation_i], pairs, roi_in_data)
                            dfc_post = from_dfc_to_mat_conn_trpz(xr_graph.loc[cf_metric, :, AL_separation_i:], pairs, roi_in_data)

                            phase_list = ['pre', 'post']
                            mat_cf = [dfc_pre, dfc_post]

                        if debug:
                            plt.plot(xr_graph[cf_metric_i, 0, :].data.reshape(-1))
                            plt.show()

                            plt.matshow(mat_cf['inspi'])
                            plt.matshow(mat_cf['expi'])
                            plt.show()

                        #dfc_phase = 'pre'
                        for dfc_phase_i, dfc_phase in enumerate(phase_list):

                            mat = mat_cf[dfc_phase_i]
                            mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                            
                            if debug:
                                np.sum(mat_values > np.percentile(mat_values, 90))

                                count, bin, fig = plt.hist(mat_values)
                                plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                                plt.show()

                            #### apply thresh
                            for chan_i in range(mat.shape[0]):
                                mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, 50))[0]] = 0

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
    df_export.to_excel(f'{sujet}_df_graph_DFC.xlsx')





def compute_graph_metric_fc(sujet, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_graph_FC.xlsx')):
        print('FC : ALREADY COMPUTED')
        return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'CPL', 'GE', 'SWN'])

    #### compute
    #cond = 'RD_CV'
    for cond in prms['conditions']:
        if cond not in ['FR_CV', 'AL']:
            continue
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'theta', [4,8]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                #cf_metric = 'ispc'
                for cf_metric in ['ispc', 'wpli']:

                    roi_in_data = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')['x'].data
                    xr_graph = xr.open_dataarray(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')
                    cf_metric_i = np.where(xr_graph['mat_type'].data == cf_metric)[0]

                    mat = xr_graph.loc[cf_metric, :, :].data
                    mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                    
                    if debug:
                        np.sum(mat_values > np.percentile(mat_values, 90))

                        count, bin, fig = plt.hist(mat_values)
                        plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                        plt.show()

                    #### apply thresh
                    for chan_i in range(mat.shape[0]):
                        mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, 30))[0]] = 0

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

                    data_export_i =    {'sujet' : [sujet], 'cond' : [cond], 'band' : [band], 'metric' : [cf_metric], 
                                    'CPL' : [CPL], 'GE' : [GE], 'SWN' : [SWN]}
                    df_export_i = pd.DataFrame.from_dict(data_export_i)

                    df_export = pd.concat([df_export, df_export_i])


    #### save
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_graph_FC.xlsx')








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







def compute_dfc_values(sujet, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))

    #### verif computation
    if os.path.exists(os.path.join(path_results, sujet, 'df', f'{sujet}_df_DFC.xlsx')):
        print('DFC values : ALREADY COMPUTED')
        return

    #### initiate df
    df_export = pd.DataFrame(columns=['sujet', 'cond', 'band', 'metric', 'phase', 'pair', 'value'])

    #### get pairs of interest
    pairs_of_interest = generate_ROI_pairs()

    #### compute
    #cond = 'SNIFF'
    for cond in prms['conditions']:
        if cond == 'FR_CV':
            continue
        #band_prep = 'hf'
        for band_prep in band_prep_list:
            #band, freq = 'l_gamma', [50,80]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                if band in ['beta', 'l_gamma', 'h_gamma']:
                    #cf_metric_i, cf_metric = 0, 'ispc'
                    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                        #### extract data
                        xr_dfc = xr.open_dataarray(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
                        pairs = xr_dfc['pairs'].data

                        #### identify pairs of interest
                        #pair_i, pair_name = 0, pairs_of_interest[0]
                        for pair_i, pair_name in enumerate(pairs_of_interest):

                            pairs_of_interest_i_list = np.where((pairs == pair_name) | (pairs == f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"))[0]
                            
                            if pairs_of_interest_i_list.shape[0] == 0:
                                continue

                            for pair_i in pairs_of_interest_i_list:

                                pairs_of_interest_name = pairs[pair_i]

                                if cond == 'SNIFF':

                                    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                                    time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
                                    select_time_vec_pre = (time_vec >= sniff_extract_pre[0]) & (time_vec <= sniff_extract_pre[1])
                                    select_time_vec_resp_evnmt = (time_vec >= sniff_extract_resp_evnmt[0]) & (time_vec <= sniff_extract_resp_evnmt[1])
                                    select_time_vec_post = (time_vec >= sniff_extract_post[0]) & (time_vec <= sniff_extract_post[1])

                                    phase_list = ['whole', 'pre', 'resp_evnmt', 'post']

                                    value_list =    [np.mean(xr_dfc[cf_metric_i, pair_i, :].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_pre].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_resp_evnmt].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_post].data)
                                                ]
                                
                                if cond == 'AC':

                                    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                                    time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
                                    select_time_vec_pre = (time_vec >= AC_extract_pre[0]) & (time_vec <= AC_extract_pre[1])
                                    select_time_vec_resp_evnmt_1 = (time_vec >= AC_extract_resp_evnmt_1[0]) & (time_vec <= AC_extract_resp_evnmt_1[1])
                                    select_time_vec_resp_evnmt_2 = (time_vec >= AC_extract_resp_evnmt_2[0]) & (time_vec <= AC_extract_resp_evnmt_2[1])
                                    select_time_vec_post = (time_vec >= AC_extract_post[0]) & (time_vec <= AC_extract_post[1])

                                    phase_list = ['whole', 'pre', 'resp_evnmt_1', 'resp_evnmt_2', 'post']

                                    value_list =    [np.mean(xr_dfc[cf_metric_i, pair_i, :].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_pre].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_resp_evnmt_1].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_resp_evnmt_2].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_post].data)
                                                ]

                                if cond == 'AL':

                                    AL_separation_i = int(AL_coeff_pre * resampled_points_AL)

                                    select_time_vec_pre = np.arange(0, AL_separation_i)
                                    select_time_vec_post = np.arange(AL_separation_i, resampled_points_AL)

                                    phase_list = ['whole', 'pre', 'post']

                                    value_list =    [np.mean(xr_dfc[cf_metric_i, pair_i, :].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_pre].data),
                                                np.mean(xr_dfc[cf_metric_i, pair_i, select_time_vec_post].data)
                                                ]

                                data_export_i =    {'sujet' : [sujet]*len(phase_list), 'cond' : [cond]*len(phase_list), 'band' : [band]*len(phase_list), 
                                        'metric' : [cf_metric]*len(phase_list), 'phase' : phase_list, 'pair' : [pairs[pair_i]]*len(phase_list), 
                                        'value' : value_list}

                                df_export_i = pd.DataFrame.from_dict(data_export_i)

                                df_export = pd.concat([df_export, df_export_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'df'))
    df_export.to_excel(f'{sujet}_df_DFC.xlsx')










########################################
######## COMPILATION FUNCTION ########
########################################




def compilation_export_df(sujet):

    print(sujet)

    #### load params
    prms = get_params(sujet)

    # #### export
    export_TF_in_df(sujet, prms)
    export_ITPC_in_df(sujet, prms)
    compute_graph_metric_dfc(sujet, prms)
    compute_graph_metric_fc(sujet, prms)
    compute_dfc_values(sujet, prms)





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compilation_export_df)(sujet) for sujet in sujet_list_FR_CV)

    #sujet = sujet_list[0]
    for sujet in sujet_list:
                
        #### export df
        compilation_export_df(sujet)
    
    
