

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import cv2
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import networkx as nx

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0quater_stats import *

from mpl_toolkits.axes_grid1 import make_axes_locatable


debug = False




########################################
######## FUNCTION ANALYSIS ########
########################################


def get_ROI_Lobes_list_and_Plots(cond, electrode_recording_type):

    #### generate anat list
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    ROI_list = np.unique(nomenclature_df['Our correspondances'].values)
    lobe_list = np.unique(nomenclature_df['Lobes'].values)

    #### fill dict with anat names
    ROI_dict_count = {}
    ROI_dict_plots = {}
    for i, _ in enumerate(ROI_list):
        ROI_dict_count[ROI_list[i]] = 0
        ROI_dict_plots[ROI_list[i]] = []

    lobe_dict_count = {}
    lobe_dict_plots = {}
    for i, _ in enumerate(lobe_list):
        lobe_dict_count[lobe_list[i]] = 0
        lobe_dict_plots[lobe_list[i]] = []

    #### filter only sujet with correct cond
    sujet_list_selected = []
    for sujet_i in sujet_list:
        prms_i = get_params(sujet_i, electrode_recording_type)
        if cond in prms_i['conditions']:
            sujet_list_selected.append(sujet_i)

    #### search for ROI & lobe that have been counted
    #sujet_i = sujet_list_selected[1]
    for sujet_i in sujet_list_selected:

        os.chdir(os.path.join(path_anatomy, sujet_i))

        if electrode_recording_type == 'monopolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        if electrode_recording_type == 'bipolaire':
            plot_loca_df = pd.read_excel(sujet_i + '_plot_loca_bi.xlsx')

        chan_list_ieeg = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        chan_list_ieeg_csv = chan_list_ieeg

        count_verif = 0

        #nchan = chan_list_ieeg_csv[0]
        for nchan in chan_list_ieeg_csv:

            ROI_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            ROI_dict_count[ROI_tmp] = ROI_dict_count[ROI_tmp] + 1
            lobe_dict_count[lobe_tmp] = lobe_dict_count[lobe_tmp] + 1
            count_verif += 1

            ROI_dict_plots[ROI_tmp].append([sujet_i, nchan])
            lobe_dict_plots[lobe_tmp].append([sujet_i, nchan])

        #### verif count
        if count_verif != len(chan_list_ieeg):
            raise ValueError('ERROR : anatomical count is not correct, count != len chan_list')

    #### exclude ROi and Lobes with 0 counts
    ROI_to_include = [ROI_i for ROI_i in ROI_list if ROI_dict_count[ROI_i] > 0]
    lobe_to_include = [Lobe_i for Lobe_i in lobe_list if lobe_dict_count[Lobe_i] > 0]

    return ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots








########################
######## STATS ########
########################



def get_tf_stats(cond, tf_plot, pixel_based_distrib, nfrex):

    tf_thresh = np.zeros(tf_plot.shape)

    if cond == 'AC':
        stretch_point = stretch_point_TF_ac_resample
    if cond == 'SNIFF':
        stretch_point = stretch_point_TF_sniff_resampled
    if cond == 'AL':
        stretch_point = resampled_points_AL
    
    phase_list = phase_stats[cond]
    phase_point = int(stretch_point/len(phase_list))

    #phase_i, phase_name = 0, phase_list[0]
    for phase_i, phase_name in enumerate(phase_list):

        start = phase_point * phase_i
        stop = phase_point * phase_i + phase_point

        #wavelet_i = 0
        for wavelet_i in range(nfrex):

            mask = np.logical_or(tf_plot[wavelet_i, start:stop] < pixel_based_distrib[phase_i, wavelet_i, 0], tf_plot[wavelet_i, start:stop] > pixel_based_distrib[phase_i, wavelet_i, 1])
            tf_thresh[wavelet_i, start:stop] = mask*1

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster_allplot)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf_plot.shape[-1])

        plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh






def get_stats_TF_pseudo_network(electrode_recording_type, stats_type, contour_all_ROI):
    
    os.chdir(os.path.join(path_results, 'allplot', 'df'))

    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['FR_CV', 'SNIFF', 'AC']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel = [_phase for _phase in df_TF['phase'].unique() if _phase not in ['FR_CV_expi', 'FR_CV_inspi']]

    stats_dfc = np.zeros((len(band_sel), len(ROI_list), len(phase_list_sel)))

    if stats_type == 'ttest':

        for band_i, band in enumerate(band_sel):
        
            for ROI_i, ROI in enumerate(ROI_list):

                baseline = df_TF.query(f"phase == 'FR_CV_whole' and ROI == '{ROI}' and band == '{band}'")

                for phase_i, phase in enumerate(phase_list_sel):

                    if phase == 'FR_CV_whole':
                        continue

                    cond = df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")

                    df = pd.concat([baseline, cond])
                    predictor = 'cond'
                    outcome = 'Pxx'

                    stats_dfc[band_i, ROI_i, phase_i] = np.round(get_df_stats_pre(df, predictor, outcome, subject='sujet', design='between', transform=False, verbose=True)['p-val'].values[0], 3)

        mask_dfc = stats_dfc < 0.05

    elif stats_type == 'perm':

        dict_time_extract = {'SNIFF_pre_01' : sniff_extract_prepre, 'SNIFF_pre_02' : sniff_extract_pre, 'SNIFF_resp_evnmt' : sniff_extract_resp_evnmt, 'SNIFF_post' : sniff_extract_post, 
                             'AC_pre_01' : AC_extract_pre_1, 'AC_pre_02' : AC_extract_pre_2, 'AC_pre_03' : AC_extract_pre_3, 'AC_pre_04' : AC_extract_pre_4, 
                             'AC_resp_evnmt_01' : AC_extract_resp_evnmt_1, 'AC_resp_evnmt_02' : AC_extract_resp_evnmt_2, 'AC_resp_evnmt_03' : AC_extract_resp_evnmt_3, 'AC_resp_evnmt_04' : AC_extract_resp_evnmt_4,
                             'AC_post_01' : AC_extract_post_1, 'AC_post_02' : AC_extract_post_2, 'AC_post_03' : AC_extract_post_3, 'AC_post_04' : AC_extract_post_4}

        for band_i, band in enumerate(band_sel):
        
            for ROI_i, ROI in enumerate(ROI_list):

                for phase_i, phase in enumerate(phase_list_sel):

                    if phase == 'FR_CV_whole':
                        continue

                    cond = phase.split('_')[0]
                    contour = contour_all_ROI[ROI][cond]

                    if cond == 'SNIFF':
                        time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff_resampled)
                
                    if cond == 'AC':
                        time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac_resample)

                    start, stop = dict_time_extract[phase]
                    freq = freq_band_dict_df_extraction['wb'][band]
                    mask_frex_band = (frex >= freq[0]) & (frex <= freq[-1])

                    contour_sel = contour[:,(time_vec >= start) & (time_vec <= stop)]
                    contour_sel = contour_sel[mask_frex_band,:]

                    stats_dfc[band_i, ROI_i, phase_i] = int((contour_sel.sum() / contour_sel.size)*100) 

        # stats_dfc[stats_dfc > np.percentile(stats_dfc[stats_dfc>0], 5)]
        
        mask_dfc = stats_dfc > 5

    return mask_dfc




def get_stats_coutour_from_TF(electrode_recording_type):

    cond_to_plot = [cond for cond in conditions if cond != 'AL']    

    ROI_list, lobe_list, ROI_to_include, lobe_to_include, ROI_dict_plots, lobe_dict_plots = get_ROI_Lobes_list_and_Plots('FR_CV', electrode_recording_type)

    contour_all_ROI = {}

    #ROI_i, ROI = 1, ROI_to_include[1]
    for ROI_i, ROI in enumerate(ROI_to_include):

        print(ROI)

        os.chdir(os.path.join(path_precompute, 'allplot', 'TF'))

        contour_all_ROI[ROI] = {}
    
        #cond = cond_to_plot[0]
        for cond in cond_to_plot:

            if cond == 'FR_CV':
                continue

            if electrode_recording_type == 'monopolaire':
                data = xr.open_dataarray(f'allsujet_{cond}_ROI.nc').loc[ROI,:,:]
            if electrode_recording_type == 'bipolaire':
                data = xr.open_dataarray(f'allsujet_{cond}_ROI_bi.nc').loc[ROI,:,:]

            #### stats
            if electrode_recording_type == 'monopolaire':
                pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}.npy')
            else:
                pixel_based_distrib = np.load(f'allsujet_{ROI}_tf_STATS_{cond}_bi.npy')

            contour_all_ROI[ROI][cond] = get_tf_stats(cond, data.values, pixel_based_distrib, nfrex)

    return contour_all_ROI













################################################
######## COMPUTE DATA RESPI PHASE ########
################################################

def generate_pseudo_network(electrode_recording_type, mask_dfc):

    #### display for phase

    mask_dfc_list = ['FR_CV', 'SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_re', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_re_01', 'AC_re_02', 'AC_re_03', 'AC_re_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
    
    mask_dfc_sel = [phase_i for phase_i, phase in enumerate(mask_dfc_list) if phase in ['SNIFF_pre_02', 'SNIFF_re', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_re_01', 'AC_re_02', 'AC_re_03', 'AC_re_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']]
    
    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['FR_CV', 'SNIFF', 'AC']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel = ['SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post', 
                      'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 'AC_resp_evnmt_01', 
                      'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04', 
                      'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
    phase_list_plot = ['pre', 're', 'post', 
                       'pre_01', 'pre_02', 'pre_03', 'pre_04', 
                       're_01', 're_02', 're_03', 're_04', 
                       'post_01', 'post_02', 'post_03', 'post_04']

    data_dfc = np.zeros((len(band_sel), len(ROI_list), len(phase_list_sel)))

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            for phase_i, phase in enumerate(phase_list_sel):

                data_dfc[band_i, ROI_i, phase_i] = np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'])
                
    #### plot
    max = np.array([np.percentile(data_dfc, 1)-np.median(data_dfc), np.percentile(data_dfc, 99)-np.median(data_dfc)]).max()

    for band_i, band in enumerate(band_sel):

        fig = plt.figure(figsize=(8,10))
        ax = plt.gca()

        im = ax.imshow(data_dfc[band_i,:,:], interpolation='none', vmin=-max, vmax=max, cmap='seismic')

        ax.set_xticks(np.arange(len(phase_list_plot)))
        ax.set_yticks(np.arange(len(ROI_list)))
        ax.set_xticklabels(phase_list_plot)
        ax.set_yticklabels(ROI_list)

        ax.vlines(x=[2.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g')
        ax.vlines(x=[0.5,1.5,6.5,10.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g', linestyle='dashed')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        ax.set_title(f"{band}", pad=55)

        mask_stats = mask_dfc[band_i,:,:][:,mask_dfc_sel]
        stat_text = mask_stats.copy().astype('object')
        stat_text[mask_stats] = '*'
        stat_text[~mask_stats] = ''
        # stat_text[:,0] = ''

        for (j,i),label in np.ndenumerate(stat_text):
            ax.text(i,j,label,ha='center',va='center', color='y', size=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)

        fig.tight_layout()

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))

        fig.savefig(f'{band}.png')
        plt.close('all')




    #### display for cond
        
    cond_to_plot = ['sniff', 'SA']

    mask_dfc_list = ['FR_CV', 'SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_re', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_re_01', 'AC_re_02', 'AC_re_03', 'AC_re_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
    mask_dfc_sel = {'sniff' : [phase_i for phase_i, phase in enumerate(mask_dfc_list) if phase in ['FR_CV', 'SNIFF_pre_02', 'SNIFF_re', 'SNIFF_post']],
                    'SA' : [phase_i for phase_i, phase in enumerate(mask_dfc_list) if phase.find('AC') != -1 or phase.find('FR_CV') != -1]}
        
    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['FR_CV', 'SNIFF', 'AC']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel = {'sniff' : ['FR_CV_whole', 'SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post'],
                      'SA' : ['FR_CV_whole', 'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 'AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04', 'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']}
    phase_list_plot = {'sniff' : ['FR_CV', 'pre', 're', 'post'],
                       'SA' : ['FR_CV', 'pre_01', 'pre_02', 'pre_03', 'pre_04', 
                       're_01', 're_02', 're_03', 're_04', 
                       'post_01', 'post_02', 'post_03', 'post_04']}

    data_dfc = {'sniff' : np.zeros((len(band_sel), len(ROI_list), len(phase_list_sel['sniff']))), 
                'SA' : np.zeros((len(band_sel), len(ROI_list), len(phase_list_sel['SA'])))}
    
    for cond_i, cond in enumerate(cond_to_plot):

        for band_i, band in enumerate(band_sel):
        
            for ROI_i, ROI in enumerate(ROI_list):

                for phase_i, phase in enumerate(phase_list_sel[cond]):

                    data_dfc[cond][band_i, ROI_i, phase_i] = np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'])
                
    #### plot
    max_list = []
    for cond in cond_to_plot:
        max_list.append(np.array([np.percentile(data_dfc[cond], 1)-np.median(data_dfc[cond]), np.percentile(data_dfc[cond], 99)-np.median(data_dfc[cond])]).max())
    max = np.array(max_list).max()

    for cond_i, cond in enumerate(cond_to_plot):

        for band_i, band in enumerate(band_sel):

            fig = plt.figure(figsize=(8,10))
            ax = plt.gca()

            im = ax.imshow(data_dfc[cond][band_i,:,:], interpolation='none', vmin=-max, vmax=max, cmap='seismic')

            ax.set_xticks(np.arange(len(phase_list_plot[cond])))
            ax.set_yticks(np.arange(len(ROI_list)))
            ax.set_xticklabels(phase_list_plot[cond])
            ax.set_yticklabels(ROI_list)

            if cond == 'sniff':
                ax.vlines(x=[0.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g')
                ax.vlines(x=[1.5,2.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g', linestyle='dashed')
            if cond == 'SA':
                ax.vlines(x=[0.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g')
                ax.vlines(x=[4.5,8.5], ymin=-0.5, ymax=len(ROI_list)-0.5, color='g', linestyle='dashed')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            ax.set_title(f"{cond} {band}", pad=55)

            mask_stats = mask_dfc[band_i,:,:][:,mask_dfc_sel[cond]]
            stat_text = mask_stats.copy().astype('object')
            stat_text[mask_stats] = '*'
            stat_text[~mask_stats] = ''
            stat_text[:,0] = ''

            for (j,i),label in np.ndenumerate(stat_text):
                ax.text(i,j,label,ha='center',va='center', color='y', size=20)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            plt.colorbar(im, cax=cax)

            fig.tight_layout()

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))

            fig.savefig(f'{cond}_{band}.png')
            plt.close('all')

        

            
            




################################
######## PXX EVOLUTION ########
################################

def get_pval(pval):

    if pval > 0.05:

        return pval

    elif pval < 0.001:

        return '***'

    elif pval < 0.01:

        return '**'

    elif pval < 0.05:

        return '*'
    



def get_data_lmm(df_TF, ROI_list, band_sel):

    data = {'ROI' : [], 'band' : [], 'phase' : [], 'coeff' : [], 'intercept' : [], 'pval' : [], 'p' : []}

    for ROI in ROI_list:

        for band in band_sel:

            #AC_chunk = 'AC_resp_evnmt'
            for AC_chunk in ['AC_pre', 'AC_resp_evnmt', 'AC_post']:

                phase_list = [_cond for _cond in df_TF['phase'].unique() if _cond.find(AC_chunk) != -1]

                _df_lmm = df_TF.query(f"ROI == '{ROI}' and phase in {phase_list} and band == '{band}'")[['sujet', 'Pxx', 'phase']]

                mdl = smf.mixedlm("Pxx ~ phase", data=_df_lmm, groups=_df_lmm['sujet'])
                mdl_fit = mdl.fit(method=["powell", "lbfgs"])

                coeff = float(mdl_fit.summary().tables[1]['Coef.'][1])
                pval = float(mdl_fit.summary().tables[1]['P>|z|'][1])
                intercept = float(mdl_fit.summary().tables[1]['Coef.'][0])

                if float(mdl_fit.summary().tables[1]['P>|z|'][1]) < 0.05:

                    p = 1

                else:

                    p = 0

                data['ROI'].append(ROI)
                data['band'].append(band)
                data['phase'].append(AC_chunk)
                data['coeff'].append(coeff)
                data['intercept'].append(intercept)
                data['pval'].append(pval)
                data['p'].append(p)

                
                # fig, axs = plt.subplots(ncols=2, figsize=(10,5))

                # sns.histplot(mdl_fit.resid, ax=axs[0])
                # sm.qqplot(mdl_fit.resid, dist=stats.norm, line='s', ax=axs[1])

                # plt.suptitle(f"{ROI}, sujet:{_df_lmm['sujet'].unique().shape[0]}")

                # fig.savefig(os.path.join(path_results, 'allplot', 'allcond', 'PSD_Coh', 'stats', f'{ROI}_qqplot_distrib_LMM_Cxy.png'))

                # plt.close()

                # plt.show()

    df_lmm = pd.DataFrame(data)

    return df_lmm


def get_data_lmm_xr(df_AC_allsig, ROI_list, band_sel):

    data = {'ROI' : [], 'band' : [], 'coeff' : [], 'intercept' : [], 'pval' : [], 'p' : [], 'z' : []}

    for ROI in ROI_list:

        for band in band_sel:

            _df_lmm = df_AC_allsig.query(f"ROI == '{ROI}' and band == '{band}'")[['sujet', 'Pxx', 'time']]

            mdl = smf.mixedlm("Pxx ~ time", data=_df_lmm, groups=_df_lmm['sujet'], re_formula="~time")
            mdl_fit = mdl.fit(method=["lbfgs"])

            coeff = float(mdl_fit.summary().tables[1]['Coef.'][1])
            pval = float(mdl_fit.summary().tables[1]['P>|z|'][1])
            intercept = float(mdl_fit.summary().tables[1]['Coef.'][0])
            z = float(mdl_fit.summary().tables[1]['z'][0])

            if float(pval) < 0.05:

                p = 1

            else:

                p = 0

            data['ROI'].append(ROI)
            data['band'].append(band)
            data['coeff'].append(coeff)
            data['intercept'].append(intercept)
            data['pval'].append(pval)
            data['p'].append(p)
            data['z'].append(z)

            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'stats_params_test'))

            fig, axs = plt.subplots(ncols=2, figsize=(10,5))

            sns.histplot(mdl_fit.resid, ax=axs[0])
            sm.qqplot(mdl_fit.resid, dist=scipy.stats.norm, line='s', ax=axs[1])

            plt.suptitle(f"{ROI}, {band}")

            fig.savefig(f'{ROI}_{band}.png')

            plt.close()

            # plt.show()

    df_lmm = pd.DataFrame(data)

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'stats_params_test'))

    df_lmm.to_excel('df_lmm.xlsx')

    return df_lmm



                
def plot_power_in_AC(electrode_recording_type):

    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['FR_CV', 'SNIFF', 'AC']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel = [_phase for _phase in df_TF['phase'].unique() if _phase not in ['FR_CV_expi', 'FR_CV_inspi']]
    phase_list_plot = ['FR_CV', 'SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_re', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_re_01', 'AC_re_02', 'AC_re_03', 'AC_re_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']

    dict_dfc = {'sujet' : [], 'band' : [], 'ROI' : [], 'phase' : [], 'Pxx' : []}

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            for phase_i, phase in enumerate(phase_list_sel):

                for sujet in sujet_list:

                    if sujet not in df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['sujet'].unique():

                        continue

                    else:

                        dict_dfc['sujet'].append(sujet)
                        dict_dfc['band'].append(band) 
                        dict_dfc['ROI'].append(ROI)
                        dict_dfc['phase'].append(phase)
                        dict_dfc['Pxx'].append(np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}' and sujet == '{sujet}'")['Pxx']))

    df_dfc = pd.DataFrame(dict_dfc)

    #### plot
    df_lmm = get_data_lmm(df_TF, ROI_list, band_sel)
    AC_phase_list = ['AC_pre', 'AC_resp_evnmt', 'AC_post']
    AC_phase_list_red = ['pre', 're', 'post']

    for ROI in ROI_list:

        # if df_lmm.query(f"ROI == '{ROI}'")['p'].values != 1:

        #     continue 

        for band_i, band in enumerate(band_sel):

            _phase_list_sel = [_cond for _cond in df_TF['phase'].unique() if _cond.find('AC') != -1]
            _df_lmm = df_dfc.query(f"ROI == '{ROI}' and band == '{band}' and phase in {_phase_list_sel}")[['sujet', 'phase', 'Pxx']]
            _df_lmm['phase_name'] = _df_lmm.loc[:, 'phase']

            for row_i in range(_df_lmm.shape[0]):

                _df_lmm['phase_name'].iloc[row_i] = f"{_df_lmm['phase_name'].iloc[row_i][3:-3]}"

            _df_lmm['phase'] = _df_lmm['phase'].str.slice_replace(0, -1, '')
            _df_lmm['phase'] = _df_lmm['phase'].astype('float')

            sns.lmplot(data=_df_lmm, x="phase", y="Pxx", hue='sujet', col='phase_name', legend=False)
            
            str_pval = ''
            for _phase_name_i, _phase_name in enumerate(AC_phase_list):
                
                pval = df_lmm.query(f"ROI == '{ROI}' and band == '{band}' and phase == '{_phase_name}'")['pval'].values[0]
                str_pval = f"{str_pval} {AC_phase_list_red[_phase_name_i]}:{get_pval(pval)}"

            plt.suptitle(f"{ROI} / {band} \n pval {str_pval}")

            plt.tight_layout()

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm'))
            plt.savefig(f'{ROI}_{band}.png')

            plt.close('all')




         
def plot_power_in_AC_allsig(electrode_recording_type):

    #### load prms
    band_prep = 'wb'
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    os.chdir(os.path.join(path_precompute, sujet_list[0], 'TF'))
    time_vec = np.linspace(-AC_length, AC_length*2, np.load(f'{sujet_list[0]}_tf_AC_bi.npy').shape[-1])
    cond = 'AC'
        
    #### prepare df
    xr_dict = {'sujet' : sujet_list, 'ROI' : ROI_list, 'band' : band_sel, 'time' : time_vec}
    xr_data = np.zeros((len(sujet_list), len(ROI_list), len(band_sel), time_vec.shape[0]))

    for sujet_i, sujet in enumerate(sujet_list):

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        for ROI_i, ROI in enumerate(ROI_list):

            print(sujet, ROI)

            prms = get_params(sujet, electrode_recording_type)
            df_loca = get_loca_df(sujet, electrode_recording_type)

            if ROI not in df_loca['ROI'].values:
                continue
            
            chan_list = df_loca.query(f"ROI == '{ROI}'")['name'].values
            chan_i_sel = [prms['chan_list_ieeg'].index(_chan) for _chan in chan_list]

            #band, freq = 'theta', [4, 8]
            for band_i, (band, freq) in enumerate(freq_band_dict_df_extraction[band_prep].items()):  

                #### load
                if electrode_recording_type == 'monopolaire':
                    data = np.median(np.median(np.load(f'{sujet}_tf_{cond}.npy')[chan_i_sel,:,:,:], axis=0), axis=0)
                else:
                    data = np.median(np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[chan_i_sel,:,:,:], axis=0), axis=0)

                #### sel freq
                mask_frex_band = (frex >= freq[0]) & (frex <= freq[-1])
                Pxx = data[mask_frex_band,:]

                xr_data[sujet_i, ROI_i, band_i, :] = np.median(Pxx, axis=0)

    xr_AC = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_AC = xr_AC.loc[:,:,:,0:12]

    df_AC_allsig = xr_AC.to_dataframe(name='Pxx').reset_index()
    df_AC_allsig = df_AC_allsig[df_AC_allsig['Pxx'] != 0]

    df_lmm = get_data_lmm_xr(df_AC_allsig, ROI_list, band_sel)

    time_vec_plot = np.linspace(0.1, 11.9, 7)
    for time_chunk_i, time_chunk in enumerate(time_vec_plot):

        if time_chunk_i == 0:
            df_AC_allsig_plot = df_AC_allsig.query(f"time < {time_chunk+0.1} and time > {time_chunk}").groupby(['sujet', 'ROI', 'band']).median().reset_index()
        else:
            _df_AC_allsig_plot = df_AC_allsig.query(f"time < {time_chunk+0.1} and time > {time_chunk}").groupby(['sujet', 'ROI', 'band']).median().reset_index()
            df_AC_allsig_plot = pd.concat([df_AC_allsig_plot, _df_AC_allsig_plot])

    # df_lmm = get_data_lmm_xr(df_AC_allsig_plot, ROI_list, band_sel)

    for ROI in ROI_list:

        # if df_lmm.query(f"ROI == '{ROI}'")['p'].values != 1:

        #     continue 

        for band_i, band in enumerate(band_sel):

            sns.lmplot(data=df_AC_allsig_plot.query(f"ROI == '{ROI}' and band == '{band}'"), x="time", y="Pxx", hue='sujet')
            plt.xlim(-1,13)               
            pval = df_lmm.query(f"ROI == '{ROI}' and band == '{band}'")['pval'].values[0]

            plt.suptitle(f"{ROI} / {band} \n pval:{get_pval(pval)}")

            # plt.tight_layout()

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm'))
            plt.savefig(f'{ROI}_{band}_allsig.png')

            plt.close('all')

    #### plot with true models
    
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'stats_params_test'))

    df_lmm = pd.read_excel('df_lmm.xlsx')

    for ROI in ROI_list:

        # if df_lmm.query(f"ROI == '{ROI}'")['p'].values != 1:

        #     continue 

        for band_i, band in enumerate(band_sel):

            df_AC_allsig_plot['time'] = np.round(df_AC_allsig_plot['time'].values).astype('int')

            fig, ax = plt.subplots(figsize=(6, 6))

            colors = {'pat_03083_1527':'tab:blue', 'pat_03105_1551':'tab:orange', 'pat_03128_1591':'tab:green', 
                      'pat_03138_1601':'tab:red', 'pat_03146_1608':'tab:purple', 'pat_03174_1634':'tab:brown'}

            _df_plot = df_AC_allsig_plot.query(f"ROI == '{ROI}' and band == '{band}'")
            ax.scatter(_df_plot['time'], _df_plot['Pxx'], c=_df_plot['sujet'].map(colors))
            plt.xlim(-1,13)               
            pval = df_lmm.query(f"ROI == '{ROI}' and band == '{band}'")['pval'].values[0]

            _lmm = df_lmm.query(f"ROI == '{ROI}' and band == '{band}'")
            a, b = _lmm['coeff'].values[0], _lmm['intercept'].values[0]

            for time_i, time in enumerate(df_AC_allsig_plot['time'].unique()):
    
                if time_i == 0:

                    _lmm_plot = pd.DataFrame({'ROI' : [ROI], 'band' : [band], 'time' : [time], 'Pxx' : [a*time + b]})

                else:

                    _lmm_plot = pd.concat([_lmm_plot, pd.DataFrame({'ROI' : [ROI], 'band' : [band], 'time' : [time], 'Pxx' : [a*time + b]})])

            plt.plot(_lmm_plot['time'], _lmm_plot['Pxx'], linewidth=5, color='k')

            df_subjects_lm_params = pd.DataFrame({'sujet' : [], 'Pxx' : [], 'time' : []})

            for sujet in _df_plot['sujet'].unique():

                _lmm_data = _df_plot.query(f"sujet == '{sujet}' and ROI == '{ROI}' and band == '{band}'")[['sujet', 'Pxx', 'time']]

                mdl = smf.mixedlm("Pxx ~ time", data=_lmm_data, groups=_lmm_data['sujet'], re_formula="~time")
                mdl_fit = mdl.fit(method=["lbfgs"])

                a = float(mdl_fit.summary().tables[1]['Coef.'][1])
                b = float(mdl_fit.summary().tables[1]['Coef.'][0])

                for time_i, time in enumerate(df_AC_allsig_plot['time'].unique()):

                    df_subjects_lm_params = pd.concat([df_subjects_lm_params, pd.DataFrame({'sujet' : [sujet], 'time' : [time], 'Pxx' : [a*time + b]})])

            for sujet in _df_plot['sujet'].unique():
    
                plt.plot(df_subjects_lm_params.query(f" sujet == '{sujet}'")['time'], df_subjects_lm_params.query(f" sujet == '{sujet}'")['Pxx'], 
                         c=colors[sujet], alpha=0.5)

            plt.suptitle(f"{ROI} / {band} \n pval:{get_pval(pval)}")

            # plt.tight_layout()

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm'))
            plt.savefig(f'realmodel_{ROI}_{band}_allsig.png')

            plt.close('all')






def plot_power_in_AC_allsig_every_subject(electrode_recording_type):

    #### load prms
    band_prep = 'wb'
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    os.chdir(os.path.join(path_precompute, sujet_list[0], 'TF'))
    time_vec = np.linspace(-AC_length, AC_length*2, np.load(f'{sujet_list[0]}_tf_AC_bi.npy').shape[-1])
    cond = 'AC'
    mask_time = (time_vec >= 0) & (time_vec <= 12)
        
    #### prepare df
    xr_data = {'sujet' : [], 'ROI' : [], 'band' : [], 'coef' : [], 'pval' : [], 'AL_time' : [], 'AL_num' : []}
    
    #sujet_i, sujet = 0, sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        os.chdir(os.path.join(path_prep, sujet, 'info'))
        df_AL = pd.read_excel(f"{sujet}_count_session.xlsx")

        #ROI_i, ROI = 0, ROI_list[0]
        for ROI_i, ROI in enumerate(ROI_list):

            print(sujet, ROI)

            prms = get_params(sujet, electrode_recording_type)
            df_loca = get_loca_df(sujet, electrode_recording_type)

            if ROI not in df_loca['ROI'].values:
                continue
            
            chan_list = df_loca.query(f"ROI == '{ROI}'")['name'].values
            chan_i_sel = [prms['chan_list_ieeg'].index(_chan) for _chan in chan_list]

            #band, freq = 'theta', [4, 8]
            for band_i, (band, freq) in enumerate(freq_band_dict_df_extraction[band_prep].items()):  

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                #### load
                if electrode_recording_type == 'monopolaire':
                    data = np.median(np.median(np.load(f'{sujet}_tf_{cond}.npy')[chan_i_sel,:,:,:], axis=0), axis=0)
                else:
                    data = np.median(np.median(np.load(f'{sujet}_tf_{cond}_bi.npy')[chan_i_sel,:,:,:], axis=0), axis=0)

                #### sel freq
                mask_frex_band = (frex >= freq[0]) & (frex <= freq[-1])
                Pxx = data[mask_frex_band,:]
                Pxx = np.median(Pxx, axis=0)

                x = time_vec[mask_time]
                y = Pxx[mask_time]

                x = sm.add_constant(x)

                model = sm.OLS(y, x).fit()

                for AL_i in range(3):
                
                    xr_data['sujet'].append(sujet)
                    xr_data['ROI'].append(ROI)
                    xr_data['band'].append(band)
                    xr_data['coef'].append(np.round(model.summary2().tables[1]['Coef.'][1],5))
                    xr_data['pval'].append(np.round(model.summary2().tables[1]['P>|t|'][1],5))
                    xr_data['AL_time'].append(df_AL[f'AL_{AL_i+1}'][0])
                    xr_data['AL_num'].append(f'AL_{AL_i+1}')
                    
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'regression_AL'))
    pd.DataFrame(xr_data).to_excel('lm_allsujet_AL.xlsx')    

    

def plot_matrix_lmm():

    #### load prms
    band_prep = 'wb'
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    band_sel = list(freq_band_dict_df_extraction['wb'].keys())
    cond = 'AC'
    thresh_stat = 0.05
    
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'stats_params_test'))

    df_lmm = pd.read_excel('df_lmm.xlsx')
    
    data_lmm_coeff = np.zeros((len(ROI_list), len(band_sel)))
    mask_lmm_stats = np.zeros((len(ROI_list), len(band_sel)))

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            data_lmm_coeff[ROI_i, band_i] = df_lmm.query(f"ROI == '{ROI}' and band == '{band}'")['coeff'].values[0]

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            if df_lmm.query(f"ROI == '{ROI}' and band == '{band}'")['pval'].values[0] <= thresh_stat:

                mask_lmm_stats[ROI_i, band_i] = True

            else:

                mask_lmm_stats[ROI_i, band_i] = False

    mask_lmm_stats = mask_lmm_stats.astype('bool')

    max = data_lmm_coeff.max()
    
    fig = plt.figure(figsize=(8,10))
    ax = plt.gca()

    im = ax.imshow(data_lmm_coeff, interpolation='none', vmin=-max, vmax=max, cmap='seismic')

    ax.set_xticks(np.arange(len(band_sel)))
    ax.set_yticks(np.arange(len(ROI_list)))
    ax.set_xticklabels(band_sel)
    ax.set_yticklabels(ROI_list)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    stat_text = mask_lmm_stats.copy().astype('object')
    stat_text[mask_lmm_stats] = '*'
    stat_text[~mask_lmm_stats] = ''

    for (j,i),label in np.ndenumerate(stat_text):
        ax.text(i,j,label,ha='center',va='center', color='y', size=50)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)

    fig.tight_layout()

    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'lmm', 'stats_params_test'))

    fig.savefig(f'matrix_lmm_stats.png')
    plt.close('all')

    
                    

def plot_network_summary_slow_freq(electrode_recording_type, mask_dfc):

    mask_dfc_list = ['FR_CV', 'SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
    
    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['SNIFF', 'AC']
    band_sel = ['theta', 'alpha']
    phase_list = ['pre', 're', 'post']
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel_sniff = ['SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post'] 
    phase_list_sel_ac = {'pre' : ['AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04'], 
                         're' : ['AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04'], 
                      'post' : ['AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']}
    phase_list_plot = ['pre', 're', 'post', 
                       'pre_01', 'pre_02', 'pre_03', 'pre_04', 
                       're_01', 're_02', 're_03', 're_04', 
                       'post_01', 'post_02', 'post_03', 'post_04']

    df_dfc = pd.DataFrame({'cond' : [], 'band' : [], 'ROI' : [], 'phase' : [], 'stat' : [], 'Pxx' : []})

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            for cond_i, cond in enumerate(cond_sel):

                if cond == 'SNIFF':

                    for phase_i, phase in enumerate(phase_list_sel_sniff):

                        _Pxx = np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'])

                        _stats = mask_dfc[band_i, ROI_i, mask_dfc_list.index(phase)]
                        
                        df_append = pd.DataFrame({'cond' : [cond], 'band' : [band], 'ROI' : [ROI], 'phase' : [phase_list[phase_i]], 'stat' : [_stats*1], 'Pxx' : [_Pxx]})
                        df_dfc = pd.concat([df_dfc, df_append])
                
                if cond == 'AC':
                    for phase_i, sel_keys in enumerate(phase_list_sel_ac.items()):
                        
                        _Pxx_list = []
                        _stat_list = []
                        for phase in sel_keys[1]:
                            _Pxx_list.append(np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx']))
                            _stat_list.append(mask_dfc[band_i, ROI_i, mask_dfc_list.index(phase)])
                        if (np.array(_stat_list)*1).sum() != 0:
                            _stats = 1
                        else:
                            _stats = 0

                        _Pxx = np.median(_Pxx_list)
                        
                        df_append = pd.DataFrame({'cond' : [cond], 'band' : [band], 'ROI' : [ROI], 'phase' : [phase_list[phase_i]], 'stat' : [_stats], 'Pxx' : [_Pxx]})
                        df_dfc = pd.concat([df_dfc, df_append])

    df_dfc_slow_freq = pd.DataFrame({'cond' : [], 'ROI' : [], 'phase' : [], 'stat' : [], 'Pxx' : []})

    for ROI_i, ROI in enumerate(ROI_list):

        for cond_i, cond in enumerate(cond_sel):

            for phase_i, phase in enumerate(phase_list):

                _stat_list = []
                _Pxx_list = []

                for band_i, band in enumerate(band_sel):

                    _stat_list.append(df_dfc.query(f"cond == '{cond}' and phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['stat'].values[0])
                    _Pxx_list.append(df_dfc.query(f"cond == '{cond}' and phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'].values[0])

                if np.array(_stat_list).sum() != 0:
                    _stat = 1
                else:
                    _stat = 0

                _Pxx = np.median(_Pxx_list)

                df_append = pd.DataFrame({'cond' : [cond], 'ROI' : [ROI], 'phase' : [phase], 'stat' : [_stat], 'Pxx' : [_Pxx]})
                df_dfc_slow_freq = pd.concat([df_dfc_slow_freq, df_append])
    
    ROI_list_short = ['AMY', 'HP', 'pINS', 'Tinf', 'Tmed', 'Tsup', 'pHIPP']

    # Conditionally replace text in the DataFrame
    for ROI_i, ROI in enumerate(ROI_list):
        ROI_to_replace = ROI_list_short[ROI_i]
        df_dfc_slow_freq['ROI'] = df_dfc_slow_freq['ROI'].replace(ROI, ROI_to_replace)

    # choose color 
    color_list = []
    for row in range(df_dfc_slow_freq.shape[0]):

        if df_dfc_slow_freq.iloc[row]['stat'] == 0 and df_dfc_slow_freq.iloc[row]['Pxx'] > 0:
            color_list.append('magenta')

        if df_dfc_slow_freq.iloc[row]['stat'] == 1 and df_dfc_slow_freq.iloc[row]['Pxx'] > 0:
            color_list.append('red')

        if df_dfc_slow_freq.iloc[row]['stat'] == 0 and df_dfc_slow_freq.iloc[row]['Pxx'] < 0:
            color_list.append('cyan')

        if df_dfc_slow_freq.iloc[row]['stat'] == 1 and df_dfc_slow_freq.iloc[row]['Pxx'] < 0:
            color_list.append('blue')

    df_dfc_slow_freq['color'] = color_list

    # To plot negative values in graph make all positiv
    df_dfc_slow_freq['Pxx'] = np.abs(df_dfc_slow_freq['Pxx'])
         
    #### plot
    
    for phase_i, phase in enumerate(phase_list):

        for cond_i, cond in enumerate(cond_sel):

            df_plot = df_dfc_slow_freq.query(f"cond == '{cond}' and phase == '{phase}'")

            # Create a graph
            plt.figure(figsize=(10,10))
            G = nx.Graph()

            # Add nodes with names, sizes, and colors based on the provided columns
            for index, row in df_plot.iterrows():
                G.add_node(row['ROI'], size=row['Pxx']*300, color=row['color'])

            # Add edges (you can modify this part based on your specific dataset structure)
            G.add_edges_from([('AMY', 'HP'), ('HP', 'pINS'), ('pINS', 'Tinf'), ('Tinf', 'Tmed'), ('Tmed', 'Tsup'), ('Tsup', 'pHIPP'), ('pHIPP', 'AMY')])

            # Get node sizes and colors from the respective attributes
            node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes]
            node_colors = [G.nodes[n]['color'] for n in G.nodes]

            # Draw the network plot
            pos = nx.circular_layout(G, scale=5)
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
                    font_size=50, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)

            # Display the plot
            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))
            plt.savefig(f"SLOW_graph_{cond}_{phase}.png", bbox_inches='tight')
            plt.close()

    #### plot only signi
            
    df_dfc_slow_freq_only_signi = df_dfc_slow_freq.copy()
    df_dfc_slow_freq_only_signi.loc[df_dfc_slow_freq_only_signi['color'] == 'magenta','Pxx'] = 0
    df_dfc_slow_freq_only_signi.loc[df_dfc_slow_freq_only_signi['color'] == 'cyan','Pxx'] = 0

    for phase_i, phase in enumerate(phase_list):

        for cond_i, cond in enumerate(cond_sel):

            df_plot = df_dfc_slow_freq_only_signi.query(f"cond == '{cond}' and phase == '{phase}'")

            # Create a graph
            plt.figure(figsize=(10,10))
            G = nx.Graph()

            # Add nodes with names, sizes, and colors based on the provided columns
            for index, row in df_plot.iterrows():
                G.add_node(row['ROI'], size=row['Pxx']*300, color=row['color'])

            # Add edges (you can modify this part based on your specific dataset structure)
            G.add_edges_from([('AMY', 'HP'), ('HP', 'pINS'), ('pINS', 'Tinf'), ('Tinf', 'Tmed'), ('Tmed', 'Tsup'), ('Tsup', 'pHIPP'), ('pHIPP', 'AMY')])

            # Get node sizes and colors from the respective attributes
            node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes]
            node_colors = [G.nodes[n]['color'] for n in G.nodes]

            # Draw the network plot
            pos = nx.circular_layout(G, scale=5)
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
                    font_size=50, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)

            # Display the plot
            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))
            plt.savefig(f"SLOW_graph_only_signi_{cond}_{phase}.png", bbox_inches='tight')
            plt.close()



def plot_network_summary_high_freq(electrode_recording_type, mask_dfc):

    band_i_sel_gamma = 3

    mask_dfc_list = ['FR_CV', 'SNIFF_pre_01', 'SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post', 
                       'AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04', 
                       'AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04', 
                       'AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']
    
    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    if electrode_recording_type == 'monopolaire':
        df_TF = pd.read_excel('allplot_df_TF.xlsx', index_col = 0)
    else:
        df_TF = pd.read_excel('allplot_df_TF_bi.xlsx', index_col = 0)
    
    ROI_list = ['amygdala', 'hippocampus', 'insula post', 'temporal inf', 'temporal med', 'temporal sup', 'parahippocampique']
    cond_sel = ['SNIFF', 'AC']
    band_sel = ['gamma']
    phase_list = ['pre', 're', 'post']
    df_TF = df_TF.query(f"ROI in {ROI_list} and cond in {cond_sel}")

    phase_list_sel_sniff = ['SNIFF_pre_02', 'SNIFF_resp_evnmt', 'SNIFF_post'] 
    phase_list_sel_ac = {'pre' : ['AC_pre_01', 'AC_pre_02', 'AC_pre_03', 'AC_pre_04'], 
                         're' : ['AC_resp_evnmt_01', 'AC_resp_evnmt_02', 'AC_resp_evnmt_03', 'AC_resp_evnmt_04'], 
                      'post' : ['AC_post_01', 'AC_post_02', 'AC_post_03', 'AC_post_04']}
    phase_list_plot = ['pre', 're', 'post', 
                       'pre_01', 'pre_02', 'pre_03', 'pre_04', 
                       're_01', 're_02', 're_03', 're_04', 
                       'post_01', 'post_02', 'post_03', 'post_04']

    df_dfc = pd.DataFrame({'cond' : [], 'band' : [], 'ROI' : [], 'phase' : [], 'stat' : [], 'Pxx' : []})

    for band_i, band in enumerate(band_sel):
    
        for ROI_i, ROI in enumerate(ROI_list):

            for cond_i, cond in enumerate(cond_sel):

                if cond == 'SNIFF':

                    for phase_i, phase in enumerate(phase_list_sel_sniff):

                        _Pxx = np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'])

                        _stats = mask_dfc[band_i_sel_gamma, ROI_i, mask_dfc_list.index(phase)]
                        
                        df_append = pd.DataFrame({'cond' : [cond], 'band' : [band], 'ROI' : [ROI], 'phase' : [phase_list[phase_i]], 'stat' : [_stats*1], 'Pxx' : [_Pxx]})
                        df_dfc = pd.concat([df_dfc, df_append])
                
                if cond == 'AC':
                    for phase_i, sel_keys in enumerate(phase_list_sel_ac.items()):
                        
                        _Pxx_list = []
                        _stat_list = []
                        for phase in sel_keys[1]:
                            _Pxx_list.append(np.median(df_TF.query(f"phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx']))
                            _stat_list.append(mask_dfc[band_i_sel_gamma, ROI_i, mask_dfc_list.index(phase)])
                        if (np.array(_stat_list)*1).sum() != 0:
                            _stats = 1
                        else:
                            _stats = 0

                        _Pxx = np.median(_Pxx_list)
                        
                        df_append = pd.DataFrame({'cond' : [cond], 'band' : [band], 'ROI' : [ROI], 'phase' : [phase_list[phase_i]], 'stat' : [_stats], 'Pxx' : [_Pxx]})
                        df_dfc = pd.concat([df_dfc, df_append])

    df_dfc_high_freq = pd.DataFrame({'cond' : [], 'ROI' : [], 'phase' : [], 'stat' : [], 'Pxx' : []})

    for ROI_i, ROI in enumerate(ROI_list):

        for cond_i, cond in enumerate(cond_sel):

            for phase_i, phase in enumerate(phase_list):

                _stat_list = []
                _Pxx_list = []

                for band_i, band in enumerate(band_sel):

                    _stat_list.append(df_dfc.query(f"cond == '{cond}' and phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['stat'].values[0])
                    _Pxx_list.append(df_dfc.query(f"cond == '{cond}' and phase == '{phase}' and ROI == '{ROI}' and band == '{band}'")['Pxx'].values[0])

                if np.array(_stat_list).sum() != 0:
                    _stat = 1
                else:
                    _stat = 0

                _Pxx = np.median(_Pxx_list)

                df_append = pd.DataFrame({'cond' : [cond], 'ROI' : [ROI], 'phase' : [phase], 'stat' : [_stat], 'Pxx' : [_Pxx]})
                df_dfc_high_freq = pd.concat([df_dfc_high_freq, df_append])
    
    ROI_list_short = ['AMY', 'HP', 'pINS', 'Tinf', 'Tmed', 'Tsup', 'pHIPP']

    # Conditionally replace text in the DataFrame
    for ROI_i, ROI in enumerate(ROI_list):
        ROI_to_replace = ROI_list_short[ROI_i]
        df_dfc_high_freq['ROI'] = df_dfc_high_freq['ROI'].replace(ROI, ROI_to_replace)

    # choose color 
    color_list = []
    for row in range(df_dfc_high_freq.shape[0]):

        if df_dfc_high_freq.iloc[row]['stat'] == 0 and df_dfc_high_freq.iloc[row]['Pxx'] > 0:
            color_list.append('magenta')

        if df_dfc_high_freq.iloc[row]['stat'] == 1 and df_dfc_high_freq.iloc[row]['Pxx'] > 0:
            color_list.append('red')

        if df_dfc_high_freq.iloc[row]['stat'] == 0 and df_dfc_high_freq.iloc[row]['Pxx'] < 0:
            color_list.append('cyan')

        if df_dfc_high_freq.iloc[row]['stat'] == 1 and df_dfc_high_freq.iloc[row]['Pxx'] < 0:
            color_list.append('blue')

    df_dfc_high_freq['color'] = color_list

    # To plot negative values in graph make all positiv
    df_dfc_high_freq['Pxx'] = np.abs(df_dfc_high_freq['Pxx'])
         
    #### plot
    
    for phase_i, phase in enumerate(phase_list):

        for cond_i, cond in enumerate(cond_sel):

            df_plot = df_dfc_high_freq.query(f"cond == '{cond}' and phase == '{phase}'")

            # Create a graph
            plt.figure(figsize=(10,10))
            G = nx.Graph()

            # Add nodes with names, sizes, and colors based on the provided columns
            for index, row in df_plot.iterrows():
                G.add_node(row['ROI'], size=row['Pxx']*300, color=row['color'])

            # Add edges (you can modify this part based on your specific dataset structure)
            G.add_edges_from([('AMY', 'HP'), ('HP', 'pINS'), ('pINS', 'Tinf'), ('Tinf', 'Tmed'), ('Tmed', 'Tsup'), ('Tsup', 'pHIPP'), ('pHIPP', 'AMY')])

            # Get node sizes and colors from the respective attributes
            node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes]
            node_colors = [G.nodes[n]['color'] for n in G.nodes]

            # Draw the network plot
            pos = nx.circular_layout(G, scale=5)
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
                    font_size=50, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)

            # Display the plot
            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))
            plt.savefig(f"HIGH_graph_{cond}_{phase}.png", bbox_inches='tight')
            plt.close()

    #### plot only signi
            
    df_dfc_slow_freq_only_signi = df_dfc_high_freq.copy()
    df_dfc_slow_freq_only_signi.loc[df_dfc_slow_freq_only_signi['color'] == 'magenta','Pxx'] = 0
    df_dfc_slow_freq_only_signi.loc[df_dfc_slow_freq_only_signi['color'] == 'cyan','Pxx'] = 0

    for phase_i, phase in enumerate(phase_list):

        for cond_i, cond in enumerate(cond_sel):

            df_plot = df_dfc_slow_freq_only_signi.query(f"cond == '{cond}' and phase == '{phase}'")

            # Create a graph
            plt.figure(figsize=(10,10))
            G = nx.Graph()

            # Add nodes with names, sizes, and colors based on the provided columns
            for index, row in df_plot.iterrows():
                G.add_node(row['ROI'], size=row['Pxx']*300, color=row['color'])

            # Add edges (you can modify this part based on your specific dataset structure)
            G.add_edges_from([('AMY', 'HP'), ('HP', 'pINS'), ('pINS', 'Tinf'), ('Tinf', 'Tmed'), ('Tmed', 'Tsup'), ('Tsup', 'pHIPP'), ('pHIPP', 'AMY')])

            # Get node sizes and colors from the respective attributes
            node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes]
            node_colors = [G.nodes[n]['color'] for n in G.nodes]

            # Draw the network plot
            pos = nx.circular_layout(G, scale=5)
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
                    font_size=50, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)

            # Display the plot
            os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'network'))
            plt.savefig(f"HIGH_graph_only_signi_{cond}_{phase}.png", bbox_inches='tight')
            plt.close()








################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #electrode_recording_type = 'bipolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        contour_all_ROI = get_stats_coutour_from_TF(electrode_recording_type)

        stats_type = 'perm'
        # stats_type = 'ttest'
        mask_dfc = get_stats_TF_pseudo_network(electrode_recording_type, stats_type, contour_all_ROI)
        generate_pseudo_network(electrode_recording_type, mask_dfc)

        plot_power_in_AC_allsig(electrode_recording_type)
        plot_power_in_AC_allsig_every_subject(electrode_recording_type)

        plot_matrix_lmm()
        plot_network_summary_slow_freq(electrode_recording_type, mask_dfc)
        plot_network_summary_high_freq(electrode_recording_type, mask_dfc)
        








