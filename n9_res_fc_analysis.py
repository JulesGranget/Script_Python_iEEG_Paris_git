



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib


from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False




#######################################
############# ISPC & PLI #############
#######################################




#band_prep, freq, band, cond, prms = 'lf', [4, 8], 'theta', 'FR_CV', prms
def compute_fc_metrics_mat(band_prep, freq, band, cond, prms):

    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    if os.path.exists(f'{sujet}_ISPC_{band}_{cond}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_ISPC_{band}_{cond}')
        ispc_mat = np.load(f'{sujet}_ISPC_{band}_{cond}.npy')

    if os.path.exists(f'{sujet}_PLI_{band}_{cond}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_PLI_{band}_{cond}')
        pli_mat = np.load(f'{sujet}_PLI_{band}_{cond}.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    
    #### load_data
    data = load_data(cond, band_prep=band_prep)

    #### get wavelets
    wavelets, nfrex = get_wavelets(band_prep, freq)

    #### get only EEG chan
    data = data[:len(prms['chan_list_ieeg']),:]

    #### compute all convolution
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

    print('CONV')

    def convolution_x_wavelets_nchan(nchan):

        print_advancement(nchan, np.size(data,0), steps=[25, 50, 75])
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data,0)))

    #### verif conv
    if debug:
        for nchan in range(10):
            for i in range(5):
                i = nchan*5 + i
                plt.plot(np.mean(np.real(convolutions[i]), 0))
            plt.show()


    if cond == 'FR_CV':

        #### compute metrics
        pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
        ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

        print('COMPUTE')

        for seed in range(np.size(data,0)) :

            print_advancement(seed, np.size(data,0), steps=[25, 50, 75])

            def compute_ispc_pli(nchan):

                if nchan == seed : 
                    return 
                    
                else :

                    # initialize output time-frequency data
                    ispc = np.zeros((nfrex))
                    pli  = np.zeros((nfrex))

                    # compute metrics
                    for fi in range(nfrex):
                        
                        as1 = convolutions[seed][fi,:]
                        as2 = convolutions[nchan][fi,:]

                        # collect "eulerized" phase angle differences
                        cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                        
                        # compute ISPC and PLI (and average over trials!)
                        ispc[fi] = np.abs(np.mean(cdd))
                        pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

                    # compute mean
                    mean_ispc = np.mean(ispc,0)
                    mean_pli = np.mean(pli,0)

                    return mean_ispc, mean_pli

            compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(nchan) for nchan in range(np.size(data,0)))
            
            #### load in mat    
            for nchan in range(np.size(data,0)) :
                    
                if nchan == seed:

                    continue

                else:
                        
                    ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                    pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

        #### save matrix
        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        np.save(f'{sujet}_ISPC_{band}_{cond}.npy', ispc_mat)

        np.save(f'{sujet}_PLI_{band}_{cond}.npy', pli_mat)

        #### supress mmap
        os.chdir(path_memmap)
        os.remove(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat')
        
        return pli_mat, ispc_mat


    





def get_pli_ispc_allsession(sujet):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, 'AC'
                for cond_i, cond in enumerate(['FR_CV']) :

                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif conv
    if debug == True:
        for band in pli_allband.keys():
            for cond in pli_allband[band].keys():
                plt.matshow(pli_allband[band][cond][0])
                plt.show()

    #### verif

    if debug == True:
                
        for band_prep in band_prep_list:
            
            for band, freq in freq_band_dict_FC[band_prep].items():

                for cond_i, cond in enumerate(['FR_CV']) :

                    print(band, cond, len(pli_allband[band][cond]))
                    print(band, cond, len(ispc_allband[band][cond]))


        #### reduce to one cond
    #### generate dict to fill
    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole':

                continue

            ispc_allband_reduced[band] = {}
            pli_allband_reduced[band] = {}

            for cond_i, cond in enumerate(['FR_CV']) :

                ispc_allband_reduced[band][cond] = []
                pli_allband_reduced[band][cond] = []

    #### fill
    
    for band_prep_i, band_prep in enumerate(band_prep_list):

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else:

                for cond_i, cond in enumerate(['FR_CV']) :

                    ispc_allband_reduced[band][cond] = ispc_allband[band][cond][0]
                    pli_allband_reduced[band][cond] = pli_allband[band][cond][0]


    return pli_allband_reduced, ispc_allband_reduced









################################
######## SAVE FIG ########
################################

def get_sorting(df_loca):

    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    index_sorted = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    chan_name_sorted_no_rep = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            chan_name_sorted_no_rep.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_no_rep.append('')
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_no_rep.append(name_i)
                rep_count = 0
                continue

    return df_sorted, index_sorted, chan_name_sorted_no_rep



def sort_mat(mat, index_new):

    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    for i_before_sort_r, i_sort_r in enumerate(index_new):
        for i_before_sort_c, i_sort_c in enumerate(index_new):
            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    #### verify sorting
    if debug:
        mat_sorted = sort_mat(mat)
        plt.matshow(mat_sorted)
        plt.show()

    return mat_sorted


#mat = ispc_allband_reduced[band][cond]
def get_mat_mean(mat, df_sorted):

    #### extract infos
    index_new = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    roi_in_data = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            roi_in_data.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                roi_in_data.append(name_i)
                rep_count = 0
                continue

    #### sort mat
    mat_sorted = sort_mat(mat, index_new)
    
    #### mean mat
    indexes_to_compute = {}
    for roi_i, roi_name in enumerate(roi_in_data):        
        i_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == roi_name]
        indexes_to_compute[roi_name] = i_to_mean
        
    mat_mean = np.zeros(( len(roi_in_data), len(roi_in_data) ))
    for roi_i_x, roi_name_x in enumerate(roi_in_data):        
        roi_chunk_dfc = mat_sorted[indexes_to_compute[roi_name_x],:]
        roi_chunk_dfc_mean = np.mean(roi_chunk_dfc, 0)
        coeff_i = []
        for roi_i_y, roi_name_y in enumerate(roi_in_data):
            if roi_name_x == roi_name_y:
                coeff_i.append(0)
                continue
            else:
                coeff_i.append( np.mean(roi_chunk_dfc_mean[indexes_to_compute[roi_name_y]]) )
        coeff_i = np.array(coeff_i)
        mat_mean[roi_i_x,:] = coeff_i

    #### verif
    if debug:
        plt.matshow(mat_mean)
        plt.show()

    return mat_mean


def mat_tresh(mat, percentile_thresh):

    thresh_value = np.percentile(mat.reshape(-1), percentile_thresh)

    for x in range(mat.shape[1]):
        for y in range(mat.shape[1]):
            if mat[x, y] < thresh_value:
                mat[x, y] = 0
            if mat[y, x] < thresh_value:
                mat[y, x] = 0

    #### verif
    if debug:
        plt.matshow(mat)
        plt.show()

    return mat

    


def save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, conditions_to_plot, prms):

    print('######## SAVEFIG FC ########')

    df_sorted, index_sorted, chan_name_sorted_no_rep = get_sorting(df_loca)
    roi_in_data = df_sorted.drop_duplicates(subset=['ROI'])['ROI'].values.tolist()

    #### identify scale
    scale = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}

    scale['ispc']['max'] = {}
    scale['ispc']['min'] = {}
    scale['pli']['max'] = {}
    scale['pli']['min'] = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole':
                continue

            band_ispc = {'min' : [], 'max' : []}
            band_pli = {'min' : [], 'max' : []}

            for cond_i, cond in enumerate(conditions_to_plot):
                band_ispc['max'].append(np.max(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted)))
                band_ispc['min'].append(np.min(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted)))
                
                band_pli['max'].append(np.max(get_mat_mean(pli_allband_reduced[band][cond], df_sorted)))
                band_pli['min'].append(np.min(get_mat_mean(pli_allband_reduced[band][cond], df_sorted)))

            scale['ispc']['max'][band] = np.max(band_ispc['max'])
            scale['ispc']['min'][band] = np.min(band_ispc['min'])
            scale['pli']['max'][band] = np.max(band_pli['max'])
            scale['pli']['min'][band] = np.min(band_pli['min'])


    #### params
    nrows, ncols = 1, 1
    conditions_to_plot = ['FR_CV']
    percentile_thresh = 90

    #### ISPC

    #band_prep, band = 'lf', 'theta'
    for band_prep in band_prep_list:

        for band in ispc_allband_reduced.keys():

            #### graph
            fig = plt.figure(facecolor='black')
            mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)

            for cond_i, cond in enumerate(conditions_to_plot):
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1),
                vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
            plt.suptitle('ISPC_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'figures'))

            fig.savefig(sujet + '_ISPC_' + band + '_graph.jpeg', dpi = 100)

            plt.close()

        
            #### matrix
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))
            mat_to_plot = mat_tresh(get_mat_mean(ispc_allband_reduced[band][cond], df_sorted), percentile_thresh)
            for c, cond_i in enumerate(conditions_to_plot):

                ax.matshow(mat_to_plot, vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                ax.set_title(cond_i)
                ax.set_yticks(range(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
                        
            plt.suptitle('ISPC_' + band)
            #plt.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
                        
            fig.savefig(sujet + '_ISPC_' + band + '_mat.jpeg', dpi = 100)

            plt.close()


    #### PLI

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band in ispc_allband_reduced.keys():

            #### graph
            fig = plt.figure(facecolor='black')
            mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)

            for cond_i, cond in enumerate(conditions_to_plot):
                mne.viz.plot_connectivity_circle(mat_to_plot, node_names=roi_in_data, n_lines=None, 
                title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1),
                vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
            plt.suptitle('PLI_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'figures'))

            fig.savefig(sujet + '_PLI_' + band + '_graph.jpeg', dpi = 100)

            plt.close()

        
            #### matrix
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))
            mat_to_plot = mat_tresh(get_mat_mean(pli_allband_reduced[band][cond], df_sorted), percentile_thresh)
            for c, cond_i in enumerate(conditions_to_plot):

                ax.matshow(mat_to_plot, vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                ax.set_title(cond_i)
                ax.set_yticks(range(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
                        
            plt.suptitle('PLI_' + band)
            #plt.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
                        
            fig.savefig(sujet + '_PLI_' + band + '_mat.jpeg', dpi = 100)

            plt.close()






def save_fig_for_allsession_FR_CV(sujet):

    prms = get_params(sujet)

    df_loca = get_loca_df(sujet)

    pli_allband_reduced, ispc_allband_reduced = get_pli_ispc_allsession(sujet)

    conditions_to_plot = ['FR_CV']

    save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, conditions_to_plot, prms)




########################
######## DFC ########
########################


def save_fig_dfc_AL(sujet):

    print('######## SAVE FIG AL ########')

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    #### find n session
    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find('ispc_theta_AL') != -1) :
            load_i.append(i)
        else:
            continue

    n_session = len(load_i)

    dict_allband = {}

    #### compute for every AL
    for band_i, band in enumerate(freq_band_whole):

        if band == 'whole':
            continue

        dict_allband[band] = {}

        #AL_i = 0
        for AL_i in range(n_session):

            #### open data
            load_i = []
            for i, session_name in enumerate(os.listdir()):
                if (session_name.find('DFC') != -1) and (session_name.find(f'AL{AL_i+1}') != -1) and (session_name.find(band) != -1):
                    load_i.append(i)
                else:
                    continue

            load_list = [os.listdir()[i] for i in load_i]

            #### generate mat
            xr_sniff = xr.open_dataarray(load_list[0])
            data_alltypes = np.zeros(( 2, xr_sniff.shape[0], xr_sniff.shape[1] ))

            #mat_type = 'pli'
            for mat_type_i, mat_type in enumerate(['ispc', 'pli']):
                #band_prep = 'lf'
            
                load_i = []
                for i, session_name in enumerate(load_list):
                    if (session_name.find(mat_type) != -1):
                        load_i.append(session_name)
                    else:
                        continue

                xr_load = xr.open_dataarray(load_i[0])

                data_alltypes[mat_type_i, :, :] = xr_load.data

            dict_xr_load = {'mat_type' : ['ispc', 'pli'], 'pairs' : xr_load['pairs'], 'times' : xr_load['times']}
            xr_load_dict = xr.DataArray(data_alltypes, coords=dict_xr_load.values(), dims=dict_xr_load.keys())

            dict_allband[band][f'AL{AL_i+1}'] = xr_load_dict

            pairs = xr_load_dict['pairs'].values

    #### plot
    os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', 'AL'))

    #pair = pairs[0]
    for pair in pairs:
        #mat_type = 'ispc'
        for mat_type_i, mat_type in enumerate(['ispc', 'pli']):

            fig, axs = plt.subplots(nrows=len(freq_band_fc_analysis.keys()))

            for band_i, band in enumerate(freq_band_fc_analysis.keys()):

                ax = axs[band_i]

                #AL_i = 0
                for AL_i in range(n_session):

                    _times = dict_allband[band][f'AL{AL_i+1}']['times'].values / dict_allband[band][f'AL{AL_i+1}']['times'].values.max()
                    ax.plot(_times, dict_allband[band][f'AL{AL_i+1}'].loc[mat_type, pair, :], label=f'AL{AL_i+1}')

                if band_i == 0:
                    ax.set_title(f'{pair} {mat_type}')
                    ax.legend()
                
                ax.set_ylabel(band)

            #plt.show()

            fig.set_figheight(15)
            fig.set_figwidth(15)
            fig.savefig(f'{sujet}_{mat_type}_{pair}.png')

            plt.close()





#cond, band_prep = 'SNIFF', 'hf'
def save_fig_dfc_AC_SNIFF(sujet, cond, band_prep):

    print(f'######## {cond} SAVEFIG ########')

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    dict_allband = {}

    #band_i, band = 0, 'l_gamma'
    for band_i, band in enumerate(freq_band_dict_FC[band_prep].keys()):

        dict_allband[band] = {}

        #### open data
        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if (session_name.find(f'pli_ispc_{band}_{cond}') != -1):
                load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        #### generate mat
        xr_load = xr.open_dataarray(load_list[0])

        dict_allband[band] = xr_load

    #### plot
    os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

    pairs = dict_allband[band]['pairs'].values

    #pair = pairs[0]
    for pair in pairs:
        #mat_type = 'ispc'
        for mat_type_i, mat_type in enumerate(['ispc', 'pli']):

            fig, axs = plt.subplots(nrows=len(freq_band_dict_FC[band_prep].keys()))
            times = dict_allband[band]['times'].values

            for band_i, band in enumerate(freq_band_dict_FC[band_prep].keys()):

                ax = axs[band_i]

                ax.plot(times, dict_allband[band].loc[mat_type, pair, :])
                ax.vlines(0, ymin=dict_allband[band].loc[mat_type, pair, :].values.min(), ymax=dict_allband[band].loc[mat_type, pair, :].values.max(), color='r')

                if band_i == 0:
                    ax.set_title(f'{pair} {mat_type}')
                
                ax.set_ylabel(band)

            #plt.show()

            fig.set_figheight(15)
            fig.set_figwidth(15)
            fig.savefig(f'{sujet}_{cond}_{mat_type}_{pair}.png')

            plt.close()












################################
######## EXECUTE ########
################################



if __name__ == '__main__':


    #### save fig
    save_fig_for_allsession_FR_CV(sujet)
    save_fig_dfc_AL(sujet)

    for cond in ['AC', 'SNIFF']:
        save_fig_dfc_AC_SNIFF(sujet, cond, 'hf')



