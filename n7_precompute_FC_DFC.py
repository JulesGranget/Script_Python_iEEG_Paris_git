



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

import frites

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





################################
######## MAT REDUCTION ########
################################



def reduce_functionnal_mat(mat, df_sorted):

    chan_name_sorted = df_sorted['ROI'].values.tolist()

    #### which roi in data
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

    #### compute index
    pairs_possible = []
    for pair_A_i, pair_A in enumerate(roi_in_data):
        for pair_B_i, pair_B in enumerate(roi_in_data[pair_A_i:]):
            if pair_A == pair_B:
                continue
            pairs_possible.append(f'{pair_A}-{pair_B}')
            
    indexes_to_compute = {}
    for pair_i, pair_name in enumerate(pairs_possible):    
        pair_A, pair_B = pair_name.split('-')[0], pair_name.split('-')[-1]
        x_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == pair_A]
        y_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == pair_B]
        
        coord = []
        for x_i in x_to_mean:
            for y_i in y_to_mean:
                if x_i == y_i:
                    continue
                else:
                    coord.append([x_i, y_i])
                    coord.append([y_i, x_i])

        indexes_to_compute[pair_name] = coord

    #### reduce mat
    mat_reduced = np.zeros((len(roi_in_data), len(roi_in_data) ))

    for roi_i_x, roi_name_x in enumerate(roi_in_data):        
        for roi_i_y, roi_name_y in enumerate(roi_in_data):
            if roi_name_x == roi_name_y:
                continue
            else:
                pair_i = f'{roi_name_x}-{roi_name_y}'
                if (pair_i in indexes_to_compute) == False:
                    pair_i = f'{roi_name_y}-{roi_name_x}'
                pair_i_data = []
                for coord_i in indexes_to_compute[pair_i]:
                    pair_i_data.append(mat[coord_i[0],coord_i[-1]])
                mat_reduced[roi_i_x, roi_i_y] = np.mean(pair_i_data)

    #### verif
    if debug:
        mat_test = np.zeros(( len(chan_name_sorted), len(chan_name_sorted) )) 
        for roi_i, roi_name in enumerate(pairs_possible): 
            i_test = indexes_to_compute[roi_name]
            for pixel in i_test:
                mat_test[pixel[0],pixel[-1]] = 1

        plt.matshow(mat_test)
        plt.show()

    return mat_reduced



#data_dfc, pairs, roi_in_data = mat_dfc_stretch[ispc_mat_i, :, :], pairs_to_compute_anat, roi_in_data
def from_dfc_to_mat_conn_trapz(data_dfc, pairs, roi_in_data):

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









#data_dfc, pairs, roi_in_data = mat_dfc_stretch[ispc_mat_i, :, :], pairs_to_compute_anat, roi_in_data
def from_dfc_to_mat_conn_mean(data_dfc, pairs, roi_in_data):

    #### fill mat
    mat_cf = np.zeros(( len(roi_in_data), len(roi_in_data) ))

    #### transform in array for indexing
    pairs = np.array(pairs)

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
            val_to_place = np.mean(x_mean)

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf




#pairs, roi_in_data = pairs_to_compute_anat, roi_in_data
def from_fc_to_mat(data_fc, pairs, roi_in_data):
    
    #### fill mat
    mat_cf = np.zeros(( 2, roi_in_data.shape[0], roi_in_data.shape[0] ))
    #metric_i = 0
    for metric_i in range(2):
        #x_i, x_name = 0, roi_in_data[0]
        for x_i, x_name in enumerate(roi_in_data):
            #y_i, y_name = 2, roi_in_data[2]
            for y_i, y_name in enumerate(roi_in_data):
                if x_name == y_name:
                    continue

                pair_to_find = f'{x_name}-{y_name}'
                pair_to_find_rev = f'{y_name}-{x_name}'
                
                x = data_fc[metric_i, pairs == pair_to_find, :]
                x_rev = data_fc[metric_i, pairs == pair_to_find_rev, :]

                x_mean = np.vstack([x, x_rev]).mean(axis=0)

                mat_cf[metric_i, x_i, y_i] = x_mean

    return mat_cf









########################################
######## CHUNK & RESAMPLE ######## 
########################################

#x = dfc_resample_ispc_i
def chunk_data_AC(x, ac_starts, nfrex, prms):

    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

    x_chunk = np.zeros((len(ac_starts), nfrex, int(stretch_point_TF_ac)))

    for start_i, start_time in enumerate(ac_starts):

        t_start = int(start_time + t_start_AC*prms['srate'])
        t_stop = int(start_time + t_stop_AC*prms['srate'])

        x_chunk[start_i,:,:] = x[:, t_start:t_stop]
    
    return x_chunk




def chunk_data_sniff(x, sniff_starts, nfrex, prms):

    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])

    x_chunk = np.zeros((len(sniff_starts), nfrex, int(stretch_point_TF_sniff)))

    for start_i, start_time in enumerate(sniff_starts):

        t_start = int(start_time + t_start_SNIFF*prms['srate'])
        t_stop = int(start_time + t_stop_SNIFF*prms['srate'])

        x_chunk[start_i,:,:] = x[:, t_start:t_stop]
    
    return x_chunk











########################################
######## PLI ISPC DFC FC ######## 
########################################

#cond, band_prep, band, freq, trial_i = 'FR_CV', 'hf', 'l_gamma', [50, 80], 0
def get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i, electrode_recording_type):

    #### load data
    data = load_data(sujet, cond, electrode_recording_type, band_prep=band_prep)

    if cond == 'AL':
        data = data[trial_i]

    data_length = data.shape[-1]
    
    #### get params
    prms = get_params(sujet, electrode_recording_type)

    wavelets, nfrex = get_wavelets(sujet, band_prep, freq, electrode_recording_type)

    #### initiate res
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data_length))

    #### generate fake convolutions
    # convolutions = np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) * 1j
    # convolutions += np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) 

    # convolutions = np.zeros((len(prms['chan_list_ieeg']), nfrex, data.shape[1])) 

    print('CONV')

    #nchan = 0
    def convolution_x_wavelets_nchan(nchan_i, nchan):

        print_advancement(nchan_i, len(prms['chan_list_ieeg']), steps=[25, 50, 75])
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan_i,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan_i,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(prms['chan_list_ieeg']))

    #### free memory
    del data        

    #### verif conv
    if debug:
        plt.plot(convolutions[0,0,:])
        plt.show()

    #### identify roi in data
    df_loca = get_loca_df(sujet, electrode_recording_type)
    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    roi_in_data = df_sorted['ROI'].unique()

    #### compute index
    pairs_possible = []
    for pair_A_i, pair_A in enumerate(roi_in_data):
        for pair_B_i, pair_B in enumerate(roi_in_data[pair_A_i:]):
            if pair_A == pair_B:
                continue
            pairs_possible.append(f'{pair_A}-{pair_B}')

    pairs_to_compute = []
    pairs_to_compute_anat = []
    for pair_A in prms['chan_list_ieeg']:

        anat_A = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_A)]
        if electrode_recording_type == 'bipolaire':
            pair_A = f"{pair_A.split('-')[0]}|{pair_A.split('-')[1]}"
        
        for pair_B in prms['chan_list_ieeg']:

            anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_B)]
            if electrode_recording_type == 'bipolaire':
                pair_B = f"{pair_B.split('-')[0]}|{pair_B.split('-')[1]}"

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
            pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    pairs_to_compute_anat = np.array(pairs_to_compute_anat)

    ######## DFC ########

    #### generate stretch point
    if cond == 'AC':
        stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

    if cond == 'SNIFF':
        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])

    print('COMPUTE DFC')

    #### identify slwin
    slwin_len = slwin_dict[band]    # in sec
    slwin_step = slwin_len*slwin_step_coeff  # in sec
    times_conv = np.arange(0, data_length/prms['srate'], 1/prms['srate'])
    win_sample = frites.conn.define_windows(times_conv, slwin_len=slwin_len, slwin_step=slwin_step)[0]

    os.chdir(path_memmap)
    dfc_metrics = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_metrics.dat', dtype=np.float64, mode='w+', shape=(2, len(pairs_to_compute), nfrex, win_sample.shape[0]))

    print('COMPUTE')   

    #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
    def compute_ispc_wpli(pair_to_compute_i, pair_to_compute):

        print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

        pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
        pair_A, pair_B = pair_A.replace('|', '-'), pair_B.replace('|', '-')
        pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

        ispc_dfc_mat = np.zeros(( nfrex, len(win_sample) ))
        wpli_dfc_mat = np.zeros(( nfrex, len(win_sample) ))

        #slwin_values_i, slwin_values = 0, win_sample[0]
        for slwin_values_i, slwin_values in enumerate(win_sample):

            # print_advancement(slwin_values_i, len(win_sample), steps=[25, 50, 75])
                
            as1 = convolutions[pair_A_i,:,slwin_values[0]:slwin_values[-1]]
            as2 = convolutions[pair_B_i,:,slwin_values[0]:slwin_values[-1]]

            ##### collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
            
            ##### compute ISPC and WPLI (and average over trials!)
            ispc_dfc_mat[:, slwin_values_i] = np.abs(np.mean(cdd, axis=1))
            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            wpli_dfc_mat[:, slwin_values_i] = np.abs( np.mean( np.imag(cdd), axis=1 ) ) / np.mean( np.abs( np.imag(cdd) ), axis=1 )

        dfc_metrics[0, pair_to_compute_i, :, :] = ispc_dfc_mat
        dfc_metrics[1, pair_to_compute_i, :, :] = wpli_dfc_mat

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

    #### verif
    if debug:
        source = os.getcwd()
        os.chdir(path_results)

        plt.pcolormesh(dfc_metrics[0, 0, :, :])
        plt.pcolormesh(dfc_metrics[1, 0, :, :])

        #plt.show()

        plt.savefig('test.png')
        plt.close('all')

        os.chdir(source)

    #### remove memmap
    os.chdir(path_memmap)
    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_convolutions.dat')
        del convolutions
    except:
        pass

    #### simulate mat data
    # dfc_metrics = np.random.random((2, len(pairs_to_compute), nfrex, win_sample.shape[0]))

    #### resample for stretch
    os.chdir(path_memmap)
    if cond == 'FR_CV':
        dfc_data_resample = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_mat_resample.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), nfrex, stretch_point_TF))
    if cond == 'AC':
        dfc_data_resample = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_mat_resample.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), nfrex, stretch_point_TF_ac))
    if cond == 'SNIFF':
        dfc_data_resample = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_mat_resample.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), nfrex, stretch_point_TF_sniff))    
    if cond == 'AL':
        dfc_data_resample = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_mat_resample.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), nfrex, n_points_AL_interpolation))    
    
    ispc_mat_i, wpli_mat_i = 0, 1

    print('RESAMPLE & STRETCH')

    #pair_to_compute_i = 0
    def resample_and_stretch_data_dfc(pair_to_compute_i, pair_to_compute):

        print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

        #### resample
        f = scipy.interpolate.interp1d(np.linspace(0, 1, win_sample.shape[0]), dfc_metrics[ispc_mat_i, pair_to_compute_i, :, :], kind='linear')
        dfc_resample_ispc_i = f(np.linspace(0, 1, data_length))

        f = scipy.interpolate.interp1d(np.linspace(0, 1, win_sample.shape[0]), dfc_metrics[wpli_mat_i, pair_to_compute_i, :, :], kind='linear')
        dfc_resample_wpli_i = f(np.linspace(0, 1, data_length))

        #### stretch
        if cond == 'FR_CV':
            respfeatures_allcond = load_respfeatures(sujet)

            dfc_resample_stretch_ispc_i = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, dfc_resample_ispc_i, prms['srate'])[0]
            dfc_resample_stretch_wpli_i = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, dfc_resample_wpli_i, prms['srate'])[0]

        if cond == 'AC':
            ac_starts = get_ac_starts(sujet)
            dfc_resample_stretch_ispc_i = chunk_data_AC(dfc_resample_ispc_i, ac_starts, nfrex, prms)
            dfc_resample_stretch_wpli_i = chunk_data_AC(dfc_resample_wpli_i, ac_starts, nfrex, prms)

        if cond == 'SNIFF':
            sniff_starts = get_sniff_starts(sujet)
            dfc_resample_stretch_ispc_i = chunk_data_sniff(dfc_resample_ispc_i, sniff_starts, nfrex, prms)
            dfc_resample_stretch_wpli_i = chunk_data_sniff(dfc_resample_wpli_i, sniff_starts, nfrex, prms)

        if cond == 'AL':
            f_ispc_i = scipy.interpolate.interp1d(range(data_length), dfc_resample_ispc_i, kind='linear')
            dfc_resample_stretch_ispc_i = f_ispc_i(range(n_points_AL_interpolation))

            f_wpli_i = scipy.interpolate.interp1d(range(data_length), dfc_resample_wpli_i, kind='linear')
            dfc_resample_stretch_wpli_i = f_wpli_i(range(n_points_AL_interpolation))

        #### plot verification figures
        if pair_to_compute_i % 200 == 0:

            for cf_metric in ['ispc', 'wpli']:

                if cf_metric == 'ispc':
                    dfc_to_plot_i = dfc_resample_stretch_ispc_i
                if cf_metric == 'ispc':
                    dfc_to_plot_i = dfc_resample_stretch_wpli_i

                os.chdir(os.path.join(path_results, sujet, 'DFC', 'verif'))

                plt.figure(figsize=(15,15))
                plt.pcolormesh(dfc_to_plot_i.mean(axis=0))
                plt.title(f'{cond} {freq} {cf_metric} MEAN, pair : {pairs_to_compute_anat_i}')
                # plt.show()

                plt.savefig(f'pair{pair_to_compute_i}_{cf_metric}_{cond}_{band}_MEAN_tf.png')
                plt.close('all')

                plt.figure(figsize=(15,15))
                plt.pcolormesh(dfc_to_plot_i.std(axis=0))
                plt.title(f'{cond} {freq} {cf_metric} SD, pair : {pairs_to_compute_anat_i}')
                # plt.show()

                plt.savefig(f'pair{pair_to_compute_i}_{cf_metric}_{cond}_{band}_SD_tf.png')
                plt.close('all')

                wavelet_i = 0
                pairs_to_compute_anat_i = pairs_to_compute_anat[pair_to_compute_i]
                plt.figure(figsize=(15,15))
                plt.plot(dfc_to_plot_i.mean(axis=0)[wavelet_i,:], label='mean')
                plt.plot(dfc_to_plot_i.mean(axis=0)[wavelet_i,:] + dfc_to_plot_i.std(axis=0)[wavelet_i,:], color='r', label='1SD')
                plt.plot(dfc_to_plot_i.mean(axis=0)[wavelet_i,:] - dfc_to_plot_i.std(axis=0)[wavelet_i,:], color='r', label='1SD')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 10)]*dfc_to_plot_i.shape[-1], linestyle='--', color='g', label='10p')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 25)]*dfc_to_plot_i.shape[-1], linestyle='-.', color='g', label='25p')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 40)]*dfc_to_plot_i.shape[-1], linestyle=':', color='g', label='40p')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 60)]*dfc_to_plot_i.shape[-1], linestyle=':', color='g', label='60p')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 75)]*dfc_to_plot_i.shape[-1], linestyle='-.', color='g', label='75p')
                plt.plot([np.percentile(dfc_to_plot_i[:,wavelet_i,:], 90)]*dfc_to_plot_i.shape[-1], linestyle='--', color='g', label='90p')
                plt.title(f'{cond} {freq} {cf_metric} pair : {pairs_to_compute_anat_i}, wavelet : {wavelet_i}')
                plt.legend()        
                # plt.show()

                plt.savefig(f'pair{pair_to_compute_i}_{cf_metric}_{cond}_{band}_MEAN_sig.png')
                plt.close('all')

        #### mean
        dfc_data_resample[0, pair_to_compute_i, :, :] = dfc_resample_stretch_ispc_i.mean(axis=0)
        dfc_data_resample[1, pair_to_compute_i, :, :] = dfc_resample_stretch_wpli_i.mean(axis=0)

        del dfc_resample_stretch_ispc_i, dfc_resample_stretch_wpli_i

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(resample_and_stretch_data_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

    mat_dfc_stretch = dfc_data_resample.copy()

    #### simulate data
    # mat_dfc_stretch = np.random.random((2, len(pairs_to_compute), nfrex, stretch_point_TF_ac)) 

    #### remove memmap
    os.chdir(path_memmap)

    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_mat_resample.dat')
        del dfc_mat_resample
    except:
        pass

    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_{electrode_recording_type}_dfc_metrics.dat')
        del dfc_metrics
    except:
        pass

    return mat_dfc_stretch








def get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, electrode_recording_type):

    #### verif computation
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'DFC', f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')):
            print(f'ALREADY DONE DFC {cond} {band}')
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'DFC', f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')):
            print(f'ALREADY DONE DFC {cond} {band}')
            return

    # if electrode_recording_type == 'monopolaire':
    #     if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')):
    #         print(f'ALREADY DONE FC {cond} {band}')
    #         return
    # if electrode_recording_type == 'bipolaire':
    #     if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs_bi.nc')):
    #         print(f'ALREADY DONE FC {cond} {band}')
    #         return


    #### get n trial for cond
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if cond != 'SNIFF':
        if electrode_recording_type == 'monopolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1 and i.find('bi') == -1])
        if electrode_recording_type == 'bipolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1 and i.find('bi') != -1])
    else:
        if electrode_recording_type == 'monopolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1 and i.find('session') != -1  and i.find('bi') == -1])
        if electrode_recording_type == 'bipolaire':
            n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1 and i.find('session') != -1  and i.find('bi') != -1])

    #### identify anat info
    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)

    pairs_to_compute = []
    pairs_to_compute_anat = []
    for pair_A in prms['chan_list_ieeg']:

        anat_A = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_A)]
        if electrode_recording_type == 'bipolaire':
            pair_A = f"{pair_A.split('-')[0]}|{pair_A.split('-')[1]}"
        
        for pair_B in prms['chan_list_ieeg']:

            anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(pair_B)]
            if electrode_recording_type == 'bipolaire':
                pair_B = f"{pair_B.split('-')[0]}|{pair_B.split('-')[1]}"

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
            pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    #### compute
    mat_stretch = []

    #### simulate AL data
    # mat_stretch = [np.random.rand( 2, len(pairs_to_compute), n_points_AL_interpolation ), np.random.rand( 2, len(pairs_to_compute), n_points_AL_interpolation ), np.random.rand( 2, len(pairs_to_compute), n_points_AL_interpolation )]

    #### for dfc computation
    wavelets, nfrex = get_wavelets(sujet, band_prep, freq, electrode_recording_type)

    #trial_i = 0
    for trial_i in range(n_trials):
        #mat_dfc_stretch_i, mat_dfc_mean_i = mat_stretch, mat_dfc_mean
        mat_dfc_stretch_i = get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i, electrode_recording_type)
        mat_stretch.append(mat_dfc_stretch_i)

    #### simulate data
    if debug:
        for trial_i in range(n_trials):
            #mat_dfc_stretch_i, mat_dfc_mean_i = mat_stretch, mat_dfc_mean
            if cond == 'FR_CV':
                mat_dfc_stretch_i = np.random.rand(2, len(pairs_to_compute), nfrex, stretch_point_TF)
            if cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                mat_dfc_stretch_i = np.random.rand(2, len(pairs_to_compute), nfrex, stretch_point_TF_ac)
            if cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                mat_dfc_stretch_i = np.random.rand(2, len(pairs_to_compute), nfrex, stretch_point_TF_sniff)
            if cond == 'AL':
                mat_dfc_stretch_i = np.random.rand(2, len(pairs_to_compute), nfrex, n_points_AL_interpolation)
            mat_stretch.append(mat_dfc_stretch_i)
    

    if cond != 'AL':
        #### mean across trials
        for trial_i in range(n_trials):
            if trial_i ==0:
                mat_stretch_mean = mat_stretch[trial_i]

            else:
                mat_stretch_mean += mat_stretch[trial_i]

        mat_stretch_mean /= n_trials

    if cond == 'AL':

        if n_trials != 3:
            raise ValueError('!! WARNING NO 3 AL !!')
        
        mat_stretch_mean = np.stack((mat_stretch[0], mat_stretch[1], mat_stretch[2]))

    #### export
    #### generate time vec
    if cond == 'FR_CV':
        time_vec = np.arange(stretch_point_TF)

    if cond == 'AL':
        time_vec = np.arange(n_points_AL_interpolation)

    if cond == 'AC':
        stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
        time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

    if cond == 'SNIFF':
        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
        time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

    #### save allpairs
    os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
    
    if cond != 'AL':
        dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute_anat, 'nfrex' : range(nfrex), 'times' : time_vec}
    if cond == 'AL':
        dict_xr = {'AL_num' : [1,2,3], 'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute_anat, 'nfrex' : range(nfrex), 'times' : time_vec}
    
    xr_export = xr.DataArray(mat_stretch_mean, coords=dict_xr.values(), dims=dict_xr.keys())
    
    if electrode_recording_type == 'monopolaire':
        xr_export.to_netcdf(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')
    if electrode_recording_type == 'bipolaire':
        xr_export.to_netcdf(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs_bi.nc')

    








################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    # sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    # sujet = 'pat_03138_1601'
    # sujet = 'pat_03146_1608'
    # sujet = 'pat_03174_1634'

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        for sujet in sujet_list:

            prms = get_params(sujet, electrode_recording_type)

            print('######## PRECOMPUTE DFC ########') 
            #cond = 'AC'
            for cond in cond_FC_DFC:
                #band_prep = 'lf'
                for band_prep in band_prep_list:
                    #band, freq = 'beta', [12,40]
                    for band, freq in freq_band_dict_FC_function[band_prep].items():

                        if band in band_name_fc_dfc:

                            if cond == 'AL':

                                # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, electrode_recording_type)
                                execute_function_in_slurm_bash_mem_choice('n7_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq, electrode_recording_type], '45G')
                            
                            elif cond == 'AC':

                                # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, electrode_recording_type)
                                execute_function_in_slurm_bash_mem_choice('n7_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq, electrode_recording_type], '30G')
                            
                            else:

                                # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, electrode_recording_type)
                                execute_function_in_slurm_bash_mem_choice('n7_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq, electrode_recording_type], '20G')
                            
        

