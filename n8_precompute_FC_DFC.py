



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




#data_fc, pairs, roi_in_data = res_pairs_fc[0,:], pairs_to_compute_anat, roi_in_data
def from_fc_to_mat(data_fc, pairs, roi_in_data):

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
            
            x = data_fc[pairs == pair_to_find]
            x_rev = data_fc[pairs == pair_to_find_rev]

            val_to_place = np.hstack([x, x_rev]).mean(axis=0)

            mat_cf[x_i, y_i] = val_to_place

    if debug:
        plt.matshow(mat_cf)
        plt.show()

    return mat_cf









########################################
######## CHUNK & RESAMPLE ######## 
########################################


def chunk_data_AC(x, ac_starts, prms):

    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

    x_chunk = np.zeros((len(ac_starts), int(stretch_point_TF_ac)))

    for start_i, start_time in enumerate(ac_starts):

        t_start = int(start_time + t_start_AC*prms['srate'])
        t_stop = int(start_time + t_stop_AC*prms['srate'])

        x_chunk[start_i,:] = x[t_start: t_stop]

    x_chunk_mean = np.mean(x_chunk, axis=0)
    
    return x_chunk_mean




def chunk_data_sniff(x, sniff_starts, prms):

    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])

    x_chunk = np.zeros((len(sniff_starts), int(stretch_point_TF_sniff)))

    for start_i, start_time in enumerate(sniff_starts):

        t_start = int(start_time + t_start_SNIFF*prms['srate'])
        t_stop = int(start_time + t_stop_SNIFF*prms['srate'])

        x_chunk[start_i,:] = x[t_start: t_stop]

    x_chunk_mean = np.mean(x_chunk, axis=0)
    
    return x_chunk_mean











########################################
######## PLI ISPC DFC FC ######## 
########################################

#cond, band_prep, band, freq, trial_i = 'FR_CV', 'hf', 'l_gamma', [50, 80], 0
def get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i):

    #### load data
    data = load_data_sujet(sujet, band_prep, cond, trial_i)
    
    #### get params
    prms = get_params(sujet)

    wavelets, nfrex = get_wavelets(sujet, band_prep, freq)

    #### initiate res
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

    #### generate fake convolutions
    # convolutions = np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) * 1j
    # convolutions += np.random.random(len(prms['chan_list_ieeg']) * nfrex * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex, data.shape[1]) 

    print('CONV')

    #nchan = 0
    def convolution_x_wavelets_nchan(nchan_i, nchan):

        # print_advancement(nchan_i, len(prms['chan_list_ieeg']), steps=[25, 50, 75])
        
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
    df_loca = get_loca_df(sujet)
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
    for pair_A_i, pair_A in enumerate(prms['chan_list_ieeg']):
        for pair_B_i, pair_B in enumerate(prms['chan_list_ieeg']):
            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue
            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    pairs_to_compute_anat = []
    for pair_i in pairs_to_compute:
        plot_A, plot_B = pair_i.split('-')
        anat_A, anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(plot_A)], df_loca['ROI'][prms['chan_list_ieeg'].index(plot_B)]
        pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    pairs_to_compute_anat = np.array(pairs_to_compute_anat)

    #### initiate results
    res_FC_DFC = {}

    #### generate stretch point
    if cond == 'AC':
        stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

    if cond == 'SNIFF':
        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])

    ######## FC ########
    #### compute
    if cond not in ['AC', 'SNIFF']:

        print('COMPUTE FC')   

        #pair_to_compute_i, pair_to_compute_name = 0, pairs_to_compute[0]
        def compute_ispc_wpli_fc(pair_to_compute_i, pair_to_compute_name):

            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

            pair_A, pair_B = pair_to_compute_name.split('-')[0], pair_to_compute_name.split('-')[-1]
            pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)
                    
            as1 = convolutions[pair_A_i,:,:]
            as2 = convolutions[pair_B_i,:,:]

            # collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
            
            # compute ISPC and PLI (and average over trials!)
            _ispc_dfc_i = np.abs(np.mean(cdd))
            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            _wpli_dfc_i = np.abs( np.mean(np.imag(cdd)) ) / np.mean(np.abs(np.imag(cdd)))

            return _ispc_dfc_i, _wpli_dfc_i

        compute_ispc_wpli_fc_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_fc)(pair_to_compute_i, pair_to_compute_name) for pair_to_compute_i, pair_to_compute_name in enumerate(pairs_to_compute))

        #### compute metrics
        res_pairs_fc = np.zeros((2, len(pairs_to_compute)))

        #### load in mat    
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
                    
            res_pairs_fc[0, pair_to_compute_i] = compute_ispc_wpli_fc_res[pair_to_compute_i][0]
            res_pairs_fc[1, pair_to_compute_i] = compute_ispc_wpli_fc_res[pair_to_compute_i][1]

        #### transform in mat
        mat_fc = np.zeros((2, len(roi_in_data), len(roi_in_data)))
        mat_fc[0,:,:] = from_fc_to_mat(res_pairs_fc[0,:], pairs_to_compute_anat, roi_in_data)
        mat_fc[1,:,:] = from_fc_to_mat(res_pairs_fc[1,:], pairs_to_compute_anat, roi_in_data)

        res_FC_DFC['FC'] = mat_fc

    else:
        #### fill with empty if cond != 'FR_CV'
        res_FC_DFC['FC'] = np.array([])

    ######## DFC ########

    print('COMPUTE DFC')

    if band in band_name_fc_dfc and cond != 'FR_CV':

        #### identify slwin
        slwin_len = slwin_dict[band]    # in sec
        slwin_step = slwin_len*slwin_step_coeff  # in sec
        times_conv = np.arange(0, convolutions.shape[-1]/prms['srate'], 1/prms['srate'])
        win_sample = frites.conn.define_windows(times_conv, slwin_len=slwin_len, slwin_step=slwin_step)[0]

        print('COMPUTE')   

        #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
        def compute_ispc_wpli(pair_to_compute_i, pair_to_compute):

            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

            pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
            pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

            ispc_dfc_i = np.zeros(( len(win_sample) ))
            wpli_dfc_i = np.zeros(( len(win_sample) ))

            #slwin_values_i, slwin_values = 0, win_sample[0]
            for slwin_values_i, slwin_values in enumerate(win_sample):

                # print_advancement(slwin_values_i, len(win_sample), steps=[25, 50, 75])
                    
                as1 = convolutions[pair_A_i,:,slwin_values[0]:slwin_values[-1]]
                as2 = convolutions[pair_B_i,:,slwin_values[0]:slwin_values[-1]]

                ##### collect "eulerized" phase angle differences
                cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                
                ##### compute ISPC and WPLI (and average over trials!)
                ispc_dfc_i[slwin_values_i] = np.abs(np.mean(cdd))
                # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
                wpli_dfc_i[slwin_values_i] = np.abs( np.mean( np.imag(cdd) ) ) / np.mean( np.abs( np.imag(cdd) ) )

            return ispc_dfc_i, wpli_dfc_i

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

        #### extract
        wpli_mat = np.zeros((len(pairs_to_compute),win_sample.shape[0]))
        ispc_mat = np.zeros((len(pairs_to_compute),win_sample.shape[0]))

        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):

            ispc_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][0]
            wpli_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][1]

        #### simulate mat data
        # wpli_mat = np.random.random(len(pairs_to_compute) * win_sample.shape[0]).reshape(len(pairs_to_compute), win_sample.shape[0])
        # ispc_mat = np.random.random(len(pairs_to_compute) * win_sample.shape[0]).reshape(len(pairs_to_compute), win_sample.shape[0])

        #### resample for stretch
        os.chdir(path_memmap)
        matrix_resampled = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_fc_mat_resample.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), convolutions.shape[-1]))

        ispc_mat_i, wpli_mat_i = 0, 1

        up_npnts = times_conv.shape[0]
        half_sig_up = int(up_npnts/2)
        half_sig_down = int(win_sample.shape[0]/2)

        def resample_sig(sig):
            #### chunk 2 times for more precision
            x_a = sig[:half_sig_down]
            x_a_resampled = scipy.signal.resample(x_a, half_sig_up)
            x_b = sig[half_sig_down:]
            x_b_resampled = scipy.signal.resample(x_b, times_conv.shape[0] - half_sig_up)
            sig_resampled = np.append(x_a_resampled, x_b_resampled)

            if debug:
                times_up = np.arange(0, convolutions.shape[-1]/prms['srate'], 1/prms['srate'])
                times_down = np.linspace(0, convolutions.shape[-1]/prms['srate'], len(win_sample))
                plt.plot(times_up, scipy.signal.resample(sig,up_npnts), label='resample')
                plt.plot(times_up, sig_resampled, label='resample_2')
                plt.plot(times_down, ispc_mat[0], label='original')
                plt.legend()
                plt.show()

            return sig_resampled

        print('RESAMPLE')

        #pair_i = 0
        for pair_i, _ in enumerate(pairs_to_compute):

            # print_advancement(pair_i, len(pairs_to_compute), steps=[25, 50, 75])

            matrix_resampled[ispc_mat_i, pair_i,:] = resample_sig(ispc_mat[pair_i,:])
            matrix_resampled[wpli_mat_i, pair_i,:] = resample_sig(wpli_mat[pair_i,:])

        #### free memory
        del compute_ispc_pli_res, ispc_mat, wpli_mat

        #### simulate resampled data
        #matrix_resampled = np.random.random(2 * len(pairs_to_compute) * convolutions.shape[-1]).reshape(2, len(pairs_to_compute), convolutions.shape[-1])

        #### stretch
        if cond == 'FR_CV':
            resp_features = load_respfeatures(sujet)[cond][trial_i]
            mat_dfc_stretch = np.zeros(( 2, len(pairs_to_compute), stretch_point_TF ))
        if cond == 'AC':
            mat_dfc_stretch = np.zeros(( 2, len(pairs_to_compute), stretch_point_TF_ac ))
        if cond == 'SNIFF':
            mat_dfc_stretch = np.zeros(( 2, len(pairs_to_compute), stretch_point_TF_sniff ))
        if cond == 'AL':
            mat_dfc_stretch = np.zeros(( 2, len(pairs_to_compute), n_points_AL_interpolation ))

        #pair_i = 0
        for pair_i, _ in enumerate(pairs_to_compute):

            print_advancement(pair_i, len(pairs_to_compute), steps=[25, 50, 75])

            #metric_i = 0
            for metric_i, _ in enumerate(['ispc_mat_i', 'wpli_mat_i']):
                
                x = matrix_resampled[metric_i,pair_i,:]
                
                if cond == 'FR_CV':
                    x_stretch, _ = stretch_data(resp_features, stretch_point_TF, x, prms['srate'])
                    mat_dfc_stretch[metric_i,pair_i,:] = np.mean(x_stretch, axis=0)
                if cond == 'AC':
                    ac_starts = get_ac_starts(sujet)
                    mat_dfc_stretch[metric_i,pair_i,:] = chunk_data_AC(x, ac_starts, prms)
                if cond == 'SNIFF':
                    sniff_starts = get_sniff_starts(sujet)
                    mat_dfc_stretch[metric_i,pair_i,:] = chunk_data_sniff(x, sniff_starts, prms)
                if cond == 'AL':
                    x_stretch = scipy.signal.resample(x, n_points_AL_interpolation)
                    mat_dfc_stretch[metric_i,pair_i,:] = x_stretch

        if debug:
            x = (mat_dfc_stretch[ispc_mat_i,0,:] - mat_dfc_stretch[ispc_mat_i,0,:].mean()) / mat_dfc_stretch[ispc_mat_i,0,:].std()
            y = (mat_dfc_stretch[wpli_mat_i,0,:] - mat_dfc_stretch[wpli_mat_i,0,:].mean()) / mat_dfc_stretch[wpli_mat_i,0,:].std()
            plt.plot(x, label='ispc')
            plt.plot(y, label='wpli')
            plt.legend()
            plt.show()

        #### remove conv
        os.chdir(path_memmap)
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_fc_convolutions.dat')
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{trial_i}_fc_mat_resample.dat')

        #### reduce mat
        mat_dfc_mean = np.zeros(( 2, len(roi_in_data), len(roi_in_data) ))

        mat_dfc_mean[ispc_mat_i,:,:] = from_dfc_to_mat_conn_mean(mat_dfc_stretch[ispc_mat_i, :, :], pairs_to_compute_anat, roi_in_data)
        mat_dfc_mean[wpli_mat_i,:,:] = from_dfc_to_mat_conn_mean(mat_dfc_stretch[wpli_mat_i, :, :], pairs_to_compute_anat, roi_in_data)

        # mat_dfc_mean[ispc_mat_i,:,:] = from_dfc_to_mat_conn_trapz(mat_dfc_stretch[ispc_mat_i, :, :], pairs_to_compute_anat, roi_in_data)
        # mat_dfc_mean[wpli_mat_i,:,:] = from_dfc_to_mat_conn_trapz(mat_dfc_stretch[wpli_mat_i, :, :], pairs_to_compute_anat, roi_in_data)

        # mat_df_sorted = sort_mat(mat_stretch[ispc_mat_i, :, :], index_sorted)
        # mat_dfc_mean[ispc_mat_i,:,:] = reduce_functionnal_mat(mat_df_sorted, df_sorted)

        #### simulate data
        # mat_dfc_stretch = np.random.random(mat_dfc_stretch.shape[0] * mat_dfc_stretch.shape[1] * mat_dfc_stretch.shape[2]).reshape(mat_dfc_stretch.shape[0], mat_dfc_stretch.shape[1], mat_dfc_stretch.shape[2]) 
        # mat_dfc_mean = np.random.random(mat_dfc_mean.shape[0] * mat_dfc_mean.shape[1] * mat_dfc_mean.shape[2]).reshape(mat_dfc_mean.shape[0], mat_dfc_mean.shape[1], mat_dfc_mean.shape[2]) 

        res_FC_DFC['DFC'] = [mat_dfc_stretch, mat_dfc_mean]

    else:

        res_FC_DFC['DFC'] = [np.array([]), np.array([])]

    return res_FC_DFC








def get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq):

    if band in band_name_fc_dfc and cond != 'FR_CV':

        if os.path.exists(os.path.join(path_precompute, sujet, 'DFC', f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')):
            print(f'ALREADY DONE DFC {cond} {band}')

    if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')):
        print(f'ALREADY DONE FC {cond} {band}')
        return

    #### get n trial for cond
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if cond != 'SNIFF':
        n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1])
    else:
        n_trials = len([i for i in os.listdir() if i.find(f'{cond}') != -1 and i.find('hf') != -1 and i.find('session') != -1 ])

    #### identify anat info
    prms = get_params(sujet)
    df_loca = get_loca_df(sujet)

    pairs_to_compute = []
    for pair_A in prms['chan_list_ieeg']:
        for pair_B in prms['chan_list_ieeg']:
            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue
            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    pairs_to_compute_anat = []
    for pair_i in pairs_to_compute:
        plot_A, plot_B = pair_i.split('-')
        anat_A, anat_B = df_loca['ROI'][prms['chan_list_ieeg'].index(plot_A)], df_loca['ROI'][prms['chan_list_ieeg'].index(plot_B)]
        pairs_to_compute_anat.append(f'{anat_A}-{anat_B}')

    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    roi_in_data = df_sorted['ROI'].unique()

    #### compute
    mat_stretch = []
    mat_dfc = []
    mat_fc = []

    #### for dfc computation
    if band in band_name_fc_dfc and cond != 'FR_CV':

        #trial_i = 0
        for trial_i in range(n_trials):
            #mat_dfc_stretch_i, mat_dfc_mean_i = mat_stretch, mat_dfc_mean
            res_fc_dfc = get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i)
            mat_fc.append(res_fc_dfc['FC'])
            mat_dfc_stretch_i, mat_dfc_mean_i = res_fc_dfc['DFC'][0], res_fc_dfc['DFC'][1]
            mat_stretch.append(mat_dfc_stretch_i)
            mat_dfc.append(mat_dfc_mean_i)

        #### mean across trials
        for trial_i in range(n_trials):
            if trial_i ==0:
                mat_stretch_mean = mat_stretch[trial_i]
                mat_dfc_mean = mat_dfc[trial_i]
                mat_fc_mean = mat_fc[trial_i]

            else:
                mat_stretch_mean += mat_stretch[trial_i]
                mat_dfc_mean += mat_dfc[trial_i]
                mat_fc_mean += mat_fc[trial_i]

        mat_stretch_mean /= n_trials
        mat_dfc_mean /= n_trials
        mat_fc_mean /= n_trials

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
        if mat_dfc_mean.shape[0] != 0:
            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
            dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute_anat, 'times' : time_vec}
            xr_export = xr.DataArray(mat_stretch_mean, coords=dict_xr.values(), dims=dict_xr.keys())
            xr_export.to_netcdf(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_allpairs.nc')

        #### save reduced pairs
        if mat_dfc_mean.shape[0] != 0:
            os.chdir(os.path.join(path_precompute, sujet, 'DFC'))
            dict_xr = {'mat_type' : ['ispc', 'wpli'], 'x' : roi_in_data, 'y' : roi_in_data}
            xr_export = xr.DataArray(mat_dfc_mean, coords=dict_xr.values(), dims=dict_xr.keys())
            xr_export.to_netcdf(f'{sujet}_DFC_wpli_ispc_{band}_{cond}_reducedpairs.nc')

        #### save FC pairs
        if mat_fc_mean.shape[0] != 0:
            os.chdir(os.path.join(path_precompute, sujet, 'FC'))
            dict_xr = {'mat_type' : ['ispc', 'wpli'], 'x' : roi_in_data, 'y' : roi_in_data}
            xr_export = xr.DataArray(mat_fc_mean, coords=dict_xr.values(), dims=dict_xr.keys())
            xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')

    #### for FC only
    elif cond not in ['AC', 'SNIFF']:
        #trial_i = 0
        for trial_i in range(n_trials):
            #mat_dfc_stretch_i, mat_dfc_mean_i = mat_stretch, mat_dfc_mean
            res_fc_dfc = get_pli_ispc_fc_dfc_trial(sujet, cond, band_prep, band, freq, trial_i)
            mat_fc.append(res_fc_dfc['FC'])

        #### mean across trials
        for trial_i in range(n_trials):
            if trial_i == 0:
                mat_fc_mean = mat_fc[trial_i]

            else:
                mat_fc_mean += mat_fc[trial_i]

        mat_fc_mean /= n_trials

        #### save FC pairs
        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        dict_xr = {'mat_type' : ['ispc', 'wpli'], 'x' : roi_in_data, 'y' : roi_in_data}
        xr_export = xr.DataArray(mat_fc_mean, coords=dict_xr.values(), dims=dict_xr.keys())
        xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{band}_{cond}_reducedpairs.nc')





################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    prms = get_params(sujet)

    print('######## PRECOMPUTE DFC ########') 
    # cond = 'AC'
    for cond in cond_FC_DFC:
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            #band, freq = 'beta', [12,40]
            for band, freq in freq_band_dict_FC_function[band_prep].items():

                # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq)
                # execute_function_in_slurm_bash('n8_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq])
                execute_function_in_slurm_bash_mem_choice('n8_precompute_FC_DFC', 'get_wpli_ispc_fc_dfc', [sujet, cond, band_prep, band, freq], '15G')

    

