



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib

import frites


from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False




#######################################
############# ISPC & PLI #############
#######################################




#band_prep, freq, band, cond, prms = 'lf', [4, 8], 'theta', 'FR_CV', prms
def compute_fc_metrics_mat(band_prep, freq, band, cond, prms):

    
    #### check if already computed
    if cond == 'FR_CV':
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
    if cond == 'FR_CV':
        data = data[:len(prms['chan_list_ieeg']),:]
    if cond == 'AL':
        data = [i[:len(prms['chan_list_ieeg']),:] for i in data]

    #### compute all convolution
    if cond == 'AL':

        #data_AL_i, data_AL = 0, data[0]
        for data_AL_i, data_AL in enumerate(data):

            if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_DFC_pli_{band}_{cond}{data_AL_i+1}.nc')) and os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_DFC_ispc_{band}_{cond}{data_AL_i+1}.nc')):
                continue

            os.chdir(path_memmap)
            convolutions = np.memmap(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data_AL.shape[1]))

            print('CONV')

            #nchan = 0
            def convolution_x_wavelets_nchan(nchan):

                print_advancement(nchan, np.size(data_AL,0), steps=[25, 50, 75])
                
                nchan_conv = np.zeros((nfrex, np.size(data_AL,1)), dtype='complex')

                x = data_AL[nchan,:]

                for fi in range(nfrex):

                    nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                convolutions[nchan,:,:] = nchan_conv

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data_AL,0)))

            # convs = np.array(convolutions)
            
            #### identify roi in data
            df_loca = get_loca_df(sujet)
            df_sorted = df_loca.sort_values(['lobes', 'ROI'])
            index_sorted = df_sorted.index.values
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

            #### identify slwin
            times = np.linspace(0, data_AL.shape[-1]/prms['srate'], data_AL.shape[-1])
            slwin_len = slwin_dict[band]    # in sec
            slwin_step = slwin_len*slwin_step_coeff  # in sec
            win_sample = frites.conn.define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]
        

            print('COMPUTE')   

            def compute_ispc_pli(pair_to_compute_i, pair_to_compute):

                print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

                pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
                pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

                ispc_dfc_i = np.zeros(( nfrex, len(win_sample) ))
                pli_dfc_i = np.zeros(( nfrex, len(win_sample) ))

                #slwin_i = win_sample[0]
                for slwin_values_i, slwin_values in enumerate(win_sample):

                    for fi in range(nfrex):
                        
                        as1 = convolutions[pair_A_i][fi, slwin_values[0]:slwin_values[-1]]
                        as2 = convolutions[pair_B_i][fi, slwin_values[0]:slwin_values[-1]]

                        # collect "eulerized" phase angle differences
                        cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                        
                        # compute ISPC and PLI (and average over trials!)
                        ispc_dfc_i[fi,slwin_values_i] = np.abs(np.mean(cdd))
                        pli_dfc_i[fi,slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))

                # compute mean
                ispc_dfc_i = np.mean(ispc_dfc_i,0)
                pli_dfc_i = np.mean(pli_dfc_i,0)

                return ispc_dfc_i, pli_dfc_i

            compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))
                
            #### compute metrics
            pli_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))
            ispc_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))

            #### load in mat    
            for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
                        
                ispc_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][0]
                pli_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][1]

            #### supress mmap
            os.chdir(path_memmap)
            os.remove(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat')


            #### generate mat results
            mat_pli_time = np.zeros(( len(pairs_possible), pli_mat.shape[-1] ))
            mat_ispc_time = np.zeros(( len(pairs_possible), ispc_mat.shape[-1] ))

            #### fill mat
            name_modified = np.array([])
            count_pairs = np.zeros(( len(pairs_possible) ))
            for pair_i in pairs_to_compute:
                pair_A, pair_B = pair_i.split('-')
                pair_A_name, pair_B_name = df_loca['ROI'][df_loca['name'] == pair_A].values[0], df_loca['ROI'][df_loca['name'] == pair_B].values[0]
                pair_name_i = f'{pair_A_name}-{pair_B_name}'
                name_modified = np.append(name_modified, pair_name_i)
            
            for pair_name_i, pair_name in enumerate(pairs_possible):
                pair_name_inv = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
                mask = (name_modified == pair_name) | (name_modified == pair_name_inv)
                count_pairs[pair_name_i] = int(np.sum(mask))
                mat_pli_time[pair_name_i,:] = np.mean(pli_mat[mask,:], axis=0)
                mat_ispc_time[pair_name_i,:] = np.mean(ispc_mat[mask,:], axis=0)


            #### save
            times = np.linspace(0, data_AL.shape[-1]/prms['srate'], len(win_sample))

            os.chdir(os.path.join(path_precompute, sujet, 'FC'))
            dict_xr_pli = {'pairs' : pairs_possible, 'times' : times}
            xr_pli = xr.DataArray(mat_pli_time, coords=dict_xr_pli.values(), dims=dict_xr_pli.keys())
            xr_pli.to_netcdf(f'{sujet}_DFC_pli_{band}_{cond}{data_AL_i+1}.nc')

            dict_xr_ispc = {'pairs' : pairs_possible, 'times' : times}
            xr_ispc = xr.DataArray(mat_ispc_time, coords=dict_xr_ispc.values(), dims=dict_xr_ispc.keys())
            xr_ispc.to_netcdf(f'{sujet}_DFC_ispc_{band}_{cond}{data_AL_i+1}.nc')


        return


    if cond == 'FR_CV':

        os.chdir(path_memmap)
        convolutions = np.memmap(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

        print('CONV')

        def convolution_x_wavelets_nchan(nchan):

            print_advancement(nchan, len(np.size(data,0)), steps=[25, 50, 75])
            
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

        #### compute metrics
        pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
        ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

        print('COMPUTE')

        for seed in range(np.size(data,0)):

            print_advancement(seed, len(np.size(data,0)), steps=[25, 50, 75])

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


    

    




def compute_pli_ispc_allband(sujet, cond):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band, freq = 'alpha', [8, 12]
        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                if cond == 'FR_CV':
                    
                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                    pli_allband[band] = pli_allcond
                    ispc_allband[band] = ispc_allcond

                if cond == 'AL':

                    compute_fc_metrics_mat(band_prep, freq, band, cond, prms)

                    continue

    if cond == 'AL':

        return

    #### verif mat
    if debug:
        for band in pli_allband.keys():
            for cond in pli_allband[band].keys():
                plt.matshow(pli_allband[band][cond][0])
                plt.show()
    #### verif

    if debug == True:
                
        for band, freq in freq_band_fc_analysis.items():

            for cond_i, cond in enumerate(conditions_FC) :

                print(band, cond, len(pli_allband[band][cond]))
                print(band, cond, len(ispc_allband[band][cond]))

    print('done')






################################
######## PLI ISPC DFC ######## 
################################

#cond, band_prep, band, freq = 'SNIFF', 'hf', 'l_gamma', [50, 80]
def get_pli_ispc_dfc(sujet, cond, band_prep, band, freq):

        data = load_data(cond, band_prep='hf')

        if cond == 'SNIFF':
            epochs_starts = get_sniff_starts(sujet)

        if cond == 'AC':
            epochs_starts = get_ac_starts(sujet)
        
        prms = get_params(sujet)

        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_DFC_pli_ispc_{band}_{cond}.nc')):
            print('ALREADY DONE')
            return

        wavelets, nfrex = get_wavelets(band_prep, freq)

        os.chdir(path_memmap)
        convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

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

        #### epoch convolutions
        if cond == 'SNIFF':
            t_start_epoch, t_stop_epoch = t_start_SNIFF, t_stop_SNIFF
        if cond == 'AC':
            t_start_epoch, t_stop_epoch = t_start_AC, t_stop_AC

        #### generate matrix epoch
        os.chdir(path_memmap)
        stretch_point_TF_epoch = int(np.abs(t_start_epoch)*prms['srate'] +  t_stop_epoch*prms['srate'])
        epochs = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_fc_epochs.dat', dtype=np.complex128, mode='w+', shape=( len(prms['chan_list_ieeg']), len(epochs_starts), nfrex, stretch_point_TF_epoch ))

        def chunk_epochs_in_signal(nchan_i, nchan):
            
            for epoch_i, epoch_time in enumerate(epochs_starts):

                _t_start = epoch_time + int(t_start_epoch*prms['srate']) 
                _t_stop = epoch_time + int(t_stop_epoch*prms['srate'])

                epochs[nchan_i, epoch_i, :, :] = convolutions[nchan_i, :, _t_start:_t_stop]

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_epochs_in_signal)(nchan_i, nchan) for nchan_i, nchan in enumerate(prms['chan_list_ieeg']))
        
        #### identify roi in data
        df_loca = get_loca_df(sujet)
        df_sorted = df_loca.sort_values(['lobes', 'ROI'])
        index_sorted = df_sorted.index.values
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

        #### identify slwin
        slwin_len = slwin_dict[band]    # in sec
        slwin_step = slwin_len*slwin_step_coeff  # in sec
        times_epoch = np.arange(t_start_epoch, t_stop_epoch, 1/prms['srate'])
        win_sample = frites.conn.define_windows(times_epoch, slwin_len=slwin_len, slwin_step=slwin_step)[0]
        times = np.linspace(t_start_epoch, t_stop_epoch, len(win_sample))

        print('COMPUTE')   

        #pair_to_compute = pairs_to_compute[0]
        def compute_ispc_pli(pair_to_compute_i, pair_to_compute):

            # print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

            pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
            pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

            ispc_dfc_i = np.zeros(( len(win_sample) ))
            pli_dfc_i = np.zeros(( len(win_sample) ))

            #slwin_values_i, slwin_values = 0, win_sample[0]
            for slwin_values_i, slwin_values in enumerate(win_sample):
                    
                as1 = epochs[pair_A_i, :, :, slwin_values[0]:slwin_values[-1]]
                as2 = epochs[pair_B_i, :, :, slwin_values[0]:slwin_values[-1]]

                # collect "eulerized" phase angle differences
                cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                
                # compute ISPC and PLI (and average over trials!)
                ispc_dfc_i[slwin_values_i] = np.abs(np.mean(cdd))
                pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
                # pli_dfc_i[slwin_values_i] = np.mean( np.abs(np.imag(cdd))*np.sign(np.imag(cdd)) ) / np.mean(np.abs(np.imag(cdd)))

            return ispc_dfc_i, pli_dfc_i

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))
        
        #### compute metrics
        pli_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))
        ispc_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))

        #### load in mat    
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
                    
            ispc_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][0]
            pli_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][1]

        #### free memory
        del compute_ispc_pli_res

        #### remove conv
        os.chdir(path_memmap)
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_fc_convolutions.dat')
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_fc_epochs.dat')
        
        #### generate mat results
        mat_pli_time = np.zeros(( len(pairs_possible), pli_mat.shape[-1] ))
        mat_ispc_time = np.zeros(( len(pairs_possible), ispc_mat.shape[-1] ))

        #### fill mat
        name_modified = np.array([])
        count_pairs = np.zeros(( len(pairs_possible) ))
        for pair_i in pairs_to_compute:
            pair_A, pair_B = pair_i.split('-')
            pair_A_name, pair_B_name = df_loca['ROI'][df_loca['name'] == pair_A].values[0], df_loca['ROI'][df_loca['name'] == pair_B].values[0]
            pair_name_i = f'{pair_A_name}-{pair_B_name}'
            name_modified = np.append(name_modified, pair_name_i)
        
        for pair_name_i, pair_name in enumerate(pairs_possible):
            pair_name_inv = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
            mask = (name_modified == pair_name) | (name_modified == pair_name_inv)
            count_pairs[pair_name_i] = int(np.sum(mask))
            mat_pli_time[pair_name_i,:] = np.mean(pli_mat[mask,:], axis=0)
            mat_ispc_time[pair_name_i,:] = np.mean(ispc_mat[mask,:], axis=0)

        #### save
        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        dict_xr_pli = {'mat_type' : ['pli', 'ispc'], 'pairs' : pairs_possible, 'times' : times}
        data_export = np.concatenate( [mat_pli_time.reshape(1, mat_pli_time.shape[0], mat_pli_time.shape[1]), 
                                        mat_ispc_time.reshape(1, mat_ispc_time.shape[0], mat_ispc_time.shape[1])], axis=0 )
        xr_export = xr.DataArray(data_export, coords=dict_xr_pli.values(), dims=dict_xr_pli.keys())
        xr_export.to_netcdf(f'{sujet}_DFC_pli_ispc_{band}_{cond}.nc')












def precompute_ispc_pli_DFC(sujet, cond, band_prep, band, freq):

    get_pli_ispc_dfc(sujet, cond, band_prep, band, freq)
    print('done')







################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #cond = 'AL'
    for cond in conditions_FC:
        #compute_pli_ispc_allband(sujet, cond)
        execute_function_in_slurm_bash('n7_precompute_fc', 'compute_pli_ispc_allband', [sujet, cond])

    band_prep = 'hf'
    #cond = 'SNIFF'
    for cond in ['AC', 'SNIFF']:
        #band, freq = 'h_gamma', [80, 120]
        for band, freq in freq_band_dict_FC_function[band_prep].items():
            #precompute_ispc_pli_DFC(sujet, cond)
            #execute_function_in_slurm_bash('n7_precompute_fc', 'precompute_ispc_pli_DFC', [sujet, cond, band_prep, band, freq])
            execute_function_in_slurm_bash_mem_choice('n7_precompute_fc', 'precompute_ispc_pli_DFC', [sujet, cond, band_prep, band, freq], '30G')

