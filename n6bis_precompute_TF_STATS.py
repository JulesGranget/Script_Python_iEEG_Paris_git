
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False








################################
######## STRETCH TF ########
################################


def compute_chunk_baselines_tf_dB(sujet, tf, cond_compute, band, srate, monopol):

    #### load baseline
    _band = band[:-2]
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    if monopol == 'monopolaire':
        baselines = np.load(f'{sujet}_{_band}_baselines.npy')
    else:
        baselines = np.load(f'{sujet}_{_band}_baselines_bi.npy')

    tf_db = np.zeros(tf.shape, dtype=tf.dtype)

    #### apply baseline
    os.chdir(path_memmap)
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            if debug:
                plt.plot(activity)
                plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
                plt.show()

            tf_db[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    #### random chunk
    if cond_compute == 'AC':
        t_stop = tf.shape[-1]-AC_length*srate
        len_erp = get_ac_starts(sujet).shape[0]
        tf_dw_length = srate_dw_stats*3*AC_length
        t_chunk_length = srate*3*AC_length
        
    if cond_compute == 'SNIFF':
        t_stop = tf.shape[-1]-SNIFF_length*srate
        len_erp = get_sniff_starts(sujet).shape[0]
        tf_dw_length = srate_dw_stats*2*SNIFF_length
        t_chunk_length = srate*2*SNIFF_length

    random_chunk_val = np.random.randint(0, high=t_stop, size=(len_erp))

    #### extract
    tf_chunk_baseline = np.zeros((tf.shape[0], len_erp, tf.shape[1], tf_dw_length))

    for nchan in range(tf.shape[0]):
    
        for chunk_i, chunk_time in enumerate(random_chunk_val):

            x_pre = tf_db[nchan,:,chunk_time:chunk_time+t_chunk_length]

            #### resample
            f = scipy.interpolate.interp1d(np.linspace(0, 1, x_pre.shape[-1]), x_pre, kind='linear')
            x_resampled = f(np.linspace(0, 1, tf_dw_length))

            #### verify
            if debug:
                plt.pcolormesh(x_pre)
                plt.pcolormesh(x_resampled)
                plt.show()

            tf_chunk_baseline[nchan,chunk_i,:,:] = x_resampled

    #### verify
    if debug:
        plt.pcolormesh(tf_chunk_baseline[0,:,:,:].mean(axis=0))
        plt.show()

    return tf_chunk_baseline




#tf = tf_allchan
def compute_chunk_tf_dB_AC(sujet, tf, band, srate, monopol):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    
    if monopol:
        baselines = np.load(f'{sujet}_{band[:-2]}_baselines.npy')
    else:
        baselines = np.load(f'{sujet}_{band[:-2]}_baselines_bi.npy')

    #### load erp starts
    ac_starts = get_ac_starts(sujet)

    #### apply baseline
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    #### length for tf dw sampled
    resample_length_AC = srate_dw_stats*AC_length*3

    #### chunk
    def chunk_tf_db_n_chan(n_chan):

        print_advancement(n_chan, tf.shape[0], steps=[25, 50, 75])

        tf_chunk = np.zeros((len(ac_starts), tf.shape[1], resample_length_AC))

        for fi in range(tf.shape[1]):

            x = tf[n_chan,fi,:]

            data_chunk = np.zeros((len(ac_starts), resample_length_AC))

            for start_i, start_time in enumerate(ac_starts):

                t_start = int(start_time + t_start_AC*srate)
                t_stop = int(start_time + t_stop_AC*srate)

                x_pre = x[t_start: t_stop]

                #### resample
                f = scipy.interpolate.interp1d(np.linspace(0, 1, x_pre.shape[0]), x_pre, kind='linear')
                x_resampled = f(np.linspace(0, 1, resample_length_AC))

                data_chunk[start_i,:] = x_resampled

            tf_chunk[:,fi,:] = data_chunk

            #### verif
            if debug:
                plt.pcolormesh(tf_chunk.mean(axis=0))
                plt.show()

                plt.plot(np.linspace(0, 1, x_pre.shape[0]), x_pre, label='pre')
                plt.plot(np.linspace(0, 1, x_resampled.shape[0]), x_resampled, label='post')
                plt.legend()
                plt.show()

        return tf_chunk

    chunk_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    tf_chunk_cond = np.zeros((tf.shape[0], len(ac_starts), tf.shape[1], resample_length_AC))

    for n_chan in range(tf.shape[0]):
        tf_chunk_cond[n_chan,:,:,:] = chunk_tf_db_nchan_res[n_chan]

    #### free RAM
    del chunk_tf_db_nchan_res

    return tf_chunk_cond




#tf = tf_allchan
def compute_chunk_tf_dB_SNIFF(sujet, tf, band, srate, monopol):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    
    if monopol:
        baselines = np.load(f'{sujet}_{band[:-2]}_baselines.npy')
    else:
        baselines = np.load(f'{sujet}_{band[:-2]}_baselines_bi.npy')

    #### load erp starts
    sniff_starts = get_sniff_starts(sujet)

    #### apply baseline
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    #### length for tf dw sampled
    resample_length_SNIFF = srate_dw_stats*SNIFF_length*2

    #### chunk
    def chunk_tf_db_n_chan(n_chan):

        print_advancement(n_chan, tf.shape[0], steps=[25, 50, 75])

        tf_chunk = np.zeros((len(sniff_starts), tf.shape[1], resample_length_SNIFF))

        for fi in range(tf.shape[1]):

            x = tf[n_chan,fi,:]

            data_chunk = np.zeros((len(sniff_starts), resample_length_SNIFF))

            for start_i, start_time in enumerate(sniff_starts):

                t_start = int(start_time + t_start_SNIFF*srate)
                t_stop = int(start_time + t_stop_SNIFF*srate)

                x_pre = x[t_start: t_stop]

                #### resample
                f = scipy.interpolate.interp1d(np.linspace(0, 1, x_pre.shape[0]), x_pre, kind='linear')
                x_resampled = f(np.linspace(0, 1, resample_length_SNIFF))

                data_chunk[start_i,:] = x_resampled

            tf_chunk[:,fi,:] = data_chunk

            #### verif
            if debug:
                plt.pcolormesh(tf_chunk.mean(axis=0))
                plt.show()

                plt.plot(np.linspace(0, 1, x_pre.shape[0]), x_pre, label='pre')
                plt.plot(np.linspace(0, 1, x_resampled.shape[0]), x_resampled, label='post')
                plt.legend()
                plt.show()

        return tf_chunk

    chunk_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    tf_chunk_cond = np.zeros((tf.shape[0], len(sniff_starts), tf.shape[1], resample_length_SNIFF))

    for n_chan in range(tf.shape[0]):
        tf_chunk_cond[n_chan,:,:,:] = chunk_tf_db_nchan_res[n_chan]

    #### free RAM
    del chunk_tf_db_nchan_res

    return tf_chunk_cond









################################
######## SHUFFLE ########
################################


def get_pixel_extrema_shuffle(nchan, tf_chunk_baseline, tf_chunk_cond):

    #### define ncycle
    n_cycle_baselines = tf_chunk_baseline.shape[1]
    n_cycle_cond = tf_chunk_cond.shape[1]
    n_cycle_tot = n_cycle_baselines + n_cycle_cond

    #### random selection
    sel = np.random.randint(low=0, high=n_cycle_tot, size=n_cycle_cond)
    sel_baseline = np.array([i for i in sel if i <= n_cycle_baselines-1])
    sel_cond = np.array([i for i in sel - n_cycle_baselines if i >= 0])

    #### extract max min
    tf_shuffle = np.concatenate((tf_chunk_baseline[nchan, sel_baseline, :, :], tf_chunk_cond[nchan, sel_cond, :, :]))
    tf_shuffle = np.mean(tf_shuffle, axis=0)
    tf_shuffle = rscore_mat(tf_shuffle)
    max, min = tf_shuffle.max(axis=1), tf_shuffle.min(axis=1)

    if debug:

        plt.pcolormesh(tf_shuffle)
        plt.show()

    return max, min

    






################################
######## COMPUTE STATS ########
################################


#cond = 'RD_SV'
def precompute_tf_STATS(sujet, monopol):

    print('#### COMPUTE TF STATS ####')

    #### identify if already computed for all

    compute_token = 0

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep_i]

        for cond in ['AC', 'SNIFF']:

            #band, freq = list(freq_band.items())[0]
            for band, freq in freq_band.items():

                if monopol == 'monopolaire':
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy') == False:
                        compute_token += 1
                else:
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy') == False:
                        compute_token += 1

    if compute_token == 0:
        print('ALL COND ALREADY COMPUTED')
        return

    #### open params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, monopol)

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep_i] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            #cond_compute = 'SNIFF'
            for cond_compute in ['AC', 'SNIFF']:

                ######## COMPUTE FOR FR_CV BASELINES ########

                cond = 'FR_CV'
                data = load_data(sujet, cond, electrode_recording_type, band_prep=band_prep)[:len(chan_list_ieeg),:]

                #### convolution
                wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

                os.chdir(path_memmap)
                if monopol == 'monopolaire':
                    tf = np.memmap(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
                else:
                    tf = np.memmap(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                print(f'CONV baselines {band}')
            
                def compute_tf_convolution_nchan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf_i = np.zeros((nfrex, x.shape[0]))

                    for fi in range(nfrex):
                        
                        tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                    tf[n_chan,:,:] = tf_i

                    return

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                #### chunk
                print('CHUNK')
                tf_chunk_baseline = compute_chunk_baselines_tf_dB(sujet, tf, cond_compute, band, srate, monopol)

                os.chdir(path_memmap)
                try:
                    if monopol == 'monopolaire':
                        os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                    else:
                        os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat')
                    del tf
                except:
                    pass

                ######## COMPUTE FOR OTHER COND ########

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                if monopol == 'monopolaire':
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond_compute}.npy'):
                        print(f'ALREADY COMPUTED {cond_compute}')
                        continue
                else:
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond_compute}_bi.npy'):
                        print(f'ALREADY COMPUTED {cond_compute}')
                        continue

                #### compute stretch for cond
                data = load_data(sujet, cond_compute, electrode_recording_type, band_prep=band_prep)[:len(chan_list_ieeg),:]

                #### convolution
                wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

                os.chdir(path_memmap)
                if monopol == 'monopolaire':
                    tf = np.memmap(f'{sujet}_{cond_compute}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
                else:
                    tf = np.memmap(f'{sujet}_{cond_compute}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                print(f'CONV {cond_compute} {band}')

                def compute_tf_convolution_nchan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf_i = np.zeros((nfrex, x.shape[0]))

                    for fi in range(nfrex):
                        
                        tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                    tf[n_chan,:,:] = tf_i

                    return

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                #### chunk
                if cond_compute == 'AC':
                    tf_chunk_cond = compute_chunk_tf_dB_AC(sujet, tf, band, srate, monopol)

                if cond_compute == 'SNIFF':
                    tf_chunk_cond = compute_chunk_tf_dB_SNIFF(sujet, tf, band, srate, monopol)

                os.chdir(path_memmap)
                try:
                    if monopol == 'monopolaire':
                        os.remove(f'{sujet}_{cond_compute}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                    else:
                        os.remove(f'{sujet}_{cond_compute}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat')
                    del tf
                except:
                    pass

                ######## COMPUTE SURROGATES & STATS ########

                print('SURROGATES')

                pixel_based_distrib = np.zeros((tf_chunk_baseline.shape[0], 50, 2))

                #nchan = 0
                for nchan in range(tf_chunk_baseline.shape[0]):

                    print_advancement(nchan, tf_chunk_baseline.shape[0], steps=[25, 50, 75])

                    pixel_based_distrib_i = np.zeros((tf_chunk_baseline.shape[2], 2, n_surrogates_tf))

                    #surrogates_i = 0
                    for surrogates_i in range(n_surrogates_tf):

                        pixel_based_distrib_i[:,0,surrogates_i], pixel_based_distrib_i[:,1,surrogates_i] =  get_pixel_extrema_shuffle(nchan, tf_chunk_baseline, tf_chunk_cond)

                    min, max = np.percentile(pixel_based_distrib_i.reshape(tf_chunk_baseline.shape[2], -1), 2.5, axis=-1), np.percentile(pixel_based_distrib_i.reshape(tf_chunk_baseline.shape[2], -1), 97.5, axis=-1) 
                    
                    if debug:
                        plt.hist(pixel_based_distrib_i[0, :, :].reshape(-1), bins=500)
                        plt.show()

                    pixel_based_distrib[nchan, :, 0], pixel_based_distrib[nchan, :, 1] = max, min

                #### plot 
                if debug:
                    for nchan in range(20):
                        tf = rscore_mat(tf_chunk_cond[nchan, :, :].mean(axis=0))
                        tf_thresh = tf.copy()
                        #wavelet_i = 0
                        for wavelet_i in range(nfrex):
                            mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
                            tf_thresh[wavelet_i, mask] = 1
                            tf_thresh[wavelet_i, np.logical_not(mask)] = 0
                    
                        plt.pcolormesh(tf)
                        plt.contour(tf_thresh, levels=0, colors='r')
                        plt.show()

                        plt.pcolormesh(tf_thresh)
                        plt.show()

                ######## SAVE ########

                print(f'SAVE {cond_compute} {band}')

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                if monopol == 'monopolaire':
                    np.save(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond_compute}.npy', pixel_based_distrib)
                else:
                    np.save(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond_compute}_bi.npy', pixel_based_distrib)




                        
                
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    

        #monopol = 'monopolaire'
        for monopol in ['monopolaire', 'bipolaire']:
    
            # precompute_tf(sujet, cond, 0, freq_band_list_precompute, band_prep_list, monopol)
            execute_function_in_slurm_bash_mem_choice('n6bis_precompute_TF_STATS', 'precompute_tf_STATS', [sujet, monopol], '30G')

        







