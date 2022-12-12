
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


#tf = tf_baselines
def compute_stretch_tf_dB(sujet, tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate, monopol):

    #### load baseline
    band = band[:-2]
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    if monopol:
        baselines = np.load(f'{sujet}_{band}_baselines.npy')
    else:
        baselines = np.load(f'{sujet}_{band}_baselines_bi.npy')

    #### apply baseline
    os.chdir(path_memmap)
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        tf_stretch_i = stretch_data_tf(respfeatures_allcond[cond][session_i], stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_stretch_i

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    #### extract
    n_cycles = respfeatures_allcond[cond][session_i]['select'].sum()

    tf_stretch_allchan = np.zeros((tf.shape[0], n_cycles, tf.shape[1], stretch_point_TF))

    for n_chan in range(tf.shape[0]):
        tf_stretch_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_stretch_allchan











################################
######## SHUFFLE ########
################################


def get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond):

    #### define ncycle
    n_cycle_baselines = tf_stretch_baselines.shape[1]
    n_cycle_cond = tf_stretch_cond.shape[1]
    n_cycle_tot = n_cycle_baselines + n_cycle_cond

    #### random selection
    sel = np.random.randint(low=0, high=n_cycle_tot, size=n_cycle_cond)
    sel_baseline = np.array([i for i in sel if i <= n_cycle_baselines-1])
    sel_cond = np.array([i for i in sel - n_cycle_baselines if i >= 0])

    #### extract max min
    tf_shuffle = np.concatenate((tf_stretch_baselines[nchan, sel_baseline, :, :], tf_stretch_cond[nchan, sel_cond, :, :]))
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

    #### open params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, monopol)
    respfeatures_allcond = load_respfeatures(sujet)
    prms = get_params(sujet, monopol)

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep_i] 

        #band, freq = list(freq_band.items())[1]
        for band, freq in freq_band.items():

            ######## COMPUTE FOR FR_CV BASELINES ########

            cond = 'FR_CV'
            session_i = 0
            data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

            #### convolution
            wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

            os.chdir(path_memmap)
            if monopol:
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

            #### stretch
            tf_stretch = compute_stretch_tf_dB(sujet, tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate, monopol)

            os.chdir(path_memmap)
            try:
                if monopol:
                    os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                else:
                    os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat')
                del tf
            except:
                pass

            n_cycles = respfeatures_allcond[cond][session_i]['select'].sum()

            if monopol:
                tf_stretch_baselines = np.memmap(f'{sujet}_baselines_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycles, nfrex, stretch_point_TF))
            else:
                tf_stretch_baselines = np.memmap(f'{sujet}_baselines_{band}_{str(freq[0])}_{str(freq[1])}_stretch_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycles, nfrex, stretch_point_TF))

            tf_stretch_baselines = tf_stretch.copy()
            del tf_stretch

            ######## COMPUTE FOR OTHER COND ########
            
            #cond = 'RD_SV'
            for cond in prms['conditions']:

                if cond == 'FR_CV':
                    continue

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                if monopol:
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy'):
                        print(f'ALREADY COMPUTED {cond}')
                        continue
                else:
                    if os.path.exists(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy'):
                        print(f'ALREADY COMPUTED {cond}')
                        continue

                #### identify info for each session
                n_session = len(respfeatures_allcond[cond])

                n_cycle_list = []
                for session_i in range(n_session):
                    n_cycle_list.append(respfeatures_allcond[cond][session_i]['select'].sum())

                n_cycle_list = np.array(n_cycle_list)

                os.chdir(path_memmap)
                if monopol:
                    tf_stretch_cond = np.memmap(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycle_list.sum(), nfrex, stretch_point_TF))
                else:
                    tf_stretch_cond = np.memmap(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_stretch_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycle_list.sum(), nfrex, stretch_point_TF))

                #session_i = 0
                for session_i in range(n_session):
            
                    #### compute stretch for cond
                    data = load_data_sujet(sujet, band_prep, cond, session_i, monopol)[:len(chan_list_ieeg),:]

                    #### convolution
                    wavelets, nfrex = get_wavelets(sujet, band_prep, freq, monopol)

                    os.chdir(path_memmap)
                    if monopol:
                        tf = np.memmap(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
                    else:
                        tf = np.memmap(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                    print(f'CONV {cond} {session_i} {band}')

                    def compute_tf_convolution_nchan(n_chan):

                        print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                        x = data[n_chan,:]

                        tf_i = np.zeros((nfrex, x.shape[0]))

                        for fi in range(nfrex):
                            
                            tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                        tf[n_chan,:,:] = tf_i

                        return

                    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                    #### stretch
                    tf_stretch = compute_stretch_tf_dB(sujet, tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate, monopol)

                    os.chdir(path_memmap)
                    try:
                        if monopol:
                            os.remove(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                        else:
                            os.remove(f'{sujet}_{cond}_{session_i}_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions_bi.dat')
                        del tf
                    except:
                        pass
                    
                    #### identify position cycle
                    if session_i == 0:
                        n_cycle_pre = 0
                        n_cycle_post = n_cycle_list[session_i]
                    else:
                        n_cycle_pre += n_cycle_list[session_i-1]
                        n_cycle_post = n_cycle_pre + n_cycle_list[session_i]
    
                    #### fill mat
                    tf_stretch_cond[:, n_cycle_pre:n_cycle_post, :, :] = tf_stretch.copy()
                    del tf_stretch

                ######## COMPUTE SURROGATES & STATS ########

                print('SURROGATES')

                pixel_based_distrib = np.zeros((tf_stretch_baselines.shape[0], 50, 2))

                #nchan = 0
                for nchan in range(tf_stretch_baselines.shape[0]):

                    print_advancement(nchan, tf_stretch_baselines.shape[0], steps=[25, 50, 75])

                    pixel_based_distrib_i = np.zeros((tf_stretch_baselines.shape[2], 2, n_surrogates_tf))

                    #surrogates_i = 0
                    for surrogates_i in range(n_surrogates_tf):

                        pixel_based_distrib_i[:,0,surrogates_i], pixel_based_distrib_i[:,1,surrogates_i] =  get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond)

                    min, max = np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 2.5, axis=-1), np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 97.5, axis=-1) 
                    
                    if debug:
                        plt.hist(pixel_based_distrib_i[0, :, :].reshape(-1), bins=500)
                        plt.show()

                    pixel_based_distrib[nchan, :, 0], pixel_based_distrib[nchan, :, 1] = max, min

                #### plot 
                if debug:
                    for nchan in range(20):
                        tf = rscore_mat(tf_stretch_cond[nchan, :, :].mean(axis=0))
                        tf_thresh = tf.copy()
                        #wavelet_i = 0
                        for wavelet_i in range(nfrex):
                            mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
                            tf_thresh[wavelet_i, mask] = 1
                            tf_thresh[wavelet_i, np.logical_not(mask)] = 0
                    
                        plt.pcolormesh(tf)
                        plt.contour(tf_thresh, levels=0)
                        plt.show()

                ######## SAVE ########
 
                print(f'SAVE {band}')

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                if monopol:
                    np.save(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy', pixel_based_distrib)
                else:
                    np.save(f'{sujet}_STATS_tf_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy', pixel_based_distrib)

                #### remove cond

                os.chdir(path_memmap)

                try:
                    if monopol:
                        os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat')
                    else:
                        os.remove(f'{sujet}_{cond}_{band}_{str(freq[0])}_{str(freq[1])}_stretch_bi.dat')
                    del tf_stretch_cond
                except:
                    pass

            #### remove baselines after cond computing
            try:
                if monopol:
                    os.remove(f'{sujet}_baselines_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat')
                else:
                    os.remove(f'{sujet}_baselines_{band}_{str(freq[0])}_{str(freq[1])}_stretch_bi.dat')
                del tf_stretch_cond
            except:
                pass


                    
            
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    

        #monopol = True
        for monopol in [True, False]:
    
            # precompute_tf(sujet, cond, 0, freq_band_list_precompute, band_prep_list, monopol)
            execute_function_in_slurm_bash_mem_choice('n7bis_precompute_TF_STATS', 'precompute_tf_STATS', [sujet, monopol], '20G')

        







