

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################
######## COMPUTE TF ########
################################


def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


def precompute_tf(sujet, cond, session_i, band_prep_list):

    print('TF PRECOMPUTE')

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### select prep to load
    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data_allsession = load_data(sujet, cond, band_prep=band_prep)

        data = data_allsession[session_i][:-4,:]
        
        freq_band = freq_band_list_precompute[band_prep_i]

        tf_allband = {}

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            print(band, ' : ', freq)
        
            print('COMPUTE')

            #### select wavelet parameters
            wavelets, nfrex = get_wavelets(sujet, band_prep, freq)

            def compute_tf_convolution_nchan(n_chan):

                print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                return tf

            tf_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

            #### fill tf_allchan 
            tf_allchan = np.zeros((len(tf_res), tf_res[0].shape[0], tf_res[0].shape[1]))
            
            for n_chan, _ in enumerate(chan_list_ieeg):

                tf_allchan[n_chan, :, :] = tf_res[n_chan]

            del tf_res

            #### dB
            #### load baseline
            os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
            
            baselines = np.load(f'{sujet}_{band[:-2]}_baselines.npy')

            #### apply baseline
            for n_chan in range(tf_allchan.shape[0]):
                
                for fi in range(tf_allchan.shape[1]):

                    activity = tf_allchan[n_chan,fi,:]
                    baseline_fi = baselines[n_chan, fi]

                    #### verify baseline
                    #plt.plot(activity)
                    #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
                    #plt.show()

                    tf_allchan[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

            tf_allband[band] = tf_allchan

            # print('SAVE')
            # os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            # np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy', tf_allchan)

        #### resample TF AL
        tf_allband_resampled = {}

        for band in tf_allband:

            for n_chan in range(data.shape[0]):

                tf_allband_resampled[band] = scipy.signal.resample(tf_allband[band][n_chan,:,:], resampled_points_AL, axis=1)

        #### export TF AL
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        for band, freq in freq_band.items():
            np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy', tf_allband_resampled[band])
        
        del tf_allband_resampled

        #### plot and save all the tf for one session
        nrows = len(freq_band)
        df_loca = get_loca_df(sujet)

        #nchan = 0
        for nchan, chan_name in enumerate(chan_list_ieeg):

            #### check if already computed
            chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'AL'))
            if os.path.exists(f'{sujet}_{chan_name}_{chan_loca}_AL{session_i+1}_{band_prep}.jpeg'):
                print('ALREADY COMPUTED')
                continue

            freq_band = freq_band_list_precompute[band_prep_i]

            #### compute scales
            scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

            for i, (band, freq) in enumerate(freq_band.items()) :

                if band == 'whole' or band == 'l_gamma':
                    continue

                data = tf_allband[band][nchan, :, :]

                scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
                scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
                scales['median_val'] = np.append(scales['median_val'], np.median(data))

            median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

            vmin = np.median(scales['median_val']) - median_diff
            vmax = np.median(scales['median_val']) + median_diff

            #### initiate fig
            fig, axs = plt.subplots(nrows=nrows)
            
            plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}_AL{session_i+1}')

            #### for plotting l_gamma down
            if band_prep == 'hf':
                keys_list_reversed = list(freq_band.keys())
                keys_list_reversed.reverse()
                freq_band_reversed = {}
                for key_i in keys_list_reversed:
                    freq_band_reversed[key_i] = freq_band[key_i]
                freq_band = freq_band_reversed

            #### plot
            for i, (band, freq) in enumerate(freq_band.items()):

                data = tf_allband[band][nchan, :, :]

                frex = np.linspace(freq[0], freq[1], np.size(data,0))
            
                ax = axs[i]

                time = np.arange(data.shape[1])/srate

                # ax.pcolormesh(time, frex, data, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
                ax.pcolormesh(time, frex, robust_zscore(data), vmin=-robust_zscore(data).max(), vmax=robust_zscore(data).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))
                
                ax.set_ylabel(band)

            #plt.show()

            #### save
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary', 'AL'))
            fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_AL{session_i+1}_{band_prep}.jpeg', dpi=150)
            plt.close()

    print('done')

            









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### load n_session
        cond = 'AL'
        n_session = len(load_data(sujet, cond, band_prep=band_prep_list[0]))

        #### compute and save tf
        #session_i = 0
        for session_i in range(n_session):
            #precompute_tf(sujet, cond, session_i, band_prep_list)
            execute_function_in_slurm_bash_mem_choice('n7_precompute_AL', 'precompute_tf', [sujet, cond, session_i, band_prep_list], '15G')
        









