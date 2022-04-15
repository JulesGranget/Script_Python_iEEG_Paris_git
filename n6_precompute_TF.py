

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config import *
from n0bis_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################




#tf, cond, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex = tf_allchan, cond, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex
def compute_stretch_tf_dB(tf, cond, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'Baselines'))
    
    baselines = np.load(f'{sujet}_{band}_baselines.npy')

    #### apply baseline
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond[cond][0], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan




#tf, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate = tf_allchan, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate
def compute_stretch_tf_dB_AC(tf, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'Baselines'))
    
    baselines = np.load(f'{sujet}_{band}_baselines.npy')

    #### apply baseline
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF_ac)))

        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            data_chunk = np.zeros(( len(ac_starts), int(np.abs(t_start_AC)*srate +  t_stop_AC*srate) ))

            for start_i, start_time in enumerate(ac_starts):

                t_start = int(start_time + t_start_AC*srate)
                t_stop = int(start_time + t_stop_AC*srate)

                data_chunk[start_i,:] = x[t_start: t_stop]

            tf_mean[fi,:] = np.mean(data_chunk, axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF_ac))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan




#tf, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate = tf_allchan, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate
def compute_stretch_tf_dB_SNIFF(tf, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'Baselines'))
    
    baselines = np.load(f'{sujet}_{band}_baselines.npy')

    #### apply baseline
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF_sniff)))

        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            data_chunk = np.zeros(( len(sniff_starts), int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate) ))

            for start_i, start_time in enumerate(sniff_starts):

                t_start = int(start_time + t_start_SNIFF*srate)
                t_stop = int(start_time + t_stop_SNIFF*srate)

                data_chunk[start_i,:] = x[t_start: t_stop]

            tf_mean[fi,:] = np.mean(data_chunk, axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF_sniff))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan






def compute_stretch_tf_itpc(tf, cond, respfeatures_allcond, stretch_point_TF, srate):
    
    #### identify number stretch
    x = tf[0,:]
    x_stretch, ratio = stretch_data(respfeatures_allcond[cond][0], stretch_point_TF, x, srate)
    nb_cycle = np.size(x_stretch, 0)
    
    #### compute tf
    tf_stretch = np.zeros((nb_cycle, np.size(tf,0), int(stretch_point_TF)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        x_stretch, ratio = stretch_data(respfeatures_allcond[cond][0], stretch_point_TF, x, srate)
        tf_stretch[:,fi,:] = x_stretch

    return tf_stretch



def compute_stretch_tf_itpc_ac(tf, cond, ac_starts, srate):
    
    #### identify number stretch
    nb_ac = len(ac_starts)
    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    
    #### compute tf
    tf_stretch = np.zeros((nb_ac, np.size(tf,0), int(stretch_point_TF_ac)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        data_chunk = np.zeros(( len(ac_starts), stretch_point_TF_ac ), dtype='complex')

        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            data_chunk[start_i,:] = x[t_start: t_stop]

        tf_stretch[:,fi,:] = data_chunk

    return tf_stretch




def compute_stretch_tf_itpc_sniff(tf, cond, sniff_starts, srate):
    
    #### identify number stretch
    nb_ac = len(sniff_starts)
    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    
    #### compute tf
    tf_stretch = np.zeros((nb_ac, np.size(tf,0), int(stretch_point_TF_sniff)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        data_chunk = np.zeros(( len(sniff_starts), stretch_point_TF_sniff ), dtype='complex')

        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_chunk[start_i,:] = x[t_start: t_stop]

        tf_stretch[:,fi,:] = data_chunk

    return tf_stretch






################################
######## PRECOMPUTE TF ########
################################


def precompute_tf(sujet, cond, band_prep_list):

    print('TF PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### select prep to load
    #band_prep = 'lf'
    for band_prep in band_prep_list:

        #### select data without aux chan
        data = load_data(cond, band_prep=band_prep)

        #### remove aux chan
        data = data[:-4,:]

        freq_band = freq_band_dict[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            #### select wavelet parameters
            wavelets = get_wavelets(band_prep, freq)

            if band_prep == 'lf':
                nfrex = nfrex_lf
            elif band_prep == 'hf':
                nfrex = nfrex_hf
            elif band_prep == 'wb':
                nfrex = nfrex_wb

            #### compute
            os.chdir(path_memmap)
            tf_allchan = np.memmap(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(np.size(data,0), nfrex, np.size(data,1)))

            def compute_tf_convolution_nchan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(np.size(data,0)))

            #### stretch
            if cond == 'FR_CV':
                print('STRETCH_VS')
                tf_allband_stretched = compute_stretch_tf_dB(tf_allchan, cond, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex, srate)

            if cond == 'AC':
                print('STRETCH_AC')
                ac_starts = get_ac_starts(sujet)
                tf_allband_stretched = compute_stretch_tf_dB_AC(tf_allchan, cond, ac_starts, stretch_point_TF, band, band_prep, nfrex, srate)

            if cond == 'SNIFF':
                print('STRETCH_SNIFF')
                sniff_starts = get_sniff_starts(sujet)
                tf_allband_stretched = compute_stretch_tf_dB_SNIFF(tf_allchan, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate)

            
            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy', tf_allband_stretched)
            
            os.chdir(path_memmap)
            os.remove(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_precompute_convolutions.dat')





################################
######## PRECOMPUTE ITPC ########
################################



def precompute_tf_itpc(sujet, cond, band_prep_list):

    print('ITPC PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    
    #### select prep to load
    for band_prep in band_prep_list:

        #### select data without aux chan
        data = load_data(cond, band_prep=band_prep)

        #### remove aux chan
        data = data[:-4,:]

        freq_band = freq_band_dict[band_prep]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)

            #### select wavelet parameters
            wavelets = get_wavelets(band_prep, freq)

            if band_prep == 'lf':
                nfrex = nfrex_lf
            elif band_prep == 'hf':
                nfrex = nfrex_hf
            elif band_prep == 'wb':
                nfrex = nfrex_wb

            #### compute itpc
            print('COMPUTE, STRETCH & ITPC')
            def compute_itpc_n_chan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                #### stretch
                if cond == 'FR_CV':
                    tf_stretch = compute_stretch_tf_itpc(tf, cond, respfeatures_allcond, stretch_point_TF, srate)

                elif cond == 'AC':
                    ac_starts = get_ac_starts(sujet)
                    tf_stretch = compute_stretch_tf_itpc_ac(tf, cond, ac_starts, srate)

                elif cond == 'SNIFF':
                    sniff_starts = get_sniff_starts(sujet)
                    tf_stretch = compute_stretch_tf_itpc_sniff(tf, cond, sniff_starts, srate)

                #### ITPC
                tf_angle = np.angle(tf_stretch)
                tf_cangle = np.exp(1j*tf_angle) 
                itpc = np.abs(np.mean(tf_cangle,0))

                if debug == True:
                    time = range(stretch_point_TF)
                    frex = range(nfrex)
                    plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
                    plt.show()

                return itpc 

            compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(np.size(data,0)))
            
            if cond == 'FR_CV':
                itpc_allchan = np.zeros((np.size(data,0),nfrex,stretch_point_TF))

            elif cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                itpc_allchan = np.zeros((np.size(data,0),nfrex,stretch_point_TF_ac))

            elif cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
                itpc_allchan = np.zeros((np.size(data,0),nfrex,stretch_point_TF_sniff))

            for n_chan in range(np.size(data,0)):

                itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy', itpc_allchan)

            del itpc_allchan



####################################
######### SNIFF CHUNKS ########
####################################


#tf, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate = tf_allchan, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate
def compute_tf_SNIFF(tf, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate):

    def chunk_tf_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)

        tf_sniff = np.zeros(( len(sniff_starts), np.size(tf,1), int(stretch_point_TF_sniff) ))

        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]

            for start_i, start_time in enumerate(sniff_starts):

                t_start = int(start_time + t_start_SNIFF*srate)
                t_stop = int(start_time + t_stop_SNIFF*srate)

                tf_sniff[start_i, fi, :] = x[t_start: t_stop]

        tf_sniff_mean = np.mean(tf_sniff, 1)

        return tf_sniff_mean

    chunk_tf_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    tf_sniff_allchan = np.zeros((np.size(tf,0), len(sniff_starts), stretch_point_TF_sniff))

    for n_chan in range(np.size(tf,0)):
        tf_sniff_allchan[n_chan,:,:] = chunk_tf_nchan_res[n_chan]

    return tf_sniff_allchan


def precompute_tf_sniff(sujet, cond, band_prep_list):

    print('TF PRECOMPUTE')

    cond = 'SNIFF'

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### select prep to load
    #band_prep = 'lf'
    for band_prep in band_prep_list:

        #### select data without aux chan
        data = load_data(cond, band_prep=band_prep)

        #### remove aux chan
        data = data[:-4,:]

        freq_band = freq_band_dict[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_ALL_{cond}.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            #### select wavelet parameters
            wavelets = get_wavelets(band_prep, freq)

            if band_prep == 'lf':
                nfrex = nfrex_lf
            elif band_prep == 'hf':
                nfrex = nfrex_hf
            elif band_prep == 'wb':
                nfrex = nfrex_wb

            #### compute
            os.chdir(path_memmap)
            tf_allchan = np.memmap(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_ALL_{cond}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(np.size(data,0), nfrex, np.size(data,1)))

            def compute_tf_convolution_nchan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(np.size(data,0)))

            #### stretch
            print('CHUNK_SNIFF')
            sniff_starts = get_sniff_starts(sujet)
            tf_chunk_SNIFF = compute_tf_SNIFF(tf_allchan, cond, sniff_starts, stretch_point_TF, band, band_prep, nfrex, srate)

            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_ALL_{cond}.npy', tf_chunk_SNIFF)
            
            os.chdir(path_memmap)
            os.remove(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_ALL_{cond}_precompute_convolutions.dat')












################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #### compute and save tf
    #cond = 'AC'
    for cond in conditions_compute_TF:

        print(cond)
    
        #precompute_tf(session_eeg, cond, 0, band_prep_list)
        execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf', [sujet, cond, band_prep_list])
        #precompute_tf_itpc(session_eeg, cond, 0, band_prep_list)
        execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf_itpc', [sujet, cond, band_prep_list])
    
    #### compute sniff chunks
    #precompute_tf_sniff(sujet, 'SNIFF', band_prep_list)
    #execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf_sniff', [sujet, 'SNIFF', band_prep_list])



