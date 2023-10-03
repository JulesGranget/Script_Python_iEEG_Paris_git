

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
######## STRETCH TF ########
################################




#tf = tf_allchan
def compute_stretch_tf(sujet, tf, cond, respfeatures_allcond, stretch_point_TF, srate, electrode_recording_type):

    #n_chan = 0
    def stretch_tf_db_n_chan(n_chan):

        tf_mean = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_mean

    #### export raw
    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))
    n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf[0,:,:], srate)[0].shape[0]
    tf_mean_allchan = np.zeros((tf.shape[0], n_cycle_stretch, tf.shape[1], stretch_point_TF))

    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    if debug:

        plt.pcolormesh(np.median(tf_mean_allchan[0,:,:,:], axis=0))
        plt.show()

    print('SAVE RAW', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

    #### norm
    tf[:] = norm_tf(sujet, tf, electrode_recording_type, norm_method)

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    #### extarct
    n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf[0,:,:], srate)[0].shape[0]
    tf_mean_allchan[:] = np.zeros((tf.shape[0], n_cycle_stretch, tf.shape[1], stretch_point_TF))

    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan






#tf = tf_allchan
def compute_stretch_tf_AC(sujet, tf, ac_starts, srate, electrode_recording_type):

    cond = 'AC'

    #n_chan = 0
    def chunk_tf_db_n_chan(n_chan):

        print_advancement(n_chan, tf.shape[0], steps=[25, 50, 75])

        #start_i, start_time = 0, ac_starts[0]
        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            x = tf_norm[n_chan,:,t_start:t_stop]     

            f = scipy.interpolate.interp1d(np.linspace(0, 1, x.shape[-1]), x, kind='linear')
            x_resampled = f(np.linspace(0, 1, stretch_point_TF_ac_resample))

            tf_mean_allchan[n_chan,start_i,:,:] = x_resampled

    #### raw chunk
    os.chdir(path_memmap)
    tf_norm = np.memmap(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape), offset=mem_crnl_cluster_offset)
    tf_mean_allchan = np.memmap(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape[0], len(ac_starts),tf.shape[1],stretch_point_TF_ac_resample), offset=mem_crnl_cluster_offset)

    tf_norm[:] = tf
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    print('SAVE RAW', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

    #### norm
    os.chdir(path_memmap)

    tf_norm[:] = norm_tf(sujet, tf, electrode_recording_type, norm_method)
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    try:
        os.remove(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat')
    except:
        pass

    return tf_mean_allchan





#tf = tf_allchan
def compute_stretch_tf_SNIFF(sujet, tf, sniff_starts, srate, electrode_recording_type):

    cond = 'SNIFF'

    #n_chan = 0
    def chunk_tf_db_n_chan(n_chan):

        print_advancement(n_chan, tf_norm.shape[0], steps=[25, 50, 75])

        #start_i, start_time = 0, sniff_starts[0]
        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            x = tf_norm[n_chan,:,t_start:t_stop]     

            f = scipy.interpolate.interp1d(np.linspace(0, 1, x.shape[-1]), x, kind='linear')
            x_resampled = f(np.linspace(0, 1, stretch_point_TF_sniff_resampled))

            tf_mean_allchan[n_chan,start_i,:,:] = x_resampled
    
    #### raw
    os.chdir(path_memmap)
    tf_norm = np.memmap(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape))
    tf_mean_allchan = np.memmap(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape[0], len(sniff_starts), tf.shape[1], stretch_point_TF_sniff_resampled))

    tf_norm[:] = tf
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    print('SAVE RAW', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

    #### norm
    os.chdir(path_memmap)
    tf_norm[:] = norm_tf(sujet, tf, electrode_recording_type, norm_method)
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    try:
        os.remove(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat')
    except:
        pass

    return tf_mean_allchan






#tf = tf_allchan
def compute_stretch_tf_AL(sujet, cond, tf, AL_len_list, srate, electrode_recording_type):

    AL_chunk_time_raw = srate*AL_chunk_pre_post_time

    #n_chan = 0
    def chunk_tf_n_chan(n_chan):

        print_advancement(n_chan, tf_norm.shape[0], steps=[25, 50, 75])

        AL_pre, AL_post = 0, AL_len_list[0]

        #AL_i = 0
        for AL_i in range(AL_n):

            if AL_i != 0:
                AL_pre, AL_post = AL_pre + AL_len_list[AL_i-1], AL_post + AL_len_list[AL_i]

            #### chunk pre
            tf_chunk = tf_norm[n_chan,:,AL_pre:AL_pre+AL_chunk_time_raw]

            f = scipy.interpolate.interp1d(np.linspace(0, 1, tf_chunk.shape[-1]), tf_chunk, kind='linear')
            tf_mean_allchan[n_chan,AL_i,:,:int(resampled_points_AL/2)] = f(np.linspace(0, 1, int(resampled_points_AL/2)))

            #### chunk post
            tf_chunk = tf_norm[n_chan,:,AL_post-AL_chunk_time_raw:AL_post]

            f = scipy.interpolate.interp1d(np.linspace(0, 1, tf_chunk.shape[-1]), tf_chunk, kind='linear')
            tf_mean_allchan[n_chan,AL_i,:,int(resampled_points_AL/2):] = f(np.linspace(0, 1, int(resampled_points_AL/2)))

    #### raw
    os.chdir(path_memmap)
    tf_norm = tf
    tf_mean_allchan = np.memmap(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape[0], AL_n, tf.shape[1], resampled_points_AL))

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    print('SAVE RAW', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_raw_{cond}.npy', tf_mean_allchan)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_raw_{cond}_bi.npy', tf_mean_allchan)

    #### norm
    os.chdir(path_memmap)
    tf_norm = np.memmap(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape))
    tf_mean_allchan = np.memmap(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(tf.shape[0], AL_n, tf.shape[1], resampled_points_AL))

    tf_norm[:] = norm_tf(sujet, tf, electrode_recording_type, norm_method)
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_tf_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    try:
        os.remove(f'{sujet}_tf_{cond}_norm_{electrode_recording_type}.dat')
    except:
        pass

    #### verif
    if debug:

        plt.pcolormesh(tf_mean_allchan[0,0,:,:])
        plt.show()

    return tf_mean_allchan





def compute_stretch_tf_itpc(tf, cond, respfeatures_allcond, stretch_point_TF, srate):
    
    tf_stretch, ratio = stretch_data(respfeatures_allcond[cond][0], stretch_point_TF, tf, srate)

    return tf_stretch



def chunk_stretch_tf_itpc_ac(sujet, tf, cond, ac_starts, srate):
    
    #### identify number stretch
    nb_ac = len(ac_starts)
    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    
    #### compute tf
    tf_stretch = np.zeros((nb_ac, tf.shape[0], int(stretch_point_TF_ac)), dtype='complex')

    for fi in range(tf.shape[0]):

        x = tf[fi,:]
        data_chunk = np.zeros(( len(ac_starts), stretch_point_TF_ac ), dtype='complex')

        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            data_chunk[start_i,:] = x[t_start: t_stop]

        tf_stretch[:,fi,:] = data_chunk

    return tf_stretch




def chunk_stretch_tf_itpc_sniff(sujet, tf, cond, sniff_starts, srate):
    
    #### identify number stretch
    nb_ac = len(sniff_starts)
    
    #### compute tf
    tf_stretch = np.zeros((nb_ac, tf.shape[0], int(stretch_point_TF_sniff_resampled)), dtype='complex')

    for fi in range(tf.shape[0]):

        x = tf[fi,:]
        data_chunk = np.zeros(( len(sniff_starts), stretch_point_TF_sniff_resampled ), dtype='complex')

        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_chunk[start_i,:] = x[t_start: t_stop]

        tf_stretch[:,fi,:] = data_chunk

    return tf_stretch






################################
######## PRECOMPUTE TF ########
################################


def precompute_tf_allconv(sujet, cond, electrode_recording_type):

    print('TF PRECOMPUTE', flush=True)

    #### verify if already computed
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(f'{sujet}_tf_{cond}.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}.npy'):
            print('ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(f'{sujet}_tf_{cond}_bi.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}_bi.npy'):
            print('ALREADY COMPUTED', flush=True)
            return

    #### params
    respfeatures_allcond = load_respfeatures(sujet)
    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    #### select data without aux chan
    if cond != 'AL':
        data = load_data(sujet, cond, electrode_recording_type)
        data = data[:len(chan_list_ieeg),:]
        
    else:
        data_AL = load_data(sujet, cond, electrode_recording_type)
        AL_len_list = np.array([data_AL[session_i].shape[-1] for session_i in range(len(data_AL))])
        data = np.zeros(( len(chan_list_ieeg), AL_len_list.sum() ))
        AL_pre, AL_post = 0, AL_len_list[0]
        #AL_i = 1
        for AL_i in range(AL_n):

            if AL_i != 0:
                AL_pre, AL_post = AL_pre + AL_len_list[AL_i-1], AL_post + AL_len_list[AL_i]

            data[:,AL_pre:AL_post] = data_AL[AL_i][:len(chan_list_ieeg),:]

    print('COMPUTE', flush=True)

    #### select wavelet parameters
    wavelets = get_wavelets()

    #### compute
    os.chdir(path_memmap)
    tf_allchan = np.memmap(f'{sujet}_tf_{cond}_precompute_convolutions_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, data.shape[1]))

    def compute_tf_convolution_nchan(n_chan):

        print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

        x = data[n_chan,:]

        tf = np.zeros((nfrex, x.shape[0]))

        for fi in range(nfrex):
            
            tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

        tf_allchan[n_chan,:,:] = tf

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

    del data

    #### stretch or chunk
    if cond == 'FR_CV':

        n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf_allchan[0,:,:], srate)[0].shape[0]
        tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), n_cycle_stretch, nfrex, stretch_point_TF))
    
        print('STRETCH_VS', flush=True)
        tf_allband_stretched[:] = compute_stretch_tf(sujet, tf_allchan, cond, respfeatures_allcond, stretch_point_TF, srate, electrode_recording_type)

    if cond == 'AC':

        ac_starts = get_ac_starts(sujet)
        tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), len(ac_starts), nfrex, stretch_point_TF_ac_resample))
        
        print('CHUNK_AC', flush=True)
        tf_allband_stretched[:] = compute_stretch_tf_AC(sujet, tf_allchan, ac_starts, srate, electrode_recording_type)

    if cond == 'SNIFF':
        
        sniff_starts = get_sniff_starts(sujet)
        tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), len(sniff_starts), nfrex, stretch_point_TF_sniff_resampled))

        print('CHUNK_SNIFF', flush=True)
        tf_allband_stretched[:] = compute_stretch_tf_SNIFF(sujet, tf_allchan, sniff_starts, srate, electrode_recording_type)

    if cond == 'AL':

        tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), AL_n, nfrex, resampled_points_AL))

        print('CHUNK_AL', flush=True)
        tf_allband_stretched[:] = compute_stretch_tf_AL(sujet, cond, tf_allchan, AL_len_list, srate, electrode_recording_type)

    if debug:

        plt.pcolormesh(np.median(tf_allband_stretched[0,:,:,:], axis=0))
        plt.show()
    
    #### save
    print('SAVE', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_{cond}.npy', tf_allband_stretched)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_{cond}_bi.npy', tf_allband_stretched)
    
    os.chdir(path_memmap)
    try:
        os.remove(f'{sujet}_tf_{cond}_precompute_convolutions_{electrode_recording_type}.dat')
    except:
        pass

    try:
        os.remove(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat')
    except:
        pass

    try:
        os.remove(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat')
    except:
        pass

    print('done', flush=True)






################################
######## PRECOMPUTE ITPC ########
################################



def precompute_itpc(sujet, cond, band_prep_list, electrode_recording_type):

    print('ITPC PRECOMPUTE', flush=True)

    respfeatures_allcond = load_respfeatures(sujet)
    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
    
    #### select prep to load
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = load_data(sujet, cond, electrode_recording_type)

        #### remove aux chan
        data = data[:len(chan_list_ieeg),:]

        freq_band = freq_band_list_precompute[band_prep_i]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if electrode_recording_type == 'monopolaire':
                if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy') :
                    print('ALREADY COMPUTED', flush=True)
                    continue
            if electrode_recording_type == 'bipolaire':
                if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy') :
                    print('ALREADY COMPUTED', flush=True)
                    continue
            
            print(band, ' : ', freq, flush=True)

            #### select wavelet parameters
            wavelets = get_wavelets()

            #### compute itpc
            print('COMPUTE, STRETCH & ITPC', flush=True)
            #n_chan = 0
            def compute_itpc_n_chan(n_chan):

                print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                x = data[n_chan,:]

                tf = np.zeros((nfrex, x.shape[0]), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                #### stretch
                if cond == 'FR_CV':
                    tf_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf, srate)[0]

                elif cond == 'AC':
                    ac_starts = get_ac_starts(sujet)
                    tf_stretch = chunk_stretch_tf_itpc_ac(sujet, tf, cond, ac_starts, srate)

                elif cond == 'SNIFF':
                    sniff_starts = get_sniff_starts(sujet)
                    tf_stretch = chunk_stretch_tf_itpc_sniff(sujet, tf, cond, sniff_starts, srate)

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

            compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(data.shape[0]))
            
            if cond == 'FR_CV':
                itpc_allchan = np.zeros((data.shape[0],nfrex,stretch_point_TF))

            elif cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
                itpc_allchan = np.zeros((data.shape[0], nfrex, stretch_point_TF_ac))

            elif cond == 'SNIFF':
                itpc_allchan = np.zeros((data.shape[0], nfrex, stretch_point_TF_sniff_resampled))

            for n_chan in range(data.shape[0]):

                itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

            #### save
            print('SAVE', flush=True)
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if electrode_recording_type == 'monopolaire':
                np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy', itpc_allchan)
            if electrode_recording_type == 'bipolaire':
                np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_bi.npy', itpc_allchan)
            
            del itpc_allchan

    print('done')













################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #electrode_recording_type = 'bipolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            #### compute and save tf
            #cond = 'AL'
            for cond in conditions:

                print(cond, flush=True)

                if cond == 'SNIFF':
                    precompute_tf_allconv(sujet, cond, electrode_recording_type)
                    # execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_tf_allconv', [sujet, cond, electrode_recording_type], '60G')
                elif cond == 'AC':
                    precompute_tf_allconv(sujet, cond, electrode_recording_type)
                    # execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_tf_allconv', [sujet, cond, electrode_recording_type], '50G')
                else:
                    # precompute_tf_allconv(sujet, cond, electrode_recording_type)
                    execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_tf_allconv', [sujet, cond, electrode_recording_type], '30G')
                    
                #precompute_itpc(sujet, cond, band_prep_list, electrode_recording_type)
                # execute_function_in_slurm_bash_mem_choice('n5_precompute_TF', 'precompute_itpc', [sujet, cond, band_prep_list, electrode_recording_type], '30G')
            
            #### compute sniff chunks
            #precompute_tf_sniff(sujet, 'SNIFF', band_prep_list)
            #execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf_sniff', [sujet, cond, band_prep_list, electrode_recording_type])



