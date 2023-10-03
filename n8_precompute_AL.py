

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






################################
######## COMPUTE TF ########
################################



def precompute_tf_AL(sujet, session_i, electrode_recording_type):

    cond = 'AL'

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'{sujet}_tf_{cond}_{str(session_i+1)}.npy')):
            print('ALREADY COMPUTED')
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'{sujet}_tf_{cond}_{str(session_i+1)}_bi.npy')):
            print('ALREADY COMPUTED')

    print('TF PRECOMPUTE')

    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    #### select data without aux chan
    data_allsession = load_data(sujet, cond, electrode_recording_type)

    data = data_allsession[session_i][:len(chan_list_ieeg),:]
    
    #### select wavelet parameters
    wavelets = get_wavelets()

    os.chdir(path_memmap)
    tf_conv = np.memmap(f'{sujet}_{cond}_{session_i}_tf_conv_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', 
                            shape=(len(chan_list_ieeg), nfrex, data.shape[-1]))

    def compute_tf_convolution_nchan(n_chan):

        print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

        x = data[n_chan,:]

        tf = np.zeros((nfrex,np.size(x)))

        for fi in range(nfrex):
            
            tf_conv[n_chan, fi, :] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

    tf_conv = norm_tf(sujet, tf_conv, electrode_recording_type, norm_method)

    if debug:

        plt.pcolormesh(tf_conv[0,:,:int(data.shape[-1]/2)])
        plt.show()

    #### resample TF AL
    f = scipy.interpolate.interp1d(np.linspace(0, 1, tf_conv.shape[-1]), tf_conv, kind='linear')
    tf_resampled = f(np.linspace(0, 1, resampled_points_AL))

    if debug:

        plt.pcolormesh(tf_resampled[0,:,:int(data.shape[-1]/2)])
        plt.show()

        plt.plot(tf_resampled[0,-1,:], label='raw')
        plt.plot(tf_resampled[0,-1,:], label='resampled')
        plt.legend()
        plt.show()

    #### export TF AL
    
    if electrode_recording_type == 'monopolaire':
        np.save(f'{sujet}_tf_{cond}_{str(session_i+1)}.npy', tf_resampled)
    if electrode_recording_type == 'bipolaire':
        np.save(f'{sujet}_tf_{cond}_{str(session_i+1)}_bi.npy', tf_resampled)    

    os.chdir(path_memmap)
    
    try:
        os.remove(f'{sujet}_{cond}_{session_i}_tf_conv_{electrode_recording_type}.dat')
    except:
        pass

    print('done')







################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:
            
        #electrode_recording_type = 'monopolaire'
        for electrode_recording_type in ['monopolaire', 'bipolaire']:

            #### load n_session
            n_session = 3

            #### compute and save tf
            #session_i = 0
            for session_i in range(n_session):
                #precompute_tf(sujet, cond, session_i, band_prep_list)
                execute_function_in_slurm_bash_mem_choice('n8_precompute_AL', 'precompute_tf_AL', [sujet, session_i, electrode_recording_type], '20G')
        









