


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n2bis_prep_load_export_nlx_trig import *

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet_i, band_prep, electrode_recording_type = 'pat_03083_1527', 'lf', 'bipolaire'
def compute_and_save_baseline(sujet_i, band_prep, electrode_recording_type):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if electrode_recording_type == 'monopolaire':
            if os.path.exists(os.path.join(path_precompute, sujet_i, 'baselines', f'{sujet_i}_{band}_baselines.npy')):
                verif_band_compute.append(True)
        if electrode_recording_type == 'bipolaire':
            if os.path.exists(os.path.join(path_precompute, sujet_i, 'baselines', f'{sujet_i}_{band}_baselines_bi.npy')):
                verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet_i} : BASELINES ALREADY COMPUTED')
        return
            
    #### open data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if electrode_recording_type == 'monopolaire':
        raw = mne.io.read_raw_fif(f'{sujet_i}_allcond_{band_prep}.fif', preload=True)
    if electrode_recording_type == 'bipolaire':
        raw = mne.io.read_raw_fif(f'{sujet_i}_allcond_{band_prep}_bi.fif', preload=True)

    #### Data vizualisation
    if debug == True :
        duration = 4.
        n_chan = 20
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    #### get correct chans
    chan_list_raw = raw.info['ch_names']
    header = chan_list_raw[0][:23]
    chan_list_raw = [nchan[23:] for nchan in chan_list_raw]
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_i, electrode_recording_type)

    drop_chan = [header + chan for chan in chan_list_raw if (chan in chan_list_ieeg) == False]

    #### remove unused chan
    raw.drop_channels(drop_chan)
    #raw.info # verify

    #### get raw params
    data = raw.get_data()
    srate = raw.info['sfreq']
    
    #### generate all wavelets to conv
    wavelets_to_conv = {}
            
    #band, freq = 'theta', [2, 10]
    for band, freq in freq_band_dict[band_prep].items():

        #### compute the wavelets
        wavelets_to_conv[band], nfrex = get_wavelets(sujet_i, band_prep, freq, electrode_recording_type)        

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### compute convolutions
    n_band_to_compute = len(list(freq_band_dict[band_prep].keys()))

    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet_i}_baseline_convolutions_{band_prep}_{electrode_recording_type}.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data.shape[0], nfrex))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

        #### load chunk indicies
        vs_starts, sniff_allsession, sniff_peaks, ac_allsession, ac_starts, al_allsession, al_starts, al_stops = get_trig_time_for_sujet(sujet)

        x = data[n_chan,:]
        #band_i, band = 0, list(wavelets_to_conv.keys())[0]
        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())
            #fi = 0
            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2
                #### chunk data
                fi_conv_chunked = np.concatenate((  fi_conv[int(vs_starts[0]*srate) : int(vs_starts[-1]*srate)],
                                                    fi_conv[int(sniff_allsession[0]*srate) : int(sniff_allsession[-1]*srate)],
                                                    fi_conv[int(ac_allsession[0]*srate) : int(ac_allsession[-1]*srate)],
                                                    fi_conv[int(al_allsession[0]*srate) : int(al_allsession[-1]*srate)],
                                                    ), axis=0)
                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv_chunked))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet_i, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        if electrode_recording_type == 'monopolaire':
            np.save(f'{sujet_i}_{band}_baselines.npy', baseline_allchan[band_i, :, :])
        if electrode_recording_type == 'bipolaire':
            np.save(f'{sujet_i}_{band}_baselines_bi.npy', baseline_allchan[band_i, :, :])

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet_i}_baseline_convolutions_{band_prep}_{electrode_recording_type}.dat')

    print('done')




################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### compute
    #compute_and_save_baseline(sujet, band_prep)
    
    #### slurm execution
    #electrode_recording_type = 'bipolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:
        #band_prep = 'lf'
        for band_prep in band_prep_list:
            execute_function_in_slurm_bash('n4_precompute_baselines', 'compute_and_save_baseline', [sujet, band_prep, electrode_recording_type])

