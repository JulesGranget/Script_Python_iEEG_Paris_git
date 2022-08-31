


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

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet_i, band_prep = 'pat_03083_1527', 'lf'
def compute_and_save_baseline(sujet_i, band_prep):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if os.path.exists(os.path.join(path_precompute, sujet_i, 'baselines', f'{sujet_i}_{band}_baselines.npy')):
            verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet_i} : BASELINES ALREADY COMPUTED')
        return
            

    #### open raw
    os.chdir(os.path.join(path_raw, sujet_i, 'raw_data', 'mat'))
    raw = mne.io.read_raw_eeglab(f'{sujet_i}_allchan.set', preload=True)

    #### Data vizualisation
    if debug == True :
        duration = 4.
        n_chan = 20
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    #### get correct chans
    chan_list_raw = raw.info['ch_names']
    header = chan_list_raw[0][:23]
    chan_list_raw = [nchan[23:] for nchan in chan_list_raw]
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_i)

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
        wavelets_to_conv[band], nfrex = get_wavelets(sujet_i, band_prep, freq)        

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
    baseline_allchan = np.memmap(f'{sujet_i}_baseline_convolutions_{band_prep}.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data.shape[0], nfrex))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

        x = data[n_chan,:]

        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())

            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2
                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet_i, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        np.save(f'{sujet_i}_{band}_baselines.npy', baseline_allchan[band_i, :, :])

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet_i}_baseline_convolutions_{band_prep}.dat')

    print('done')




################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### compute
    #compute_and_save_baseline(sujet, band_prep)
    
    #### slurm execution
    for band_prep in band_prep_list:
        execute_function_in_slurm_bash('n4_precompute_baselines', 'compute_and_save_baseline', [sujet, band_prep])

