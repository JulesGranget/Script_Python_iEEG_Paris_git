


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


def compute_and_save_baseline(sujet, electrode_recording_type):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    if electrode_recording_type == 'monopolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_baselines.npy')):
            verif_band_compute.append(True)
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_baselines_bi.npy')):
            verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet} : BASELINES ALREADY COMPUTED')
        return
            
    #### open data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    if electrode_recording_type == 'monopolaire':
        raw = mne.io.read_raw_fif(f'{sujet}_allcond.fif', preload=True)
    if electrode_recording_type == 'bipolaire':
        raw = mne.io.read_raw_fif(f'{sujet}_allcond_bi.fif', preload=True)

    #### Data vizualisation
    if debug == True :
        duration = 4.
        n_chan = 20
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    #### get correct chans
    chan_list_raw = raw.info['ch_names']
    header = chan_list_raw[0][:23]
    chan_list_raw = [nchan[23:] for nchan in chan_list_raw]
    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    drop_chan = [header + chan for chan in chan_list_raw if (chan in chan_list_ieeg) == False]

    #### remove unused chan
    raw.drop_channels(drop_chan)
    #raw.info # verify

    #### get raw params
    data = raw.get_data()
    data = data[:len(chan_list_ieeg),:]
    srate = raw.info['sfreq']
    
    #### generate all wavelets to conv
    wavelets = get_wavelets()      

    #### compute convolutions
    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_baseline_convolutions_{electrode_recording_type}.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, 4))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, np.size(data,0), steps=[25, 50, 75])

        #### load chunk indicies
        vs_starts, sniff_allsession, sniff_peaks, ac_allsession, ac_starts, al_allsession, al_starts, al_stops = get_trig_time_for_sujet(sujet)

        x = data[n_chan,:]

        baseline_coeff = np.zeros((frex.shape[0], 4))

        #fi = 0
        for fi in range(nfrex):
            
            fi_conv = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2
            #### chunk data
            fi_conv_chunked = np.concatenate((  fi_conv[int(vs_starts[0]*srate) : int(vs_starts[-1]*srate)],
                                                fi_conv[int(sniff_allsession[0]*srate) : int(sniff_allsession[-1]*srate)],
                                                fi_conv[int(ac_allsession[0]*srate) : int(ac_allsession[-1]*srate)],
                                                fi_conv[int(al_allsession[0]*srate) : int(al_allsession[-1]*srate)],
                                                ), axis=0)
    
            baseline_coeff[fi,0] = np.mean(fi_conv_chunked)
            baseline_coeff[fi,1] = np.std(fi_conv_chunked)
            baseline_coeff[fi,2] = np.median(fi_conv_chunked)
            baseline_coeff[fi,3] = np.median(np.abs(x-np.median(fi_conv_chunked))) * 1.4826

        if debug:

            fig, axs = plt.subplots(ncols=2)
            axs[0].set_title('mean std')
            axs[0].plot(baseline_coeff[:,0], label='mean')
            axs[0].plot(baseline_coeff[:,1], label='std')
            axs[0].legend()
            axs[0].set_yscale('log')
            axs[1].set_title('median mad')
            axs[1].plot(baseline_coeff[:,2], label='median')
            axs[1].plot(baseline_coeff[:,3], label='mad')
            axs[1].legend()
            axs[1].set_yscale('log')
            plt.show()
    

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    #### save baseline
    xr_dict = {'chan' : chan_list_ieeg, 'frex' : range(frex.shape[0]), 'metrics' : ['mean', 'std', 'median', 'mad']}
    xr_baseline = xr.DataArray(baseline_allchan, dims=xr_dict.keys(), coords=xr_dict.values())
    
    if electrode_recording_type == 'monopolaire':
        xr_baseline.to_netcdf(f'{sujet}_baselines.nc')
    if electrode_recording_type == 'bipolaire':
        xr_baseline.to_netcdf(f'{sujet}_baselines_bi.nc')

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_baseline_convolutions_{electrode_recording_type}.dat')

    print('done')




################################
######## EXECUTE ########
################################


if __name__== '__main__':
    
    #### slurm execution
    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:
                
            execute_function_in_slurm_bash('n5_precompute_baselines', 'compute_and_save_baseline', [sujet, electrode_recording_type])

