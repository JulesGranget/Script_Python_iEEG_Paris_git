

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import joblib

from n0_config import *

debug = False

########################################
######## COMPUTE BASELINE ######## 
########################################


def compute_baseline(data, srate):

    print('#### COMPUTE BASELINES ####')
    
    #### generate all wavelets to conv
    wavelets_to_conv = {}
    
    #band_prep, band_prep_i = 'lf', 0
    for band_prep_i, band_prep in enumerate(band_prep_list):
        
        #### select wavelet parameters
        if band_prep == 'lf':
            wavetime = np.arange(-2,2,1/srate)
            nfrex = nfrex_lf
            ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

        if band_prep == 'hf':
            wavetime = np.arange(-.5,.5,1/srate)
            nfrex = nfrex_hf
            ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

        #band, freq = 'theta', [2, 10]
        for band, freq in freq_band_list[band_prep_i].items():

            #### compute wavelets
            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle_list[fi] / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()

            wavelets_to_conv[band] = wavelets

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

        #### count frequencies to compute
    n_fi2conv = 0
    for band in list(wavelets_to_conv.keys()):
        n_fi2conv += wavelets_to_conv[band].shape[0]

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        if n_chan/np.size(data,0) % .2 <= .01:
            print("{:.2f}".format(n_chan/np.size(data,0)))

        x = data[n_chan,:]

        baseline_coeff = np.array(())

        for band in list(wavelets_to_conv.keys()):

            for fi in range(wavelets_to_conv[band].shape[0]):
    
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2
                baseline_coeff = np.append(baseline_coeff, np.median(fi_conv))

        return baseline_coeff

    baselines_i = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data,0)))

    return baselines_i





################################
######## COMPUTE ########
################################

#sujet = 'pat_02459_0912'
#sujet = 'pat_02476_0929'
sujet = 'pat_02495_0949'

os.chdir(os.path.join(pos.chdir(path_main_workdir)ath_prep, sujet, 'sections'))


file_selection = [file_i for file_i in os.listdir() if file_i.find('lf') != -1]
baselines = []

for file_i in file_selection:

    #### open
    raw_allchan = mne.io.read_raw_fif(os.path.join(path_prep, sujet, 'sections', file_i), preload=True)
    data = raw_allchan.get_data()[:-4,:]
    chan_list = raw_allchan.info['ch_names'][:-4]
    srate = int(raw_allchan.info['sfreq'])

    baselines_i = compute_baseline(data, srate)
    baselines.append(baselines_i)

baselines_whole = np.zeros((len(baselines), len(baselines[0]), len(baselines[0][0])))

for block_i in range(len(baselines)):
    for nchan in range(len(baselines[0])):
        baselines_whole[block_i,nchan,:] = baselines[block_i][nchan]

baselines_mean = np.mean(baselines_whole,0)


os.chdir(os.path.join(path_prep, sujet, 'baseline'))
np.save(sujet + '_baselines.npy', baselines_mean)



