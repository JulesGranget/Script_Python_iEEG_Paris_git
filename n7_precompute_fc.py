



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




#######################################
############# ISPC & PLI #############
#######################################




#band_prep, freq, band, cond, prms = 'lf', [4, 8], 'theta', 'FR_CV', prms
def compute_fc_metrics_mat(band_prep, freq, band, cond, prms):

    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    if os.path.exists(f'{sujet}_ISPC_{band}_{cond}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_ISPC_{band}_{cond}')
        ispc_mat = np.load(f'{sujet}_ISPC_{band}_{cond}.npy')

    if os.path.exists(f'{sujet}_PLI_{band}_{cond}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_PLI_{band}_{cond}')
        pli_mat = np.load(f'{sujet}_PLI_{band}_{cond}.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    
    #### load_data
    data = load_data(cond, band_prep=band_prep)

    #### get wavelets
    wavelets, nfrex = get_wavelets(band_prep, freq)

    #### get only EEG chan
    data = data[:len(prms['chan_list_ieeg']),:]

    #### compute all convolution
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

    print('CONV')

    def convolution_x_wavelets_nchan(nchan):

        if nchan/np.size(data,0) % .25 <= .01:
            print("{:.2f}".format(nchan/len(prms['chan_list_ieeg'])))
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data,0)))

    #### verif conv
    if debug:
        for nchan in range(10):
            for i in range(5):
                i = nchan*5 + i
                plt.plot(np.mean(np.real(convolutions[i]), 0))
            plt.show()


    if cond == 'FR_CV':

        #### compute metrics
        pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
        ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

        print('COMPUTE')

        for seed in range(np.size(data,0)) :

            if seed/len(prms['chan_list_ieeg']) % .25 <= .01:
                print("{:.2f}".format(seed/len(prms['chan_list_ieeg'])))

            def compute_ispc_pli(nchan):

                if nchan == seed : 
                    return 
                    
                else :

                    # initialize output time-frequency data
                    ispc = np.zeros((nfrex))
                    pli  = np.zeros((nfrex))

                    # compute metrics
                    for fi in range(nfrex):
                        
                        as1 = convolutions[seed][fi,:]
                        as2 = convolutions[nchan][fi,:]

                        # collect "eulerized" phase angle differences
                        cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                        
                        # compute ISPC and PLI (and average over trials!)
                        ispc[fi] = np.abs(np.mean(cdd))
                        pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

                    # compute mean
                    mean_ispc = np.mean(ispc,0)
                    mean_pli = np.mean(pli,0)

                    return mean_ispc, mean_pli

            compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(nchan) for nchan in range(np.size(data,0)))
            
            #### load in mat    
            for nchan in range(np.size(data,0)) :
                    
                if nchan == seed:

                    continue

                else:
                        
                    ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                    pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

        #### save matrix
        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        np.save(f'{sujet}_ISPC_{band}_{cond}.npy', ispc_mat)

        np.save(f'{sujet}_PLI_{band}_{cond}.npy', pli_mat)

        #### supress mmap
        os.chdir(path_memmap)
        os.remove(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat')
        
        return pli_mat, ispc_mat


    

    




#session_eeg=0
def compute_pli_ispc_allband(sujet):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band, freq = 'theta', [2, 10]
        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, 'FR_CV'
                for cond_i, cond in enumerate(conditions_FC) :

                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]


                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif mat
    if debug:
        for band in pli_allband.keys():
            for cond in pli_allband[band].keys():
                plt.matshow(pli_allband[band][cond][0])
                plt.show()
    #### verif

    if debug == True:
                
        for band, freq in freq_band_fc_analysis.items():

            for cond_i, cond in enumerate(conditions_FC) :

                print(band, cond, len(pli_allband[band][cond]))
                print(band, cond, len(ispc_allband[band][cond]))

    print('done')






################################
######## EXECUTE ########
################################



if __name__ == '__main__':


    #compute_pli_ispc_allband(sujet)
    execute_function_in_slurm_bash('n7_fc_analysis', 'compute_pli_ispc_allband', [sujet])

