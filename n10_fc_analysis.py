



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




#######################################
############# ISPC & PLI #############
#######################################

#session_eeg, band_prep, freq, band, cond, session_i, prms = 0, 'wb', [2, 10], 'theta', 'FR_CV', 0, prms
def compute_fc_metrics_mat(band_prep, freq, band, cond, prms):
    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    if os.path.exists(f'{sujet}_ISPC_{band}_{cond}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_ISPC_{band}_{cond}')
        ispc_mat = np.load(f'{sujet}_ISPC_{band}_{cond}.npy')

    if os.path.exists(f'{sujet}_PLI_{band}_{cond}_.npy'):
        print(f'ALREADY COMPUTED : {sujet}_PLI_{band}_{cond}')
        pli_mat = np.load(f'{sujet}_PLI_{band}_{cond}.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    
    #### load_data
    data = load_data(cond, band_prep=band_prep)

    #### select wavelet parameters
    if band_prep == 'wb':
        wavetime = np.arange(-2,2,1/prms['srate'])
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex) 

    if band_prep == 'lf':
        wavetime = np.arange(-2,2,1/prms['srate'])
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/prms['srate'])
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

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

        #band = 'theta'
        for band, freq in freq_band_dict[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, conditions[0]
                #session_i = 0
                for cond_i, cond in enumerate(conditions_compute_TF) :

                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]


                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif

    if debug == True:
                
        for band, freq in freq_band_fc_analysis.items():

            for cond_i, cond in enumerate(conditions_compute_TF) :

                print(band, cond, len(pli_allband[band][cond]))
                print(band, cond, len(ispc_allband[band][cond]))







def get_pli_ispc_allsession(sujet):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band, freq in freq_band_dict[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, conditions[0]
                #session_i = 0
                for cond_i, cond in enumerate(conditions_compute_TF) :

                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif

    if debug == True:
                
        for band_prep in band_prep_list:
            
            for band, freq in freq_band_dict[band_prep].items():

                for cond_i, cond in enumerate(conditions_compute_TF) :

                    print(band, cond, len(pli_allband[band][cond]))
                    print(band, cond, len(ispc_allband[band][cond]))


        #### reduce to one cond
    #### generate dict to fill
    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            ispc_allband_reduced[band] = {}
            pli_allband_reduced[band] = {}

            for cond_i, cond in enumerate(conditions_compute_TF) :

                ispc_allband_reduced[band][cond] = []
                pli_allband_reduced[band][cond] = []

    #### fill
    
    for band_prep_i, band_prep in enumerate(band_prep_list):

        for band, freq in freq_band_dict[band_prep].items():

            if band == 'whole' :

                continue

            else:

                for cond_i, cond in enumerate(conditions_compute_TF) :

                    ispc_allband_reduced[band][cond] = ispc_allband[band][cond][0]
                    pli_allband_reduced[band][cond] = pli_allband[band][cond][0]


    return pli_allband_reduced, ispc_allband_reduced



################################
######## SAVE FIG ########
################################

def save_fig_FC(pli_allband_reduced, ispc_allband_reduced, prms):

    print('######## SAVEFIG FC ########')

    #### sort matrix

    #def sort_mat(mat):

    #    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    #    for i_before_sort_r, i_sort_r in enumerate(df_sorted.index.values):
    #        for i_before_sort_c, i_sort_c in enumerate(df_sorted.index.values):
    #            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    #    return mat_sorted

    #### verify sorting
    #mat = pli_allband_reduced.get(band).get(cond)
    #mat_sorted = sort_mat(mat)
    #plt.matshow(mat_sorted)
    #plt.show()

    #### prepare sort
    #df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    #chan_name_sorted = df_sorted['ROI'].values.tolist()


    #chan_name_sorted_mat = []
    #rep_count = 0
    #for i, name_i in enumerate(chan_name_sorted):
    #    if i == 0:
    #        chan_name_sorted_mat.append(name_i)
    #        continue
    #    else:
    #        if name_i == chan_name_sorted[i-(rep_count+1)]:
    #            chan_name_sorted_mat.append('')
    #            rep_count += 1
    #            continue
    #        if name_i != chan_name_sorted[i-(rep_count+1)]:
    #            chan_name_sorted_mat.append(name_i)
    #            rep_count = 0
    #            continue
    #            

    #### identify scale
    scale = {}

    scale = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}

    scale['ispc']['max'] = {}
    scale['ispc']['min'] = {}
    scale['pli']['max'] = {}
    scale['pli']['min'] = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            band_ispc = {'min' : [], 'max' : []}
            band_pli = {'min' : [], 'max' : []}

            for cond_i, cond in enumerate(conditions_compute_TF):
                band_ispc['max'].append(np.max(ispc_allband_reduced[band][cond]))
                band_ispc['min'].append(np.min(ispc_allband_reduced[band][cond]))
                
                band_pli['max'].append(np.max(pli_allband_reduced[band][cond]))
                band_pli['min'].append(np.min(pli_allband_reduced[band][cond]))

            scale['ispc']['max'][band] = np.max(band_ispc['max'])
            scale['ispc']['min'][band] = np.min(band_ispc['min'])
            scale['pli']['max'][band] = np.max(band_pli['max'])
            scale['pli']['min'][band] = np.min(band_pli['min'])


    #### ISPC

    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC'))

    nrows, ncols = 1, len(conditions_compute_TF)

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            #### graph
            fig = plt.figure(facecolor='black')
            for cond_i, cond in enumerate(conditions_compute_TF):
                mne.viz.plot_connectivity_circle(ispc_allband_reduced[band][cond], node_names=prms['chan_list_ieeg'], n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
            plt.suptitle('ISPC_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            fig.savefig(sujet + '_ISPC_' + band + '_graph.jpeg', dpi = 100)

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))
                
            for c, cond_i in enumerate(conditions_compute_TF):
                ax = axs[c]
                ax.matshow(ispc_allband_reduced[band][cond], vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                ax.set_title(cond)
                ax.set_yticks(range(len(prms['chan_list_ieeg'])))
                ax.set_yticklabels(prms['chan_list_ieeg'])
                        
            plt.suptitle('ISPC_' + band)
            #plt.show()
                        
            fig.savefig(sujet + '_ISPC_' + band + '_mat.jpeg', dpi = 100)


    #### PLI

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI'))

    nrows, ncols = 1, len(conditions_compute_TF)

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            #### graph
            fig = plt.figure(facecolor='black')
            for cond_i, cond in enumerate(conditions_compute_TF):
                mne.viz.plot_connectivity_circle(pli_allband_reduced[band][cond], node_names=prms['chan_list_ieeg'], n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
            plt.suptitle('PLI_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            fig.savefig(sujet + '_PLI_' + band + '_graph.jpeg', dpi = 100)

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

            for c, cond_i in enumerate(conditions_compute_TF):
                ax = axs[c]
                ax.matshow(pli_allband_reduced[band][cond], vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                ax.set_title(cond)
                ax.set_yticks(range(len(prms['chan_list_ieeg'])))
                ax.set_yticklabels(prms['chan_list_ieeg'])
                        
            plt.suptitle('PLI_' + band)
            #plt.show()
                        
            fig.savefig(sujet + '_PLI_' + band + '_mat.jpeg', dpi = 100)






def save_fig_for_allsession(sujet):

    prms = get_params(sujet)

    pli_allband_reduced, ispc_allband_reduced = get_pli_ispc_allsession(sujet)

    save_fig_FC(pli_allband_reduced, ispc_allband_reduced, prms)




################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #### params
    compute_metrics = True
    plot_fig = False

    #### compute fc metrics
    if compute_metrics:
        #compute_pli_ispc_allband(sujet)
        execute_function_in_slurm_bash('n10_fc_analysis', 'compute_pli_ispc_allband', [sujet])

    #### save fig
    if plot_fig:

        save_fig_for_allsession(sujet)