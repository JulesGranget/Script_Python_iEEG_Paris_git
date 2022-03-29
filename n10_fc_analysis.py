



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




#band_prep, freq, band, cond, prms = 'lf', [4, 8], 'theta', 'FR_CV', prms
def compute_fc_metrics_mat_AL(band_prep, freq, band, cond, prms):
    
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
    data_allAL = load_data(cond, band_prep=band_prep)

    #### prepare mat loading
    pli_mat_allAL = np.zeros((len(data_allAL),len(prms['chan_list_ieeg']),len(prms['chan_list_ieeg'])))
    ispc_mat_allAL = np.zeros((len(data_allAL),len(prms['chan_list_ieeg']),len(prms['chan_list_ieeg'])))

    for AL_i in range(len(data_allAL)):

        data = data_allAL[AL_i]

        #### get wavelets
        nfrex = get_nfrex(band_prep)
        wavelets = get_wavelets(band_prep, freq)

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
        pli_mat = np.zeros((len(prms['chan_list_ieeg']),len(prms['chan_list_ieeg'])))
        ispc_mat = np.zeros((len(prms['chan_list_ieeg']),len(prms['chan_list_ieeg'])))

        print('COMPUTE')

        for seed in range(len(prms['chan_list_ieeg'])) :

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
            for nchan in range(len(prms['chan_list_ieeg'])) :
                    
                if nchan == seed:

                    continue

                else:
                        
                    ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                    pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

        ispc_mat_allAL[AL_i,:,:] = ispc_mat
        pli_mat_allAL[AL_i,:,:] = pli_mat

    #### mean on all AL
    ispc_mat_allAL_mean = np.mean(ispc_mat_allAL, 0)
    pli_mat_allAL_mean = np.mean(pli_mat_allAL, 0)


    #### save matrix
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    np.save(f'{sujet}_ISPC_{band}_{cond}.npy', ispc_mat_allAL_mean)

    np.save(f'{sujet}_PLI_{band}_{cond}.npy', pli_mat_allAL_mean)

    #### supress mmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_{band_prep}_{band}_{cond}_fc_convolutions.dat')
        
    return pli_mat_allAL_mean, ispc_mat_allAL_mean

    








#band_prep, freq, band, cond, prms = 'lf', [4, 8], 'theta', 'FR_CV', prms
def compute_fc_metrics_mat(band_prep, freq, band, cond, prms):

    if cond == 'AL':

        pli_mat, ispc_mat = compute_fc_metrics_mat_AL(band_prep, freq, band, cond, prms)
        return pli_mat, ispc_mat
    
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
    nfrex = get_nfrex(band_prep)
    wavelets = get_wavelets(band_prep, freq)

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


    elif cond == 'AC':

        #### get ac_starts
        ac_starts = get_ac_starts(sujet)
        
        #### compute metrics
        pli_mat = np.zeros((np.size(data,0), np.size(data,0)))
        ispc_mat = np.zeros((np.size(data,0), np.size(data,0)))

        for seed in range(np.size(data,0)) :

            if seed/len(prms['chan_list_ieeg']) % .25 <= .01:
                print("{:.2f}".format(seed/len(prms['chan_list_ieeg'])))

            def compute_ispc_pli(nchan):

                if nchan == seed : 
                    return 
                    
                else :

                    # initialize output time-frequency data
                    ispc = np.zeros((len(ac_starts), nfrex))
                    pli  = np.zeros((len(ac_starts), nfrex))

                    for ac_i, ac_start_i in enumerate(ac_starts):

                        #### chunk conv
                        t_start = int(ac_start_i + t_start_AC*prms['srate'])
                        t_stop = int(ac_start_i + t_stop_AC*prms['srate'])              

                        # compute metrics
                        for fi in range(nfrex):
                            
                            as1 = convolutions[seed][fi, t_start:t_stop]
                            as2 = convolutions[nchan][fi, t_start:t_stop]

                            # collect "eulerized" phase angle differences
                            cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                            
                            # compute ISPC and PLI (and average over trials!)
                            ispc[ac_i, fi] = np.abs(np.mean(cdd))
                            pli[ac_i, fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

                    # compute mean
                    mean_ispc = np.mean(ispc)
                    mean_pli = np.mean(pli)

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







def get_pli_ispc_allsession(sujet):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, 'AC'
                for cond_i, cond in enumerate(conditions_FC) :

                    print(band, cond)

                    pli_mat, ispc_mat = compute_fc_metrics_mat(band_prep, freq, band, cond, prms)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif conv
    if debug == True:
        for band in pli_allband.keys():
            for cond in pli_allband[band].keys():
                plt.matshow(pli_allband[band][cond][0])
                plt.show()

    #### verif

    if debug == True:
                
        for band_prep in band_prep_list:
            
            for band, freq in freq_band_dict_FC[band_prep].items():

                for cond_i, cond in enumerate(conditions_FC) :

                    print(band, cond, len(pli_allband[band][cond]))
                    print(band, cond, len(ispc_allband[band][cond]))


        #### reduce to one cond
    #### generate dict to fill
    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole':

                continue

            ispc_allband_reduced[band] = {}
            pli_allband_reduced[band] = {}

            for cond_i, cond in enumerate(conditions_FC) :

                ispc_allband_reduced[band][cond] = []
                pli_allband_reduced[band][cond] = []

    #### fill
    
    for band_prep_i, band_prep in enumerate(band_prep_list):

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole' :

                continue

            else:

                for cond_i, cond in enumerate(conditions_FC) :

                    ispc_allband_reduced[band][cond] = ispc_allband[band][cond][0]
                    pli_allband_reduced[band][cond] = pli_allband[band][cond][0]


    return pli_allband_reduced, ispc_allband_reduced















################################
######## SAVE FIG ########
################################

def save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, prms):

    print('######## SAVEFIG FC ########')

    #### sort matrix
    #mat = ispc_allband_reduced[band][cond]
    def sort_mat(mat):

        mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
        for i_before_sort_r, i_sort_r in enumerate(df_sorted.index.values):
            for i_before_sort_c, i_sort_c in enumerate(df_sorted.index.values):
                mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

        return mat_sorted

    #### verify sorting
    #mat = pli_allband_reduced.get(band).get(cond)
    #mat_sorted = sort_mat(mat)
    #plt.matshow(mat_sorted)
    #plt.show()

    #### prepare sort
    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    chan_name_sorted_mat = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            chan_name_sorted_mat.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_mat.append('')
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_mat.append(name_i)
                rep_count = 0
                continue
                

    #### identify scale
    scale = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}

    scale['ispc']['max'] = {}
    scale['ispc']['min'] = {}
    scale['pli']['max'] = {}
    scale['pli']['min'] = {}

    for band_prep in band_prep_list:

        for band, freq in freq_band_dict_FC[band_prep].items():

            if band == 'whole':
                continue

            band_ispc = {'min' : [], 'max' : []}
            band_pli = {'min' : [], 'max' : []}

            for cond_i, cond in enumerate(conditions_FC):
                band_ispc['max'].append(np.max(ispc_allband_reduced[band][cond]))
                band_ispc['min'].append(np.min(ispc_allband_reduced[band][cond]))
                
                band_pli['max'].append(np.max(pli_allband_reduced[band][cond]))
                band_pli['min'].append(np.min(pli_allband_reduced[band][cond]))

            scale['ispc']['max'][band] = np.max(band_ispc['max'])
            scale['ispc']['min'][band] = np.min(band_ispc['min'])
            scale['pli']['max'][band] = np.max(band_pli['max'])
            scale['pli']['min'][band] = np.min(band_pli['min'])


    #### ISPC
    nrows, ncols = 1, len(conditions_FC)

    #band_prep, band = 'lf', 'theta'
    for band_prep in band_prep_list:

        for band in ispc_allband_reduced.keys():

            #### graph
            fig = plt.figure(facecolor='black')
            for cond_i, cond in enumerate(conditions_FC):
                mne.viz.plot_connectivity_circle(sort_mat(ispc_allband_reduced[band][cond]), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
            plt.suptitle('ISPC_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'figures'))

            fig.savefig(sujet + '_ISPC_' + band + '_graph.jpeg', dpi = 100)

            plt.close()

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))
                
            for c, cond_i in enumerate(conditions_FC):
                ax = axs[c]
                ax.matshow(sort_mat(ispc_allband_reduced[band][cond_i]), vmin=scale['ispc']['min'][band], vmax=scale['ispc']['max'][band])
                ax.set_title(cond_i)
                ax.set_yticks(range(len(chan_name_sorted)))
                ax.set_yticklabels(chan_name_sorted)
                        
            plt.suptitle('ISPC_' + band)
            #plt.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC', 'matrix'))
                        
            fig.savefig(sujet + '_ISPC_' + band + '_mat.jpeg', dpi = 100)

            plt.close()


    #### PLI
    nrows, ncols = 1, len(conditions_FC)

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band in ispc_allband_reduced.keys():

            #### graph
            fig = plt.figure(facecolor='black')
            for cond_i, cond in enumerate(conditions_FC):
                mne.viz.plot_connectivity_circle(sort_mat(pli_allband_reduced[band][cond]), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
            plt.suptitle('PLI_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'figures'))

            fig.savefig(sujet + '_PLI_' + band + '_graph.jpeg', dpi = 100)

            plt.close()

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

            for c, cond_i in enumerate(conditions_FC):
                ax = axs[c]
                ax.matshow(sort_mat(pli_allband_reduced[band][cond_i]), vmin=scale['pli']['min'][band], vmax=scale['pli']['max'][band])
                ax.set_title(cond_i)
                ax.set_yticks(range(len(chan_name_sorted)))
                ax.set_yticklabels(chan_name_sorted)
                        
            plt.suptitle('PLI_' + band)
            #plt.show()

            os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI', 'matrix'))
                        
            fig.savefig(sujet + '_PLI_' + band + '_mat.jpeg', dpi = 100)

            plt.close()






def save_fig_for_allsession(sujet):

    prms = get_params(sujet)

    df_loca = get_loca_df(sujet)

    pli_allband_reduced, ispc_allband_reduced = get_pli_ispc_allsession(sujet)

    save_fig_FC(pli_allband_reduced, ispc_allband_reduced, df_loca, prms)




################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #### params
    compute_metrics = False
    plot_fig = True

    #### compute fc metrics
    if compute_metrics:
        #compute_pli_ispc_allband(sujet)
        execute_function_in_slurm_bash('n10_fc_analysis', 'compute_pli_ispc_allband', [sujet])

    #### save fig
    if plot_fig:

        save_fig_for_allsession(sujet)