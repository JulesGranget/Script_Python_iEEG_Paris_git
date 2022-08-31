
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config_params import *
from n0bis_config_analysis_functions import *

import joblib

debug = False






################################################
######## CXY CYCLE FREQ SURROGATES ########
################################################




def precompute_surrogates_coh(sujet, band_prep, cond, session_i):
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    
    print(cond)

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy') == True :
        print('ALREADY COMPUTED')
        return

    if cond == 'FR_MV':
        respi_i = chan_list.index('ventral')
    else:
        respi_i = chan_list.index('nasal')

    respi = data_tmp[respi_i,:]

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    surrogates_n_chan = np.zeros((np.size(data_tmp,0),len(hzCxy)))

    def compute_surrogates_coh_n_chan(n_chan):

        print_advancement(n_chan, np.size(data_tmp,0), steps=[25, 50, 75])

        x = data_tmp[n_chan,:]
        y = respi

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))

        for surr_i in range(n_surrogates_coh):

            x_shift = shuffle_Cxy(x)
            #y_shift = shuffle_Cxy(y)
            hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

            surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
        compute_surrogates_coh_tmp = surrogates_val_tmp_sorted[percentile_i,:]

        return compute_surrogates_coh_tmp
    
    compute_surrogates_coh_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_coh_n_chan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[n_chan,:] = compute_surrogates_coh_results[n_chan]

    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy', surrogates_n_chan)

    print('done')



def precompute_surrogates_cyclefreq(sujet, band_prep, cond, session_i):
    
    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy') == True :
        print('ALREADY COMPUTED')
        return

    #### compute surrogates
    surrogates_n_chan = np.zeros((3, data_tmp.shape[0], stretch_point_surrogates))

    respfeatures_i = respfeatures_allcond[cond][session_i]

    def compute_surrogates_cyclefreq_nchan(n_chan):

        print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

        x = data_tmp[n_chan,:]

        surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

        for surr_i in range(n_surrogates_cyclefreq):

            # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

            x_shift = shuffle_Cxy(x)
            #y_shift = shuffle_Cxy(y)

            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

            x_stretch_mean = np.mean(x_stretch, axis=0)

            surrogates_val_tmp[surr_i,:] = x_stretch_mean

        mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
        percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

        up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
        dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

        return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    #### fill results
    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[0, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][0]
        surrogates_n_chan[1, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][1]
        surrogates_n_chan[2, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][2]
    
    #### save
    np.save(f'{sujet}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy', surrogates_n_chan)

    print('done')






################################
######## MI / MVL ########
################################




#x = x_stretch_linear
def shuffle_windows(x):

    n_cycles_stretch = int( x.shape[0]/stretch_point_surrogates )

    shuffle_win = np.zeros(( n_cycles_stretch, stretch_point_surrogates ))

    for cycle_i in range(n_cycles_stretch):

        cut_i = np.random.randint(0, x.shape[0]-stretch_point_surrogates, 1)
        shuffle_win[cycle_i,:] = x[int(cut_i):int(cut_i+stretch_point_surrogates)]

    x_shuffled = np.mean(shuffle_win, axis=0)

    if debug:
        plt.plot(x_shuffled)
        plt.show()

    return x_shuffled



def precompute_MI(sujet, band_prep, cond, session_i):

    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)
    respfeatures_i = respfeatures_allcond[cond][session_i]
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_MI_{band_prep}.npy') == True :
        print('ALREADY COMPUTED')
        return

    #### compute surrogates
    #n_chan = 95
    def compute_surrogates_cyclefreq_nchan(n_chan):

        print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

        x = data_tmp[n_chan,:]
        x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x, srate)
        x_stretch_linear = x_stretch.reshape(-1) 

        surrogates_stretch_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

        for surr_i in range(n_surrogates_cyclefreq):

            # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

            surrogates_stretch_tmp[surr_i,:] = shuffle_windows(x_stretch_linear)

        #### compute MI
        MI_surrogates_i = np.array([])
        MI_bin_i = int(stretch_point_surrogates / MI_n_bin)
        x_bin_surr = np.zeros(( stretch_point_surrogates, MI_n_bin ))
        for surr_i in range(n_surrogates_cyclefreq):

            x = surrogates_stretch_tmp[surr_i,:]

            x_bin = np.zeros(( MI_n_bin ))

            for bin_i in range(MI_n_bin):
                x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

            # x += np.abs(x.min())*2
            # x = x/np.sum(x)
            x_bin += np.abs(x_bin.min())*2
            x_bin = x_bin/np.sum(x_bin)

            x_bin_surr[surr_i, :] = x_bin
            
            MI_surrogates_i = np.append(MI_surrogates_i, Shannon_MI(x_bin))

        if debug:
            times_binned = np.arange(int(stretch_point_surrogates/MI_n_bin), stretch_point_surrogates, int(stretch_point_surrogates/MI_n_bin))
            _99th = np.percentile(MI_surrogates_i, 99) 
            plot_i = np.where(MI_surrogates_i > _99th)[0]
            for i in plot_i:
                plt.plot(np.mean(x_stretch,axis=0), label='original')
                plt.plot(x_bin_surr[i,:], label='shuffle')
                plt.title(f'MI : {MI_surrogates_i[i]}')
                plt.legend()
                plt.show()

            for i in range(n_surrogates_cyclefreq):
                plt.plot(surrogates_stretch_tmp[i,:])
            plt.plot(np.mean(x_stretch,axis=0), linewidth=5)
            plt.show()

        return MI_surrogates_i

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(data_tmp.shape[0]))

    #### fill results
    MI_surrogates = np.zeros(( data_tmp.shape[0], n_surrogates_cyclefreq ))

    for n_chan in range(data_tmp.shape[0]):

        MI_surrogates[n_chan,:] = compute_surrogates_cyclefreq_results[n_chan]

    #### verif
    if debug:
        count, values, fig = plt.hist(MI_surrogates[95,:])
        plt.vlines(np.percentile(MI_surrogates[0,:], 99), ymin=0, ymax=count.max())
        plt.vlines(np.percentile(MI_surrogates[0,:], 95), ymin=0, ymax=count.max())
        plt.show()
    
    #### save
    np.save(f'{sujet}_{cond}_{str(session_i+1)}_MI_{band_prep}.npy', MI_surrogates)

    print('done')








def precompute_MVL(sujet, band_prep, cond, session_i):

    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)
    respfeatures_i = respfeatures_allcond[cond][session_i]
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}.npy') == True :
        print('ALREADY COMPUTED')
        return

    #### compute surrogates
    #n_chan = 95
    def compute_surrogates_cyclefreq_nchan(n_chan):

        print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

        #### stretch
        x = data_tmp[n_chan,:]
        x_zscore = zscore(x)
        x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_zscore, srate)

        MVL_nchan = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

        x_stretch_linear = x_stretch.reshape(-1) 
        
        #### surrogates
        surrogates_stretch_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

        for surr_i in range(n_surrogates_cyclefreq):

            # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

            surrogates_stretch_tmp[surr_i,:] = shuffle_windows(x_stretch_linear)

        #### compute MVL
        MVL_surrogates_i = np.array([])
        for surr_i in range(n_surrogates_cyclefreq):

            x = surrogates_stretch_tmp[surr_i,:]
            
            MVL_surrogates_i = np.append(MVL_surrogates_i, get_MVL(x-x.min()))

        return MVL_nchan, MVL_surrogates_i

    compute_surrogates_MVL = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(data_tmp.shape[0]))

    #### fill results
    MVL_surrogates = np.zeros(( data_tmp.shape[0], n_surrogates_cyclefreq ))
    MVL_val = np.zeros(( data_tmp.shape[0] ))

    for n_chan in range(data_tmp.shape[0]):

        MVL_surrogates[n_chan,:] = compute_surrogates_MVL[n_chan][1]
        MVL_val[n_chan] = compute_surrogates_MVL[n_chan][0]

    #### verif
    if debug:
        n_chan = 95
        count, values, fig = plt.hist(MVL_surrogates[n_chan,:])
        plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 99), ymin=0, ymax=count.max())
        plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 95), ymin=0, ymax=count.max())
        plt.vlines(MVL_val[n_chan], ymin=0, ymax=count.max(), color='r')
        plt.show()
    
    #### save
    np.save(f'{sujet}_{cond}_{str(session_i+1)}_MVL_{band_prep}.npy', MVL_surrogates)

    print('done')





################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #### load data
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respfeatures_allcond = load_respfeatures(sujet)

    #### compute and save
    print('######## COMPUTE SURROGATES ########')

    #band_prep = band_prep_list[1]
    for band_prep in band_prep_list:

        print(f'COMPUTE FOR {band_prep}')

        #cond = 'FR_CV'
        for cond in ['FR_CV']:

            if len(respfeatures_allcond[cond]) == 1:

                # precompute_surrogates_cyclefreq(sujet, band_prep, cond, 0)
                execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_cyclefreq', [sujet, band_prep, cond, 0])

                # precompute_MI(sujet, band_prep, cond, 0)
                # execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_MI', [sujet, band_prep, cond, 0])
                # precompute_MVL(sujet, band_prep, cond, 0)
                execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_MVL', [sujet, band_prep, cond, 0])               

                if band_prep == 'lf':
                    # precompute_surrogates_coh(sujet, band_prep, cond, 0)
                    execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_coh', [sujet, band_prep, cond, 0])

            elif len(respfeatures_allcond[cond]) > 1:

                for session_i in range(len(respfeatures_allcond[cond])):

                    # precompute_surrogates_cyclefreq(sujet, band_prep, cond, session_i)
                    execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_cyclefreq', [sujet, band_prep, cond, session_i])

                    # precompute_MI(sujet, band_prep, cond, session_i)
                    # execute_function_in_slurm_bash('n6_precompute_surrogates', 'precompute_MI', [sujet, band_prep, cond, session_i])
                    # precompute_MVL(sujet, band_prep, cond, 0)
                    execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_MVL', [sujet, band_prep, cond, session_i])

                    if band_prep == 'lf':
                        # precompute_surrogates_coh(sujet, band_prep, cond, session_i)
                        execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_coh', [sujet, band_prep, cond, session_i])







