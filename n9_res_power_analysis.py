
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







########################################
######## PSD & COH PRECOMPUTE ########
########################################



#dict2reduce = cyclefreq_binned_allcond
def reduce_data(dict2reduce, prms):

    #### adjust for on FR_CV
    prms['conditions'] = ['FR_CV']

    #### identify count
    dict_count = {}
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[band_prep_list[0]][cond])
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[list(dict2reduce.keys())[0]][cond])
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[cond])    

    #### for Pxx & Cyclefreq reduce
    if np.sum([True for i in list(dict2reduce.keys()) if i in band_prep_list]) > 0:
    
        #### generate dict
        dict_reduced = {}

        for band_prep in band_prep_list:
            dict_reduced[band_prep] = {}

            for cond in prms['conditions']:
                dict_reduced[band_prep][cond] = np.zeros(( dict2reduce[band_prep][cond][0].shape ))

        #### fill
        for band_prep in band_prep_list:

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[band_prep][cond] += dict2reduce[band_prep][cond][session_i]

                dict_reduced[band_prep][cond] /= dict_count[cond]

    #### for Cxy & MVL reduce
    elif np.sum([True for i in list(dict2reduce.keys()) if i in prms['conditions']]) > 0:

        #### generate dict
        dict_reduced = {}

        for cond in prms['conditions']:

            dict_reduced[cond] = np.zeros(( dict2reduce[cond][0].shape ))

        #### fill
        for cond in prms['conditions']:

            for session_i in range(dict_count[cond]):

                dict_reduced[cond] += dict2reduce[cond][session_i]

            dict_reduced[cond] /= dict_count[cond]

    #### for surrogates
    else:
        
        #### generate dict
        dict_reduced = {}
        for key in list(dict2reduce.keys()):
            dict_reduced[key] = {}
            for cond in prms['conditions']:
                dict_reduced[key][cond] = np.zeros(( dict2reduce[key][cond][0].shape ))

        #### fill
        #key = 'Cxy'
        for key in list(dict2reduce.keys()):

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[key][cond] += dict2reduce[key][cond][session_i]

                dict_reduced[key][cond] /= dict_count[cond]

    #### verify
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[band_prep][cond].shape
                except:
                    raise ValueError('reducing wrong')
        
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        list_surr = list(dict2reduce.keys())

        for surr_i in list_surr:
        
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[surr_i][cond].shape
                except:
                    raise ValueError('reducing wrong')
    
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            try: 
                _ = dict_reduced[cond].shape
            except:
                raise ValueError('reducing wrong')

    return dict_reduced






def load_surrogates(sujet, respfeatures_allcond, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}, 'cyclefreq_lf' : {}, 'cyclefreq_hf' : {}, 'MVL' : {}}

    for cond in ['FR_CV']:

        if len(respfeatures_allcond[cond]) == 1:

            surrogates_allcond['Cxy'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy')]
            surrogates_allcond['cyclefreq_lf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy')]
            surrogates_allcond['cyclefreq_hf'][cond] = [np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy')]
            surrogates_allcond['MVL'][cond] = [np.load(f'{sujet}_{cond}_{str(1)}_MVL_lf.npy')]

        elif len(respfeatures_allcond[cond]) > 1:

            data_load = {'Cxy' : [], 'cyclefreq_lf' : [], 'cyclefreq_hf' : [], 'MVL' : []}

            for session_i in range(len(respfeatures_allcond[cond])):

                data_load['Cxy'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
                data_load['cyclefreq_lf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
                data_load['cyclefreq_hf'].append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
                data_load['MVL'].append(np.load(f'{sujet}_{cond}_{str(session_i+1)}_MVL_lf.npy'))
            
            surrogates_allcond['Cxy'][cond] = data_load['Cxy']
            surrogates_allcond['cyclefreq_lf'][cond] = data_load['cyclefreq_lf']
            surrogates_allcond['cyclefreq_hf'][cond] = data_load['cyclefreq_hf']
            surrogates_allcond['MVL'][cond] = data_load['MVL']


    return surrogates_allcond







#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond_session(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms):
    
    print(cond)

    #### extract data
    chan_i = prms['chan_list'].index('nasal')
    respi = load_data_sujet(sujet, band_prep, cond, session_i)[chan_i,:]
    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    #### prepare analysis
    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### compute
    Cxy_for_cond = np.zeros(( data_tmp.shape[0], len(hzCxy)))
    Pxx_for_cond = np.zeros(( data_tmp.shape[0], len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( data_tmp.shape[0], stretch_point_surrogates))
    # MI_for_cond = np.zeros(( data_tmp.shape[0] ))
    MVL_for_cond = np.zeros(( data_tmp.shape[0] ))
    # cyclefreq_binned_for_cond = np.zeros(( data_tmp.shape[0], MI_n_bin))

    # MI_bin_i = int(stretch_point_surrogates / MI_n_bin)

    for n_chan in range(data_tmp.shape[0]):

        #### Pxx, Cxy, CycleFreq
        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        x_stretch, trash = stretch_data(respfeatures_allcond[cond][session_i], stretch_point_surrogates, x, prms['srate'])
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

        #### MVL
        x_zscore = zscore(x)
        x_stretch, trash = stretch_data(respfeatures_allcond[cond][session_i], stretch_point_surrogates, x_zscore, prms['srate'])

        MVL_for_cond[n_chan] = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

        # #### MI
        # x = x_stretch_mean

        # x_bin = np.zeros(( MI_n_bin ))

        # for bin_i in range(MI_n_bin):
        #     x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

        # cyclefreq_binned_for_cond[n_chan,:] = x_bin

        # x_bin += np.abs(x_bin.min())*2 #supress zero values
        # x_bin = x_bin/np.sum(x_bin) #transform into probabilities
            
        # MI_for_cond[n_chan] = Shannon_MI(x_bin)

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond

        





def compute_all_PxxCxyCyclefreq(sujet, respfeatures_allcond, prms):

    Pxx_allcond = {'lf' : {}, 'hf' : {}}
    Cxy_allcond = {}
    cyclefreq_allcond = {'lf' : {}, 'hf' : {}}
    MVL_allcond = {}

    #band_prep = band_prep_list[1]
    for band_prep in band_prep_list:

        print(band_prep)

        for cond in ['FR_CV']:

            if ( len(respfeatures_allcond[cond]) == 1 ) & (band_prep == 'lf'):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, band_prep, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond['lf'][cond] = [Pxx_for_cond]
                Cxy_allcond[cond] = [Cxy_for_cond]
                MVL_allcond[cond] = [MVL_for_cond]
                cyclefreq_allcond['lf'][cond] = [cyclefreq_for_cond]

            elif ( len(respfeatures_allcond[cond]) == 1 ) & (band_prep == 'hf') :

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, band_prep, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond['hf'][cond] = [Pxx_for_cond]
                cyclefreq_allcond['hf'][cond] = [cyclefreq_for_cond]

            elif (len(respfeatures_allcond[cond]) > 1) & (band_prep == 'lf'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []
                MVL_load = []

                for session_i, _ in enumerate(respfeatures_allcond[cond]):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    Cxy_load.append(Cxy_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)
                    MVL_load.append(MVL_for_cond)

                Pxx_allcond['lf'][cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                MVL_allcond[cond] = MVL_load
                cyclefreq_allcond['lf'][cond] = cyclefreq_load

            elif (len(respfeatures_allcond[cond]) > 1) & (band_prep == 'hf'):

                Pxx_load = []
                cyclefreq_load = []

                for session_i, _ in enumerate(respfeatures_allcond[cond]):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, band_prep, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond['hf'][cond] = Pxx_load
                cyclefreq_allcond['hf'][cond] = cyclefreq_load

    return Pxx_allcond, Cxy_allcond, cyclefreq_allcond, MVL_allcond




def compute_reduced_PxxCxyCyclefreqSurrogates(sujet, respfeatures_allcond, surrogates_allcond, prms):


    if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'allcond_{sujet}_Pxx.pkl')) == False:
    
        #### compute metrics
        Pxx_allcond, Cxy_allcond, cyclefreq_allcond, MVL_allcond = compute_all_PxxCxyCyclefreq(sujet, respfeatures_allcond, prms)

        #### reduce
        Pxx_allcond_red = reduce_data(Pxx_allcond, prms)
        Cxy_allcond_red = reduce_data(Cxy_allcond, prms)
        cyclefreq_allcond_red = reduce_data(cyclefreq_allcond, prms)
        MVL_allcond_red = reduce_data(MVL_allcond, prms)
        surrogates_allcond_red = reduce_data(surrogates_allcond, prms)

        #### save 
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        with open(f'allcond_{sujet}_Pxx.pkl', 'wb') as f:
            pickle.dump(Pxx_allcond_red, f)

        with open(f'allcond_{sujet}_Cxy.pkl', 'wb') as f:
            pickle.dump(Cxy_allcond_red, f)

        with open(f'allcond_{sujet}_surrogates.pkl', 'wb') as f:
            pickle.dump(surrogates_allcond_red, f)

        with open(f'allcond_{sujet}_cyclefreq.pkl', 'wb') as f:
            pickle.dump(cyclefreq_allcond_red, f)

        with open(f'allcond_{sujet}_MVL.pkl', 'wb') as f:
            pickle.dump(MVL_allcond_red, f)

    else:

        print('ALREADY COMPUTED')

    print('done') 




################################################
######## PLOT & SAVE PSD AND COH ########
################################################




def get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond



#n_chan = 0
def plot_save_PSD_Cxy_CF_MVL(sujet, n_chan):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
    df_loca = get_loca_df(sujet)

    #### adjust params just for FR_CV
    prms['conditions'] = ['FR_CV']

    #### identify chan params
    chan_name = prms['chan_list_ieeg'][n_chan]
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]
    
    #### plot
    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    band_prep = 'lf'

    fig, axs = plt.subplots(nrows=4, ncols=len(prms['conditions']))
    plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV'   
    for c, cond in enumerate(prms['conditions']):

        #### identify respi mean
        respi_mean = []
        for trial_i, _ in enumerate(respfeatures_allcond[cond]):
            respi_mean.append(np.round(respfeatures_allcond[cond][trial_i]['cycle_freq'].median(), 3))
        respi_mean = np.round(np.mean(respi_mean),3)
                
        #### plot
        if len(prms['conditions']) == 1:
            ax = axs[0]
        else:      
            ax = axs[0, c]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx, Pxx_allcond['lf'][cond][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][n_chan,:].max(), color='r')
        ax.set_xlim(0,60)

        if len(prms['conditions']) == 1:
            ax = axs[1]
        else:      
            ax = axs[1, c]
        Pxx_sel_min = Pxx_allcond['lf'][cond][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
        Pxx_sel_max = Pxx_allcond['lf'][cond][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
        ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond['lf'][cond][n_chan,remove_zero_pad:], color='k')
        ax.set_xlim(0, 2)
        ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
        ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond['lf'][cond][n_chan,remove_zero_pad:].max(), color='r')

        if len(prms['conditions']) == 1:
            ax = axs[2]
        else:      
            ax = axs[2, c]
        ax.plot(hzCxy,Cxy_allcond[cond][n_chan,:], color='k')
        ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][n_chan,:], color='c')
        ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

        if len(prms['conditions']) == 1:
            ax = axs[3]
        else:      
            ax = axs[3, c]
        MVL_i = np.round(MVL_allcond[cond][n_chan], 5)
        MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][n_chan,:], 99)
        if MVL_i > MVL_surr:
            MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
        else:
            MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
        # ax.set_title(MVL_p, rotation=0)
        ax.set_xlabel(MVL_p)

        ax.plot(cyclefreq_allcond['lf'][cond][n_chan,:], color='k')
        ax.plot(surrogates_allcond['cyclefreq_lf'][cond][0, n_chan,:], color='b')
        ax.plot(surrogates_allcond['cyclefreq_lf'][cond][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(surrogates_allcond['cyclefreq_lf'][cond][2, n_chan,:], color='c', linestyle='dotted')
        if stretch_TF_auto:
            ax.vlines(prms['respi_ratio_allcond'][cond]*stretch_point_surrogates, ymin=surrogates_allcond['cyclefreq_lf'][cond][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][1, n_chan,:].max(), colors='r')
        else:
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_lf'][cond][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_lf'][cond][1, n_chan,:].max(), colors='r')
        #plt.show() 

    #### save
    os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))
    fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()

    return


    







################################
######## LOAD TF & ITPC ########
################################


def compute_TF_ITPC(sujet, prms):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_stretch.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'allcond_{sujet}_itpc_stretch.pkl')):
                print('ALREADY COMPUTED')
                continue

        #### load file with reducing to one TF
        tf_stretch_allcond = {}

        #band_prep = 'lf'
        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            #### chose nfrex
            if band_prep == 'lf':
                nfrex = nfrex_lf
            elif band_prep == 'hf':
                nfrex = nfrex_hf

            #cond = 'FR_CV'
            for cond in conditions_compute_TF:

                tf_stretch_allcond[band_prep][cond] = {}

                #### impose good order in dict
                for band, freq in freq_band_dict[band_prep].items():

                    if cond == 'FR_CV':
                        _stretch_point_cond = stretch_point_TF

                    elif cond == 'AC':
                        _stretch_point_cond = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

                    elif cond == 'SNIFF':
                        _stretch_point_cond = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                    
                    tf_stretch_allcond[band_prep][cond][band] = np.zeros(( len(prms['chan_list_ieeg']), nfrex, _stretch_point_cond ))

                #### load file
                for band, freq in freq_band_dict[band_prep].items():
                    
                    for file_i in os.listdir(): 
                        if file_i.find(f'{freq[0]}_{freq[1]}_{cond}') != -1:
                            file_to_load = file_i
                        else:
                            continue

                    tf_stretch_allcond[band_prep][cond][band] += np.load(file_to_load)

        #### verif
        for band_prep in band_prep_list:
            for cond in conditions_compute_TF:
                for band, freq in freq_band_dict[band_prep].items():
                    if len(tf_stretch_allcond[band_prep][cond][band]) != len(prms['chan_list_ieeg']):
                        raise ValueError(f'reducing incorrect : {band_prep} {cond} {band}')
               
        #### save
        if tf_mode == 'TF':
            with open(f'allcond_{sujet}_tf_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'allcond_{sujet}_itpc_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done')





def compute_TF_AL(sujet, prms):

    print('######## LOAD TF ########')
    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_AL_stretch.pkl')):
        print('ALREADY COMPUTED')
        return

    #### identify n_session
    cond = 'AL'

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
    n_session = len([file_i for file_i in os.listdir() if file_i.find(cond) != -1])
            
    #### load file with reducing to one TF
    tf_stretch_allcond = {}

    #band_prep = 'lf'
    for band_prep in band_prep_list:

        tf_stretch_allcond[band_prep] = {}

        #### chose nfrex
        if band_prep == 'lf':
            nfrex = nfrex_lf
        elif band_prep == 'hf':
            nfrex = nfrex_hf

        #### impose good order in dict
        for band, freq in freq_band_dict[band_prep].items():
            
            tf_stretch_allcond[band_prep][band] = np.zeros(( len(prms['chan_list_ieeg']), nfrex, resampled_points_AL ))

        #### load file
        for band, freq in freq_band_dict[band_prep].items():

            for session_i in range(n_session):
            
                for file_i in os.listdir(): 
                    if file_i.find(f'{freq[0]}_{freq[1]}_{cond}_{str(session_i+1)}') != -1:
                        file_to_load = file_i
                    else:
                        continue

                tf_stretch_allcond[band_prep][band] += np.load(file_to_load)

        #### mean
        for band, freq in freq_band_dict[band_prep].items():

            tf_stretch_allcond[band_prep][band] /= n_session

    #### verif
    for band_prep in band_prep_list:
        for cond in conditions_compute_TF:
            for band, freq in freq_band_dict[band_prep].items():
                if len(tf_stretch_allcond[band_prep][band]) != len(prms['chan_list_ieeg']):
                    raise ValueError(f'reducing incorrect : {band_prep} {band}')
            
    #### save
    with open(f'allcond_{sujet}_tf_AL_stretch.pkl', 'wb') as f:
        pickle.dump(tf_stretch_allcond, f)

    print('done')




########################################
######## PLOT & SAVE TF & ITPC ########
########################################


def get_tf_itpc_stretch_allcond(sujet, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'allcond_{sujet}_tf_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'allcond_{sujet}_itpc_stretch.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond



def zscore(data):

    data_zscore = (data - data.mean()) / data.std()

    return data_zscore



def robust_zscore(data):
    
    _median = np.median(data) 
    MAD = np.median(np.abs(data-np.median(data)))
    data_zscore = (0.6745*(data-_median))/ MAD
        
    return data_zscore


#n_chan, tf_mode, band_prep = 0, 'TF', 'lf'
def save_TF_ITPC_n_chan(sujet, n_chan, tf_mode, band_prep):

    #### load prms
    prms = get_params(sujet)
    df_loca = get_loca_df(sujet)

    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))
    
    chan_name = prms['chan_list_ieeg'][n_chan]
    chan_loca = df_loca['ROI'][df_loca['name'] == chan_name].values[0]

    print_advancement(n_chan, len(prms['chan_list_ieeg']), steps=[25, 50, 75])

    freq_band = freq_band_dict[band_prep]

    #### scale
    vmaxs = {}
    vmins = {}
    for cond in conditions_compute_TF:

        scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

            del data

        median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

        vmin = np.median(scales['median_val']) - median_diff
        vmax = np.median(scales['median_val']) + median_diff

        vmaxs[cond] = vmax
        vmins[cond] = vmin

    #### plot
    fig, axs = plt.subplots(nrows=len(freq_band), ncols=len(conditions_compute_TF))
    plt.suptitle(f'{sujet}_{chan_name}_{chan_loca}')

    fig.set_figheight(10)
    fig.set_figwidth(10)

    #### for plotting l_gamma down
    if band_prep == 'hf':
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(conditions_compute_TF):

        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = get_tf_itpc_stretch_allcond(sujet, tf_mode)[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions_compute_TF) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            #### generate time vec
            if cond == 'FR_CV':
                time_vec = np.arange(stretch_point_TF)

            if cond == 'AC':
                stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

            if cond == 'SNIFF':
                stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

            # ax.pcolormesh(time_vec, frex, data, vmin=vmins[cond], vmax=vmaxs[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
            # ax.pcolormesh(time_vec, frex, data, shading='gouraud', cmap=plt.get_cmap('seismic'))
            # ax.pcolormesh(time_vec, frex, zscore(data), shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.pcolormesh(time_vec, frex, robust_zscore(data), vmin=-robust_zscore(data).max(), vmax=robust_zscore(data).max(), shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            if cond == 'FR_CV':
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            if cond == 'AC':
                ax.vlines([0, 10], ymin=freq[0], ymax=freq[1], colors='g')
            if cond == 'SNIFF':
                ax.vlines(0, ymin=freq[0], ymax=freq[1], colors='g')

            del data

    #plt.show()

    #### save
    fig.savefig(f'{sujet}_{chan_name}_{chan_loca}_{band_prep}.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()







########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet):
    
    #### load params
    prms = get_params(sujet)
    respfeatures_allcond = load_respfeatures(sujet)
        
    surrogates_allcond = load_surrogates(sujet, respfeatures_allcond, prms)

    #### compute & reduce surrogates
    print('######## COMPUTE & REDUCE PSD AND COH ########')
    compute_reduced_PxxCxyCyclefreqSurrogates(sujet, respfeatures_allcond, surrogates_allcond, prms)
    
    #### compute joblib
    print('######## PLOT & SAVE PSD AND COH ########')
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Cxy_CF_MVL)(sujet, n_chan) for n_chan in range(len(prms['chan_list_ieeg'])))

    print('done')

    


def compilation_compute_TF_ITPC(sujet):

    prms = get_params(sujet)

    compute_TF_ITPC(sujet, prms)
    compute_TF_AL(sujet, prms)
    
    #tf_mode = 'ITPC'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')
        
        #band_prep = 'lf'
        for band_prep in band_prep_list: 

            print(band_prep)

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(sujet, n_chan, tf_mode, band_prep) for n_chan, tf_mode, band_prep in zip(range(len(prms['chan_list_ieeg'])), [tf_mode]*len(prms['chan_list_ieeg']), [band_prep]*len(prms['chan_list_ieeg'])))

    print('done')






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)

        #### Pxx Cxy CycleFreq
        compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet)
        # execute_function_in_slurm_bash_mem_choice('n9_res_power_analysis', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [sujet], 15)


        #### TF & ITPC
        compilation_compute_TF_ITPC(sujet)
        # execute_function_in_slurm_bash_mem_choice('n9_res_power_analysis', 'compilation_compute_TF_ITPC', [sujet], 15)



        