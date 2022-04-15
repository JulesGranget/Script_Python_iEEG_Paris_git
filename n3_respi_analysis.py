
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config import *

debug = False

################################
######## LOAD DATA ########
################################

def load_raw_allcond(conditions):
    
    #### adjust conditions
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    dirlist_subject = os.listdir()

    cond_keep = []
    for cond in conditions:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                cond_keep.append(cond)
                break

    conditions = cond_keep

    #### load data lf hf
    raw_allcond = {}

    for band_prep_i in band_prep_list:

        raw_tmp = {}
        for cond in conditions:

            load_i = []
            for session_i, session_name in enumerate(os.listdir()):
                if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep_i) != -1 ):
                    load_i.append(session_i)
                else:
                    continue

            load_list = [os.listdir()[i] for i in load_i]

            data = []
            for load_name in load_list:
                data.append(mne.io.read_raw_fif(load_name, preload=True))

            raw_tmp[cond] = data

        raw_allcond[band_prep_i] = raw_tmp

    srate = int(raw_allcond.get(band_prep_list[0])[cond][0].info['sfreq'])
    chan_list_all = raw_allcond.get(band_prep_list[0])[cond][0].info['ch_names']

    return raw_allcond, conditions, srate, chan_list_all

########################################
######## LOAD RESP FEATURES ########
########################################

def load_resp_features():
    
    os.chdir(os.path.join(path_respfeatures, sujet, 'RESPI'))

    respi_allcond = {}
    for cond in 'FR_CV':

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if (session_name.find(cond) != -1) & (session_name.find('fig') == -1):
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data = []
        for load_name in load_list:
            data.append(pd.read_excel(load_name))

        respi_allcond[cond] = data

    return respi_allcond



########################################
######## COMPUTE RESPI FEATURES ########
########################################

def analyse_resp(resp_sig, sr, t_start, condition):
    
    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt inspi = '-'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '-',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = True,
                                                    manual_baseline = 0.,

                                                    high_pass_filter = True,
                                                    constrain_frequency = None,
                                                    median_windows_filter = True,

                                                    # clean
                                                    eliminate_time_shortest_ratio = 8,
                                                    eliminate_amplitude_shortest_ratio = 10,
                                                    eliminate_mode = 'OR', ) # 'AND')


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r', label='inspi')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g', label='expi')
    ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    ax.legend()
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    







def analyse_resp_debug(resp_sig, sr, t_start, condition, params):

    if params.get('smooth'):
        resp_sig = scipy.signal.savgol_filter(resp_sig, srate*1+1, 3) # window size 51, polynomial order 3

    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt inspi = '-'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '-',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = params.get('baseline_with_average'),
                                                    manual_baseline = params.get('manual_baseline'),

                                                    high_pass_filter = params.get('high_pass_filter'),
                                                    constrain_frequency = params.get('constrain_frequency'),
                                                    median_windows_filter = params.get('median_windows_filter'),

                                                    # clean
                                                    eliminate_time_shortest_ratio = params.get('eliminate_time_shortest_ratio'),
                                                    eliminate_amplitude_shortest_ratio = params.get('eliminate_amplitude_shortest_ratio'),
                                                    eliminate_mode = params.get('eliminate_mode') )
    


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g')
    ax.set_ylabel('resp')
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    


########################################
######## CORRECT RESPFEATURES ########
########################################


def correct_resp_features(respi_allcond, respi_sig, srate):

    #### extract cond and session
    for cond in respi_allcond:

        for session_i in respi_sig[cond]:

            respi_allcond[cond][session_i][0]['cycle_duration'] = np.append(np.diff(respi_allcond[cond][session_i][0]['inspi_time']), np.round( np.mean( np.diff(respi_allcond[cond][session_i][0]['inspi_time'])), 2 ) )
            respi_allcond[cond][session_i][0]['insp_duration'] = respi_allcond[cond][session_i][0]['expi_time'] - respi_allcond[cond][session_i][0]['inspi_time']
            respi_allcond[cond][session_i][0]['exp_duration'] = respi_allcond[cond][session_i][0]['cycle_duration'] - respi_allcond[cond][session_i][0]['insp_duration']
            respi_allcond[cond][session_i][0]['cycle_freq'] = 1/respi_allcond[cond][session_i][0]['cycle_duration']

            cycle_indexes = np.concatenate((respi_allcond[cond][session_i][0]['inspi_index'].values.reshape(-1,1), respi_allcond[cond][session_i][0]['expi_index'].values.reshape(-1,1)), axis=1)
            cycle_freq = respi_allcond[cond][session_i][0]['cycle_freq'].values
            cycle_amplitudes = respi_allcond[cond][session_i][0]['total_amplitude'].values

            fig0, axs = plt.subplots(nrows=3, sharex=True)
            plt.suptitle(cond)
            times = np.arange(respi_sig[cond][session_i].size)/srate
            
                # respi signal with inspi expi markers
            ax = axs[0]
            ax.plot(times, respi_sig[cond][session_i])
            ax.plot(times[cycle_indexes[:, 0]], respi_sig[cond][session_i][cycle_indexes[:, 0]], ls='None', marker='o', color='r', label='inspi')
            ax.plot(times[cycle_indexes[:, 1]], respi_sig[cond][session_i][cycle_indexes[:, 1]], ls='None', marker='o', color='g', label='expi')
            ax.set_xlim(0,120)
            ax.set_ylabel('resp')
            ax.legend()
            

                # instantaneous frequency
            ax = axs[1]
            ax.plot(times[cycle_indexes[:, 0]], cycle_freq)
            ax.set_ylim(0, max(cycle_freq)*1.1)
            ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
            ax.legend()
            ax.set_ylabel('freq')

                # instantaneous amplitude
            ax = axs[2]
            ax.plot(times[cycle_indexes[:, 0]], cycle_amplitudes)
            ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
            ax.set_ylabel('amplitude')
            ax.legend()

            plt.close()
            
            
            # respi cycle features

            fig1, axs = plt.subplots(nrows=2)
            plt.suptitle(cond)

                # histogram cycle freq
            ax = axs[0]
            count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
            ax.plot(bins[:-1], count)
            ax.set_xlim(0,.6)
            ax.set_ylabel('n')
            ax.set_xlabel('freq')
            ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
            W, pval = scipy.stats.shapiro(cycle_freq)
            ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
            ax.legend()
            
                # histogram inspi/expi ratio
            ax = axs[1]
            ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
            count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
            ax.plot(bins[:-1], count)
            ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
            ax.set_ylabel('n')
            ax.set_xlabel('ratio')
            ax.legend()

            plt.close()

            #### replace
            respi_allcond[cond][session_i][1] = fig0
            respi_allcond[cond][session_i][2] = fig1

    return respi_allcond



def zscore(sig):

    sig_clean = ( sig - np.mean(sig) ) / np.std(sig)

    return sig_clean



########################################
######## VERIF RESPIFEATURES ########
########################################



if __name__ == '__main__':

    ############################
    ######## PARAMETERS ########
    ############################

    
    #### whole protocole
    # sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    sujet = 'pat_03138_1601'


    #### FR_CV only
    #sujet = 'pat_02459_0912'
    #sujet = 'pat_02476_0929'
    #sujet = 'pat_02495_0949'

    ############################
    ######## LOAD DATA ########
    ############################

    conditions = ['FR_CV']
    raw_allcond, conditions, srate, chan_list_all = load_raw_allcond(conditions)

    ########################################
    ######## COMPUTE RESPIFEATURES ########
    ########################################

    respi_allcond = {}
    respi_sig = {}
    band_prep = band_prep_list[0]
    cond = 'FR_CV'

    
    for cond in conditions:

        respi_sig[cond] = {}
        data = []

        for session_i in range(len(raw_allcond.get(band_prep)[cond])):

            respi_i = chan_list_all.index('nasal')

            respi = raw_allcond.get(band_prep)[cond][session_i].get_data()[respi_i, :]
            if sujet == 'pat_03105_1551':
                respi *= -1
            respi_clean = scipy.signal.detrend(respi)
            respi_clean = mne.filter.filter_data(respi, srate, 0, 0.5, verbose='CRITICAL')
            respi_clean = zscore(respi_clean)

            if debug:
                plt.plot(respi, label='respi')
                plt.plot(respi_clean, label='respi_clean')
                plt.legend()
                plt.show()

            respi_sig[cond][session_i] = respi_clean
            resp_features, fig0, fig1 = analyse_resp(respi_clean, srate, 0, cond)
            data.append([resp_features, fig0, fig1])
        

    respi_allcond[cond] = data


    ########################################
    ######## VERIF RESPIFEATURES ########
    ########################################
    
    #### info to debug

    cond = 'FR_CV'
    session_i = 0

    respi_allcond[cond][session_i][1].show()
    respi_allcond[cond][session_i][2].show()

    #### recompute
    params = {

    'smooth' : True,

    'baseline_with_average' : False,
    'manual_baseline' : -.1,

    'high_pass_filter' : True,
    'constrain_frequency' : None,
    'median_windows_filter' : True,

    'eliminate_time_shortest_ratio' : 8,
    'eliminate_amplitude_shortest_ratio' : 10,
    'eliminate_mode' : 'OR'

    }

    #respi_i = chan_list.index('ventral')
    respi_i = chan_list_all.index('nasal')

    respi = raw_allcond.get(band_prep)[cond][session_i].get_data()[respi_i, :]
    if sujet == 'pat_03105_1551':
        respi *= -1
    respi_clean = scipy.signal.detrend(respi)
    respi_clean = mne.filter.filter_data(respi, srate, 0, 0.5, verbose='CRITICAL')
    respi_clean = zscore(respi_clean)

    resp_features, fig0, fig1 = analyse_resp_debug(respi_clean, srate, 0, cond, params)
    fig0.show()
    fig1.show()

    #### replace
    respi_allcond[cond][session_i] = [resp_features, fig0, fig1]

        #### check inspi marker
        
    #### inspect pre
    fig, ax = plt.subplots()
    times = np.arange(respi_sig[cond][session_i].size)/srate
    ax.plot(times, respi_sig[cond][session_i])
    ax.plot(respi_allcond[cond][session_i][0]['inspi_time'], respi_sig[cond][session_i][respi_allcond[cond][session_i][0]['inspi_index']], ls='None', marker='o', color='r', label='inspi')
    ax.plot(respi_allcond[cond][session_i][0]['expi_time'], respi_sig[cond][session_i][respi_allcond[cond][session_i][0]['expi_index']], ls='None', marker='o', color='g', label='expi')
    ax.set_ylabel('resp')
    ax.legend()
    plt.show()

    #### modify
    corrected_time_inspi = [3.02, 8.41, 12.38, 16.88, 20.89, 24.72, 28.73, 32.52, 36.03, 40.62, 44.84, 49.96, 54.39, 58.96, 63.36, 67.61, 72.46, 76.99, 81.85, 87.19, 91.68, 96.055, 100.08, 104.15, 108.31, 112.04, 116.43, 121.22, 124.49, 128.35, 131.69, 136.98, 142.06, 146.52, 152.43, 157.28, 161.52, 165.72, 170.50, 174.50, 178.29, 182.71, 187.40, 192.17, 196.91, 200.80, 205.45, 210.35, 215.62, 220.95, 225.58, 229.73, 233.61, 238.20, 243.16, 246.85, 250.14, 254.61, 258.34, 262.70, 267.10, 271.57, 275.80, 279.36, 284.22, 289.14, 294.19, 298.12]
    corrected_index_inspi = [int(i*srate) for i in corrected_time_inspi]
    corrected_time_expi = respi_allcond[cond][session_i][0]['expi_time']
    corrected_index_expi = [int(i*srate) for i in corrected_time_expi]

    #### inspect post
    ms = 10
    fig, ax = plt.subplots()
    times = np.arange(respi_sig[cond][session_i].size)/srate
    ax.plot(times, respi_sig[cond][session_i])
    ax.plot(corrected_time_inspi, respi_sig[cond][session_i][corrected_index_inspi], ls='None', marker='o', color='r', label='inspi')
    ax.plot(respi_allcond[cond][session_i][0]['inspi_time'], respi_sig[cond][session_i][respi_allcond[cond][session_i][0]['inspi_index']], ls='None', marker='x', ms=ms, color='r', label='inspi_pre')
    ax.plot(corrected_time_expi, respi_sig[cond][session_i][respi_allcond[cond][session_i][0]['expi_index']], ls='None', marker='o', color='g', label='expi')
    ax.plot(respi_allcond[cond][session_i][0]['expi_time'], respi_sig[cond][session_i][respi_allcond[cond][session_i][0]['expi_index']], ls='None', marker='x', ms=ms, color='g', label='expi_pre')
    ax.set_ylabel('resp')
    ax.legend()
    plt.show()

    #### when ok switch in df
    if sujet == 'pat_03083_1527':
        cond, session_i = 'FR_CV', 0
        corrected_time_inspi = [0.9, 5.24, 10.01, 14.29, 18.71, 23.36, 28.32, 33.21, 35.29, 38.03, 43.18, 47.75, 50.06, 55.74, 57.17, 61.22, 68.74, 73.88, 80.18, 86.50, 92.60, 98.23, 103.80, 108.82, 113.27, 118.05, 123.50, 128.48, 133.27, 137.76, 142.60, 148.44, 154.26, 159.55, 165.23, 170.04, 175.44, 178.20, 180.79, 190.16, 195.63, 200.92, 202.60, 206.21, 209.48, 213.83, 216.73, 221.92, 226.09, 230.54, 234.68, 239.54, 242.90, 247.51, 253.09, 257.91, 263.16, 266.81, 269.50, 273.88, 277.86, 281.15, 284.66, 288.86, 292.34, 298.65]
        corrected_index_inspi = [int(i*srate) for i in corrected_time_inspi]
        corrected_time_expi = respi_allcond[cond][session_i][0]['expi_time']
        corrected_index_expi = [int(i*srate) for i in corrected_time_expi]
        respi_allcond[cond][session_i][0]['inspi_time'] = corrected_time_inspi
        respi_allcond[cond][session_i][0]['inspi_index'] = corrected_index_inspi
        respi_allcond[cond][session_i][0]['expi_time'] = corrected_time_expi
        respi_allcond[cond][session_i][0]['expi_index'] = corrected_index_expi

        respi_allcond = correct_resp_features(respi_allcond, respi_sig, srate)

    if sujet == 'pat_03105_1551':
        cond, session_i = 'FR_CV', 0
        corrected_time_inspi = [1.91, 6.24, 10.13, 14.46, 17.79, 20.30, 24.24, 26.45, 29.81, 32.97, 36.57, 39.94, 42.87, 46.08, 49.23, 53.61, 56.78, 60.89, 64.38, 66.17, 68.95, 72.76, 76.14, 79.06, 83.09, 86.67, 89.92, 93, 97.12, 99.91, 103.13, 106.19, 109.39, 112.82, 116.18, 119.38, 122.94, 125.71, 129.97, 133.09, 136.65, 140.22, 143.50, 146.85, 149.44, 152.82, 156.04, 159.04, 161.68, 165.54, 168.15, 171.23, 174.31, 177.38, 180.73, 183.14, 186.38, 189.28, 193.11, 197.85, 203.07, 206.42, 209.82, 212.73, 216.36, 219.61, 222.12, 226.30, 231.25, 235.47, 238.47, 241.65, 244.69, 247.81, 251.90, 254.88, 257.82, 260.32, 263.17, 266.07, 269.01, 272.02, 274.83, 277.47, 287.60, 280.81, 283.83, 290.64, 293.60, 297.02]
        corrected_index_inspi = [int(i*srate) for i in corrected_time_inspi]
        corrected_time_expi = respi_allcond[cond][session_i][0]['expi_time']
        corrected_index_expi = [int(i*srate) for i in corrected_time_expi]
        respi_allcond[cond][session_i][0]['inspi_time'] = corrected_time_inspi
        respi_allcond[cond][session_i][0]['inspi_index'] = corrected_index_inspi
        respi_allcond[cond][session_i][0]['expi_time'] = corrected_time_expi
        respi_allcond[cond][session_i][0]['expi_index'] = corrected_index_expi

        respi_allcond = correct_resp_features(respi_allcond, respi_sig, srate)

    if sujet == 'pat_03128_1591':
        cond, session_i = 'FR_CV', 0
        corrected_time_inspi = [5.02, 9.83, 18.17, 24.39, 30.33, 34.95, 40.70, 45.47, 51.72, 57.34, 63.12, 68.92, 73.67, 78.31, 83.38, 88.50, 94.06, 98.44, 103.85, 109.02, 114.40, 120.52, 125.85, 131.57, 137.85, 142.63, 148.17, 153.02, 158.46, 163.31, 168.45, 173.25, 177.87, 183.15, 189.13, 194.44, 199.45, 204.12, 209.08, 214.22, 219.02, 224.11, 228.65, 233.90, 239.36, 244.61, 249.75, 254.34, 258.99, 264.13, 269.04, 274.18, 279.01, 283.92, 289.07, 293.49, 298.35]
        corrected_index_inspi = [int(i*srate) for i in corrected_time_inspi]
        corrected_time_expi = respi_allcond[cond][session_i][0]['expi_time']
        corrected_index_expi = [int(i*srate) for i in corrected_time_expi]
        respi_allcond[cond][session_i][0]['inspi_time'] = corrected_time_inspi
        respi_allcond[cond][session_i][0]['inspi_index'] = corrected_index_inspi
        respi_allcond[cond][session_i][0]['expi_time'] = corrected_time_expi
        respi_allcond[cond][session_i][0]['expi_index'] = corrected_index_expi

        respi_allcond = correct_resp_features(respi_allcond, respi_sig, srate)

    if sujet == 'pat_03138_1601':
        cond, session_i = 'FR_CV', 0
        corrected_time_inspi = [3.02, 8.41, 12.38, 16.88, 20.89, 24.72, 28.73, 32.52, 36.03, 40.62, 44.84, 49.96, 54.39, 58.96, 63.36, 67.61, 72.46, 76.99, 81.85, 87.19, 91.68, 96.055, 100.08, 104.15, 108.31, 112.04, 116.43, 121.22, 124.49, 128.35, 131.69, 136.98, 142.06, 146.52, 152.43, 157.28, 161.52, 165.72, 170.50, 174.50, 178.29, 182.71, 187.40, 192.17, 196.91, 200.80, 205.45, 210.35, 215.62, 220.95, 225.58, 229.73, 233.61, 238.20, 243.16, 246.85, 250.14, 254.61, 258.34, 262.70, 267.10, 271.57, 275.80, 279.36, 284.22, 289.14, 294.19, 298.12]
        corrected_index_inspi = [int(i*srate) for i in corrected_time_inspi]
        corrected_time_expi = respi_allcond[cond][session_i][0]['expi_time']
        corrected_index_expi = [int(i*srate) for i in corrected_time_expi]
        respi_allcond[cond][session_i][0]['inspi_time'] = corrected_time_inspi
        respi_allcond[cond][session_i][0]['inspi_index'] = corrected_index_inspi
        respi_allcond[cond][session_i][0]['expi_time'] = corrected_time_expi
        respi_allcond[cond][session_i][0]['expi_index'] = corrected_index_expi

        respi_allcond = correct_resp_features(respi_allcond, respi_sig, srate)

    

    ################################
    ######## SAVE FIG ########
    ################################


    #### when everything ok
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))

    respi_allcond['FR_CV'][0][0].to_excel(sujet + '_FR_CV_respfeatures.xlsx')
    respi_allcond['FR_CV'][0][1].savefig(sujet + '_FR_CV_fig0.jpeg')
    respi_allcond['FR_CV'][0][2].savefig(sujet + '_FR_CV_fig1.jpeg')

