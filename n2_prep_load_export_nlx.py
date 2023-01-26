

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.signal
import pandas as pd
import neo
import scipy.io
import xarray as xr

from n0_config_params import *
from n2bis_prep_load_export_nlx_trig import *

debug = False









################################
######## EXTRACT NLX ########
################################



def import_raw_data(sujet):

    os.chdir(os.path.join(path_raw, sujet, 'raw_data', 'mat'))

    if sujet == 'pat_03146_1608':

        raw_1 = mne.io.read_raw_eeglab(f'{sujet}_1_allchan.set')
        raw_2 = mne.io.read_raw_eeglab(f'{sujet}_2_allchan.set')

        mne.rename_channels(raw_2.info, {ch_B : ch_A for ch_B, ch_A in zip(raw_2.info['ch_names'], raw_1.info['ch_names'])})

        raw = mne.concatenate_raws([raw_1, raw_2], preload=True)

        del raw_1, raw_2

    else:

        raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)

    return raw






def organize_raw(sujet, raw):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    #### extract data
    data = raw.get_data()

    #### identify aux chan
    nasal_i = chan_list_clean.index(aux_chan[sujet]['nasal'])
    ventral_i = chan_list_clean.index(aux_chan[sujet]['ventral'])
    ecg_i = chan_list_clean.index(aux_chan[sujet]['ECG'])

    data_aux = np.vstack((data[nasal_i,:], data[ventral_i,:], data[ecg_i,:]))

    if debug:
        plt.plot(data_aux[0,:])
        plt.plot(data_aux[1,:])
        plt.plot(data_aux[2,:])
        plt.show()

    #### remove from data
    data_ieeg = data.copy()

    #### remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean

    #### remove chan that are not in parcelisation
    os.chdir(os.path.join(path_anatomy, sujet))
    
    plot_loca_df = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_not_in_parcel = []
    chan_i_rmv = 0
    #chan_i, chan_name = 0, chan_list_ieeg[0]
    for chan_i, chan_name in enumerate(chan_list_ieeg):

        if chan_name not in plot_loca_df['plot'].values:

            chan_not_in_parcel.append(chan_name)
            data_ieeg = np.delete(data_ieeg, chan_i-chan_i_rmv, axis=0)

            chan_i_rmv += 1

    [chan_list_ieeg.remove(chan_name) for chan_name in chan_not_in_parcel]

    #### filter channel from plot_loca
    chan_to_reject = plot_loca_df['plot'][plot_loca_df['select'] == 0].values

    chan_i_to_remove = [chan_list_ieeg.index(nchan) for nchan in chan_list_ieeg if nchan in chan_to_reject]

    data_ieeg_rmv = data_ieeg.copy()
    chan_list_ieeg_rmv = chan_list_ieeg.copy()
    rmv_count = 0
    for remove_i in chan_i_to_remove:
        remove_i_adj = remove_i - rmv_count
        data_ieeg_rmv = np.delete(data_ieeg_rmv, remove_i_adj, 0)
        rmv_count += 1

        chan_list_ieeg_rmv.remove(chan_list_ieeg[remove_i])

    if data_ieeg.shape[0] - len(chan_to_reject) != data_ieeg_rmv.shape[0]:
        raise ValueError('problem on chan selection from plot_loca')

    if data_ieeg.shape[1] != data_ieeg_rmv.shape[1]:
        raise ValueError('problem on chan selection from plot_loca')

    data_ieeg = data_ieeg_rmv.copy()
    chan_list_ieeg = chan_list_ieeg_rmv.copy()

    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate





def organize_raw_bi(sujet, raw):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    #### extract data
    data = raw.get_data()

    #### identify aux chan
    nasal_i = chan_list_clean.index(aux_chan[sujet]['nasal'])
    ventral_i = chan_list_clean.index(aux_chan[sujet]['ventral'])
    ecg_i = chan_list_clean.index(aux_chan[sujet]['ECG'])

    data_aux = np.vstack((data[nasal_i,:], data[ventral_i,:], data[ecg_i,:]))

    if debug:
        plt.plot(data_aux[0,:])
        plt.plot(data_aux[1,:])
        plt.plot(data_aux[2,:])
        plt.show()

    #### remove from data
    data_ieeg = data.copy()

    #### remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean

    #### bipol channels
    os.chdir(os.path.join(path_anatomy, sujet))
    
    plot_loca_df_bi = pd.read_excel(sujet + '_plot_loca_bi.xlsx')
    plot_loca_df_mono = pd.read_excel(sujet + '_plot_loca.xlsx')
    plot_loca_df = plot_loca_df_mono.copy()

    for row_i in plot_loca_df.index:
        plot_name, plot_i = plot_loca_df['plot'][row_i].split('_')[0], plot_loca_df['plot'][row_i].split('_')[-1]
        if len(plot_i) == 1:
            plot_loca_df['plot'][row_i] = f'{plot_name}_0{plot_i}'

    data_ieeg_bi = np.zeros((plot_loca_df_bi.index.shape[0], data_ieeg.shape[-1]))

    for row_i in plot_loca_df_bi.index: 
        plot_A, plot_B = plot_loca_df_bi['plot'][row_i].split('-')[0], plot_loca_df_bi['plot'][row_i].split('-')[-1]
        plot_A_i, plot_B_i = np.where(plot_loca_df['plot'] == plot_A)[0][0], np.where(plot_loca_df['plot'] == plot_B)[0][0]
        data_ieeg_bi[row_i, :] = data_ieeg[plot_A_i, :] - data_ieeg[plot_B_i, :]

        #### verify
        if debug:

            plt.plot(data_ieeg[plot_A_i, :] - data_ieeg[plot_B_i, :])
            plt.show()

    #### filter channel from plot_loca
    chan_list_ieeg = plot_loca_df_bi['plot'].values.tolist()
    chan_to_reject = plot_loca_df_bi['plot'][plot_loca_df_bi['select'] == 0].values

    chan_i_to_remove = [chan_list_ieeg.index(nchan) for nchan in chan_list_ieeg if nchan in chan_to_reject]

    data_ieeg_rmv = data_ieeg_bi.copy()
    chan_list_ieeg_rmv = chan_list_ieeg.copy()
    rmv_count = 0
    for remove_i in chan_i_to_remove:
        remove_i_adj = remove_i - rmv_count
        data_ieeg_rmv = np.delete(data_ieeg_rmv, remove_i_adj, 0)
        rmv_count += 1

        chan_list_ieeg_rmv.remove(chan_list_ieeg[remove_i])

    if data_ieeg_bi.shape[0] - len(chan_to_reject) != data_ieeg_rmv.shape[0]:
        raise ValueError('problem on chan selection from plot_loca')

    if data_ieeg_bi.shape[1] != data_ieeg_rmv.shape[1]:
        raise ValueError('problem on chan selection from plot_loca')

    data_ieeg_bi = data_ieeg_rmv.copy()
    chan_list_ieeg = chan_list_ieeg_rmv.copy()

    return data_ieeg_bi, chan_list_ieeg, data_aux, chan_list_aux, srate








def get_trigger_from_ncs(events_mat):

    #### generate trigger df
    events_mat = events_mat[events_mat[:,-2] >= 0, 1:3]
    trig_name = events_mat[:,0]
    trig_time = events_mat[:,1]

    mask = events_mat[:,0] != 0
    trig_name = trig_name[mask]
    trig_time = trig_time[mask]

    #### clean name
    trig_name_clean = []
    trig_clean_i = []
    for trig_i, trig_name_i in enumerate(trig_name):
        for cond in conditions_trig.keys():
            if str(int(trig_name_i)) in conditions_trig[cond]:
                trig_name_clean.append(str(int(trig_name_i)))
                trig_clean_i.append(trig_i)

    trig_time = trig_time[trig_clean_i]

    #### clean time
    trig_start = [i for i, trig_i in enumerate(trig_name_clean) if trig_i[-1] == '1']
    trig_time_clean_i = []
    for start_i in trig_start:
        diff_i = trig_time[start_i + 1] - trig_time[start_i]
        if diff_i > 10 :
            trig_time_clean_i.append(start_i)
            trig_time_clean_i.append(start_i + 1)
            
    mask = np.array(trig_time_clean_i)
    
    trig_time_clean = trig_time[mask]
    trig_name_clean = np.array(trig_name_clean)[mask]

    #### concert from sec to 

    data_dict = {'name': trig_name_clean, 'time': trig_time_clean}
    trigger = pd.DataFrame(data_dict)

    return trigger
    











################################
######## MONITORING ########
################################



def verif_trig(raw_aux, trigger):

    _chan_list_aux = raw_aux.info['ch_names']
    _data_aux = raw_aux.get_data()
    srate = raw_aux.info['sfreq']
    
    respi_i = _chan_list_aux.index('nasal')
    respi = _data_aux[respi_i,:]
    respi = mne.filter.filter_data(respi, srate, 0.5, 2, verbose='CRITICAL')

    times = np.arange(respi.shape[0])/srate
    plt.plot(times, respi)
    plt.vlines(trigger, ymin=np.min(respi), ymax=np.max(respi), colors='r')
    plt.show()




# to compare during preprocessing
def compare_pre_post(pre, post, nchan):

    # compare before after
    x_pre = pre[nchan,:]
    x_post = post[nchan,:]

    nwind = int(10*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx_pre = scipy.signal.welch(x_pre,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
    hzPxx, Pxx_post = scipy.signal.welch(x_post,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

    plt.plot(x_pre, label='x_pre')
    plt.plot(x_post, label='x_post')
    plt.legend()
    plt.title(chan_list_ieeg[nchan])
    plt.show()

    plt.semilogy(hzPxx, Pxx_pre, label='Pxx_pre')
    plt.semilogy(hzPxx, Pxx_post, label='Pxx_post')
    plt.legend()
    plt.xlim(0,150)
    plt.title(chan_list_ieeg[nchan])
    plt.show()










################################
######## PREPROCESSING ########
################################

#raw, prep_step = raw_ieeg, prep_step_lf
def preprocessing_ieeg(raw, prep_step):


    #### 1. Initiate preprocessing step

    def mean_centered(raw):
        
        data = raw.get_data()
        
        # mean centered
        data_mc = np.zeros((np.size(data,0),np.size(data,1)))
        for chan in range(np.size(data,0)):
            data_mc[chan,:] = data[chan,:] - np.mean(data[chan,:])
            #data_mc[chan,:] = scipy.signal.detrend(data_mc[chan,:])

        # fill raw
        for chan in range(np.size(data,0)):
            raw[chan,:] = data_mc[chan,:]    

        # verif
        if debug == True :
            # all data
            duration = .5
            n_chan = 10
            raw.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

            # compare before after
            compare_pre_post(raw.get_data(), raw.get_data(), 4)

        return raw

    def line_noise_removing(raw):

        raw_post = raw.copy()

        raw_post.notch_filter(50)

        if debug == True :

            # compare before after
            compare_pre_post(raw.get_data(), raw_post.get_data(), 4)

        return raw_post

    def high_pass(raw, h_freq, l_freq):

        raw_post = raw.copy()

        #filter_length = int(srate*10) # give sec
        filter_length = 'auto'

        if debug == True :
            h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
            flim = (0.1, srate / 2.)
            mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

        raw_eeg_mc_hp = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')

        if debug == True :
            duration = 60.
            n_chan = 20
            raw_eeg_mc_hp.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        return raw_post

    def low_pass(raw, h_freq, l_freq):

        raw_post = raw.copy()

        filter_length = int(srate*10) # in samples

        if debug == True :
            h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
            flim = (0.1, srate / 2.)
            mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

        raw_post = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hann', fir_design='firwin2')

        if debug == True :
            duration = .5
            n_chan = 10
            raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        return raw_post

    def average_reref(raw):

        raw_post = raw.copy()

        raw_post.set_eeg_reference('average')

        if debug == True :
            duration = .5
            n_chan = 10
            raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        return raw_post

    #### 2. Execute preprocessing 

    if debug:
        raw_init = raw.copy() # first data

    if prep_step['mean_centered']['execute']:
        print('mean_centered_detrend')
        raw_post = mean_centered(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    if prep_step['line_noise_removing']['execute']:
        print('line_noise_removing')
        raw_post = line_noise_removing(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    if prep_step['high_pass']['execute']:
        print('high_pass')
        h_freq = prep_step.get('high_pass').get('params').get('h_freq')
        l_freq = prep_step.get('high_pass').get('params').get('l_freq')
        raw_post = high_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    if prep_step['low_pass']['execute']:
        print('low_pass')
        h_freq = prep_step.get('low_pass').get('params').get('h_freq')
        l_freq = prep_step.get('low_pass').get('params').get('l_freq')
        raw_post = low_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    if prep_step['average_reref']['execute']:
        print('average_reref')
        raw_post = average_reref(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post.copy()

    return raw












################################
######## ECG DETECTION ########
################################

def ecg_detection(data_aux, chan_list_aux, srate):

    ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    #ecg = mne.filter.filter_data(ecg, srate, 8, 15, verbose='CRITICAL')

    info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
    raw_aux = mne.io.RawArray(data_aux, info_aux)

    raw_aux.notch_filter(50, picks='misc')


    if debug:
        ecg = raw_aux.get_data()[-1,:]
        plt.plot(ecg)
        plt.show()

    # ECG
    event_id = 999
    ch_name = 'ECG'
    qrs_threshold = .5 #between o and 1
    ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold)
    ecg_events_time = list(ecg_events[0][:,0])

    return raw_aux, chan_list_aux, ecg_events_time





################################
######## CHOP & SAVE ########
################################

#raw, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time = raw_preproc_lf, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time
def generate_final_raw(data_preproc, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time):

    #### save alldata + stim chan
    data_ieeg = data_preproc.get_data()
    data_all = np.vstack(( data_ieeg, data_aux, np.zeros(( len(data_ieeg[0,:]) )) ))
    chan_list_all = chan_list_ieeg + chan_list_aux + ['ECG_cR']

    ch_types = ['seeg'] * (len(chan_list_all)-4) + ['misc'] * 4
    info = mne.create_info(chan_list_all, srate, ch_types=ch_types)
    raw_all = mne.io.RawArray(data_all, info)

    #### add cR events
    event_cR = np.zeros((len(ecg_events_time),3))
    for cR, _ in enumerate(ecg_events_time):
        event_cR[cR, 0] = int(ecg_events_time[cR])
        event_cR[cR, 2] = 10

    raw_all.add_events(event_cR, stim_channel='ECG_cR', replace=True)
    raw_all.info['ch_names']

    return raw_all












################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #electrode_recording_type = 'monopolaire'
    for electrode_recording_type in ['monopolaire', 'bipolaire']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            #### check if already computed
            print(f'######## COMPUTE {sujet} : {electrode_recording_type} ########')

            os.chdir(os.path.join(path_prep, sujet, 'sections'))
            if os.path.exists(f'{sujet}_allcond_lf.fif') and electrode_recording_type == 'monopolaire':
                print('ALREADY COMPUTED')
                continue

            if os.path.exists(f'{sujet}_allcond_lf_bi.fif') and electrode_recording_type == 'bipolaire':
                print('ALREADY COMPUTED')
                continue

            ################################
            ######## EXTRACT NLX ########
            ################################

            # electrode_recording_type = 'monopolaire'
            # electrode_recording_type = 'bipolaire'

            #### whole protocole
            # sujet = 'pat_03083_1527'
            # sujet = 'pat_03105_1551'
            # sujet = 'pat_03128_1591'
            # sujet = 'pat_03138_1601'
            # sujet = 'pat_03146_1608'
            # sujet = 'pat_03174_1634'

            #### load data
            raw = import_raw_data(sujet)

            if electrode_recording_type == 'bipolaire':
            
                data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw_bi(sujet, raw)

            if electrode_recording_type == 'monopolaire':
            
                data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(sujet, raw)

            info = mne.create_info(ch_names=chan_list_ieeg, ch_types=['eeg']*data_ieeg.shape[0], sfreq=srate)
            raw_ieeg = mne.io.RawArray(data_ieeg, info)

            del raw



            ################################
            ######## AUX PROCESSING ########
            ################################

            raw_aux, chan_list_aux, ecg_events_time = ecg_detection(data_aux, chan_list_aux, srate)



            ################################
            ######## TIRGGER EXPORT ########
            ################################


            #### load trig
            os.chdir(os.path.join(path_raw, sujet, 'events', 'mat'))

            if sujet == 'pat_03146_1608':

                trig = get_trigger_from_ncs(scipy.io.loadmat(f'{sujet}_1_events.mat')['ts'])

            else:

                trig = get_trigger_from_ncs(scipy.io.loadmat(f'{sujet}_events.mat')['ts'])

            #### adjust trig
            if debug:
                
                #### all trig
                verif_trig(raw_aux, trig['time'].values)

                #### VS
                vs_starts = [1062, 1362]
                verif_trig(raw_aux, vs_starts)
                
                #### sniff trig
                sniff_allsession = [1880, 3150]
                verif_trig(raw_aux, sniff_allsession)

                raw_sniff = raw_aux.copy()
                raw_sniff.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )

                respi_i = chan_list_aux.index('nasal')
                respi = raw_sniff.get_data()[respi_i,:]
                height = np.max(respi)*0.6

                plt.plot(respi)
                plt.show()

                sniff_peaks = scipy.signal.find_peaks(respi*-1, height=height, threshold=None, distance=srate, rel_height=0.5)[0]

                sniff_peaks = [11690, 16639, 22067, 28218, 36059, 46448, 55871, 66084, 78150, 87843, 99638, 108351, 117956, 128798, 143108, 157681, 169808, 289990, 292836, 301279, 308435, 312378, 319786, 326674, 335955, 344749, 350285, 359802, 371881, 379452, 389489, 394762, 400358, 404401, 408740, 414330, 418217, 423949, 431127, 438434, 446627, 477538, 483479, 489853, 499283, 503825, 510578, 517425, 524462, 538373, 542962, 548505, 553216, 557738, 563105, 571088, 576144, 580856, 586466, 593084, 599938, 607211, 611638, 619891, 626299, 631383]

                verif_trig(raw_sniff, (np.array(sniff_peaks)/srate))

                #### all trig
                verif_trig(raw_aux, trig['time'].values)
                
                #### AC trig
                ac_allsession = [4430, 5600]
                verif_trig(raw_aux, ac_allsession)

                raw_ac = raw_aux.copy()
                raw_ac.crop( tmin = ac_allsession[0] , tmax= ac_allsession[1] )

                respi_i = chan_list_aux.index('nasal')
                respi = raw_ac.get_data()[respi_i,:]

                plt.plot(respi)
                plt.show()

                ac_starts = [22419, 47654, 63601, 75760, 88055, 100699, 114525, 125131, 136555, 148070, 158798, 172214, 213252, 224601, 238320, 249608, 263911, 275453, 287832, 299600, 311698, 325259, 336560, 348505, 360739, 411185, 418278, 429301, 440703, 452353, 463355, 475111, 486938, 499113, 512919, 524008, 537024, 549926, 560310]
                ac_starts = [i + 33*srate for i in ac_starts]
                
                verif_trig(raw_ac, [int(i)/srate for i in ac_starts])

                #### all trig
                verif_trig(raw_aux, trig['time'].values)
                
                #### AL trig
                al_allsession = [3397, 4268]
                verif_trig(raw_aux, al_allsession)

                raw_al = raw_aux.copy()
                raw_al.crop( tmin = al_allsession[0] , tmax= al_allsession[1] )

                respi_i = chan_list_aux.index('nasal')
                respi = raw_al.get_data()[respi_i,:]
                plt.plot(respi)
                plt.show()

                al_starts = [18303, 148594, 330040]
                al_stops = [65939, 225389, 411280]

                verif_trig(raw_al, [int(i)/srate for i in al_starts])
                verif_trig(raw_al, [int(i)/srate for i in al_stops])


            #### correct values
            vs_starts, sniff_allsession, sniff_peaks, ac_allsession, ac_starts, al_allsession, al_starts, al_stops = get_trig_time_for_sujet(sujet)

            #### verif ECG
            if debug:
                ecg_i = chan_list_aux.index('ECG')
                ecg = raw_aux.get_data()[ecg_i,:]
                times = np.arange(ecg.shape[0])/srate
                plt.plot(times, ecg)
                plt.vlines([int(i)/srate for i in ecg_events_time], ymin=min(ecg), ymax=max(ecg), colors='k')
                trig_plot = [vs_starts, sniff_allsession, ac_allsession, al_allsession]
                for cond_i in range(4):
                    plt.vlines(trig_plot[cond_i], ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
                plt.show()

                #### add events if necessary
                corrected = []
                cR_init = trig['time'].values
                ecg_events_corrected = cR_init + corrected

                #### verify add events
                plt.plot(ecg)
                plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
                plt.vlines(ecg_events_corrected, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
                plt.legend()
                plt.show()

            #### adjust trig for some patients
            if debug:
                ecg_events_corrected = [807.477, 1005.575, 1013.808, 1029.236, 1044.353, 1060.154, 1068.274, 1094.088, 1102.949, 1111.021, 1168.914, 1200.529, 1201.315, 1209.257, 1217.05, 1224.98, 1233.673, 1241.987, 1251.478, 1259.322, 1267.724, 1277.123, 1283.784, 2330.470, 2338.375, 2346.012, 2354.411, 2362.824, 2372.09, 2380.129, 2388.978, 2407.518, 2417.976, 2426.805, 2436.726, 2446.832, 2458.523, 2469.778, 2488, 2507.286, 2517.201, 2525.579, 2534.69, 2543.681, 2554.103, 2571.197, 2608.240, 2617.033, 2625.784]
                ecg_events_time += ecg_events_corrected
                ecg_events_time.sort()

            #### adjust
            ecg_events_time += ecg_events_corrected_allsujet[sujet]
            ecg_events_time.sort()




            ################################
            ######## PREPROCESSING ########
            ################################
            
            # band_prep = 'hf'
            for band_prep in band_prep_list:

                #### choose preproc
                if band_prep == 'lf':
                    data_preproc  = preprocessing_ieeg(raw_ieeg, prep_step_lf)

                if band_prep == 'hf':
                    data_preproc  = preprocessing_ieeg(raw_ieeg, prep_step_hf)

                #### verif
                if debug == True:
                    compare_pre_post(data_ieeg, data_preproc.get_data(), 0)


                ################################
                ######## EXPORT DATA ########
                ################################

                #### generate raw_all
                raw_all = generate_final_raw(data_preproc, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time)

                #### make RAM space
                del data_preproc

                #### initiate count session :

                count_session = {
                    'FR_CV' : [],
                    'SNIFF' : [],
                    'AC' : [],
                    }

                for al_i in range(len(al_starts)):
                    count_session[f'AL_{al_i+1}'] = []

                #### save folder
                os.chdir(os.path.join(path_prep, sujet, 'sections'))
                
                #### Export all preproc
                if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_allcond_{band_prep}.fif')) == False or os.path.exists(os.path.join(os.getcwd(), f'{sujet}_allcond_{band_prep}_bi.fif')) == False:
                    raw_vs_ieeg = raw_all.copy()
                    
                    if electrode_recording_type == 'monopolaire':
                        raw_vs_ieeg.save(f'{sujet}_allcond_{band_prep}.fif')
                    if electrode_recording_type == 'bipolaire':
                        raw_vs_ieeg.save(f'{sujet}_allcond_{band_prep}_bi.fif')
                
                    del raw_vs_ieeg
                
                #### Export VS
                if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_FR_CV_{band_prep}.fif')) == False or os.path.exists(os.path.join(os.getcwd(), f'{sujet}_FR_CV_{band_prep}_bi.fif')) == False:
                    raw_vs_ieeg = raw_all.copy()
                    raw_vs_ieeg.crop( tmin = vs_starts[0] , tmax= vs_starts[1] )

                    count_session['FR_CV'].append(raw_vs_ieeg.get_data().shape[1]/srate)
                    
                    if electrode_recording_type == 'monopolaire':
                        raw_vs_ieeg.save(f'{sujet}_FR_CV_{band_prep}.fif')
                    if electrode_recording_type == 'bipolaire':
                        raw_vs_ieeg.save(f'{sujet}_FR_CV_{band_prep}_bi.fif')
                
                    del raw_vs_ieeg


                #### generate AL
                if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_AL_1_{band_prep}.fif')) == False or os.path.exists(os.path.join(os.getcwd(), f'{sujet}_AL_1_{band_prep}_bi.fif')) == False:
                    raw_al_ieeg = raw_all.copy()
                    raw_al_ieeg.crop( tmin = al_allsession[0] , tmax= al_allsession[1] )
                    for trig_i in range(len(al_starts)):
                        raw_al_ieeg_i = raw_al_ieeg.copy()
                        raw_al_ieeg_i.crop( tmin = int(al_starts[trig_i]/srate) , tmax= int(al_stops[trig_i]/srate) )
                        
                        if electrode_recording_type == 'monopolaire':
                            raw_al_ieeg_i.save(f'{sujet}_AL_{trig_i+1}_{band_prep}.fif')
                        if electrode_recording_type == 'bipolaire':
                            raw_al_ieeg_i.save(f'{sujet}_AL_{trig_i+1}_{band_prep}_bi.fif')

                        count_session[f'AL_{trig_i+1}'].append(raw_al_ieeg_i.get_data().shape[1]/srate)
                        del raw_al_ieeg_i


                #### generate xr SNIFF
                if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_{band_prep}.nc")) == False or os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_{band_prep}_bi.nc")) == False:
                    raw_sniff_ieeg = raw_all.copy()
                    raw_sniff_ieeg.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )

                    data = raw_sniff_ieeg.get_data()
                    times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
                    data_epoch = np.zeros((len(chan_list_ieeg), len(sniff_peaks), len(times)))
                    for nchan in range(len(chan_list_ieeg)):
                        for sniff_i, sniff_time in enumerate(sniff_peaks):
                            _t_start = sniff_time + int(t_start_SNIFF*srate) 
                            _t_stop = sniff_time + int(t_stop_SNIFF*srate)

                            data_epoch[nchan, sniff_i, :] = data[nchan, _t_start:_t_stop]

                    count_session['SNIFF'].append(data_epoch.shape[1])

                    dims = ['chan_list', 'sniffs', 'times']
                    coords = [chan_list_ieeg, range(len(sniff_peaks)), times]
                    xr_epoch_SNIFF = xr.DataArray(data_epoch, coords=coords, dims=dims)

                    if electrode_recording_type == 'monopolaire':
                        xr_epoch_SNIFF.to_netcdf(f"{sujet}_SNIFF_{band_prep}.nc")
                    if electrode_recording_type == 'bipolaire':
                        xr_epoch_SNIFF.to_netcdf(f"{sujet}_SNIFF_{band_prep}_bi.nc")

                    #### make space
                    del xr_epoch_SNIFF
                    del raw_sniff_ieeg


                if debug:
                    maxs = []
                    mins = []
                    for nchan in chan_list_ieeg:
                        plt.title(nchan)
                        plt.plot(xr_epoch_SNIFF['times'], xr_epoch_SNIFF.mean('sniffs').loc[nchan, :])
                        mins.append(np.min(xr_epoch_SNIFF.mean('sniffs').loc[nchan, :]))
                        maxs.append(np.max(xr_epoch_SNIFF.mean('sniffs').loc[nchan, :]))
                        plt.vlines(0, ymax=np.max(maxs), ymin=np.min(mins), colors='r')
                        plt.show()

                #### generate SNIFF
                if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_session_{band_prep}.fif")) == False or os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_session_{band_prep}_bi.fif")) == False:
                    raw_sniff_ieeg = raw_all.copy()
                    raw_sniff_ieeg.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )

                    if electrode_recording_type == 'monopolaire':
                        raw_sniff_ieeg.save(f'{sujet}_SNIFF_session_{band_prep}.fif')
                    if electrode_recording_type == 'bipolaire':
                        raw_sniff_ieeg.save(f'{sujet}_SNIFF_session_{band_prep}_bi.fif')
                    
                    del raw_sniff_ieeg


                #### generate AC
                if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_AC_session_{band_prep}.fif")) == False or os.path.exists(os.path.join(os.getcwd(), f"{sujet}_AC_session_{band_prep}_bi.fif")) == False:
                    raw_ac_ieeg = raw_all.copy()
                    raw_ac_ieeg.crop( tmin = ac_allsession[0] , tmax= ac_allsession[1] )

                    if electrode_recording_type == 'monopolaire':
                        raw_ac_ieeg.save(f'{sujet}_AC_session_{band_prep}.fif')
                    if electrode_recording_type == 'bipolaire':
                        raw_ac_ieeg.save(f'{sujet}_AC_session_{band_prep}_bi.fif')

                    count_session['AC'].append(len(ac_starts))

                    del raw_ac_ieeg

                if electrode_recording_type == 'monopolaire':
                        
                    #### export count session
                    os.chdir(os.path.join(path_prep, sujet, 'info'))
                    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_count_session.xlsx")) == False:

                        df = pd.DataFrame(count_session)
                        df.to_excel(f'{sujet}_count_session.xlsx')

                    #### export AC starts
                    len_ac_session = (ac_allsession[1] - ac_allsession[0])*srate

                    if ac_starts[-1]+t_stop_AC*srate >= len_ac_session:
                        ac_starts = ac_starts[:-1]

                    if ac_starts[0]+t_start_AC*srate <= 0:
                        ac_starts = ac_starts[1:]

                    ac_starts = [str(i) for i in ac_starts]
                    with open(f'{sujet}_AC_starts.txt', 'w') as f:
                        f.write('\n'.join(ac_starts))
                        f.close()

                    #### export SNIFF starts
                    len_sniff_session = (sniff_allsession[1] - sniff_allsession[0])*srate

                    if int(sniff_peaks[-1])+int(t_stop_SNIFF)*srate >= len_sniff_session:
                        sniff_peaks = sniff_peaks[:-1]

                    if int(sniff_peaks[0])+int(t_start_SNIFF)*srate <= 0:
                        sniff_peaks = sniff_peaks[1:]

                    sniff_peaks = [str(i) for i in sniff_peaks]
                    with open(f'{sujet}_SNIFF_starts.txt', 'w') as f:
                        f.write('\n'.join(sniff_peaks))
                        f.close()

                    #### for next iteration change to int
                    ac_starts = [int(i) for i in ac_starts]
                    sniff_peaks = [int(i) for i in sniff_peaks]


                
