

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.signal
import pandas as pd
import neo

from n0_config import *

debug = False


################################
######## EXTRACT NLX ########
################################

#sujet_i = 'pat_03083_1527'
def open_raw_neuralynx(sujet_i):

    os.chdir(os.path.join(path_data, sujet_i, 'eeg'))
    folder_to_open = [file_i for file_i in os.listdir() if file_i.find('unused') == -1]
    os.chdir(os.path.join(path_data, sujet_i, 'eeg', folder_to_open[0]))

    reader = neo.io.NeuralynxIO(dirname=os.path.join(path_data, sujet_i, 'eeg', folder_to_open[0]))
















def organize_raw(raw):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    #### extract data
    data = raw.get_data()

    #### identify aux chan
    nasal_i = chan_list_clean.index(aux_chan.get('nasal'))
    ventral_i = chan_list_clean.index(aux_chan.get('ventral'))
    ecg_i = chan_list_clean.index(aux_chan.get('ECG'))

    aux_i_list = [nasal_i, ventral_i, ecg_i]

    data_aux = np.vstack((data[nasal_i,:], data[ventral_i,:], data[ecg_i,:]))

    if debug:
        plt.plot(data_aux[0,:])
        plt.plot(data_aux[1,:])
        plt.plot(data_aux[2,:])
        plt.show()

    #### remove from data
    chan_list_ieeg = chan_list_clean.copy()
    data_ieeg = data.copy()

    #remove PRES1
    aux_i = chan_list_ieeg.index('PRES1')
    data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
    chan_list_ieeg.remove('PRES1')    

    # remove other aux
    for aux_name in aux_chan.keys():

        aux_i = chan_list_ieeg.index(aux_chan.get(aux_name))
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_ieeg.remove(aux_chan.get(aux_name))        

    chan_list_aux = list(aux_chan.keys())


    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate




################################
######## COMPARISON ########
################################


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

#raw, srate = raw_egg, srate
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

def chop_save_trc(data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc, export_info):

    #### save alldata + stim chan
    data_all = np.vstack(( data_ieeg, data_aux, np.zeros(( len(data_ieeg[0,:]) )) ))
    chan_list_all = chan_list_ieeg + chan_list_aux + ['ECG_cR']

    ch_types = ['seeg'] * (len(chan_list_all)-4) + ['misc'] * 4
    info = mne.create_info(chan_list_all, srate, ch_types=ch_types)
    raw_all = mne.io.RawArray(data_all, info)

    #### add cR events
    event_cR = np.zeros((len(ecg_events_time),3))
    for cR in range(len(ecg_events_time)):
        event_cR[cR, 0] = ecg_events_time[cR]
        event_cR[cR, 2] = 10

    raw_all.add_events(event_cR, stim_channel='ECG_cR', replace=True)
    raw_all.info['ch_names']


    #### save chunk
    count_session = {
    'FR_CV' : 0,
    'SNIFF' : 0,
    'AL' : 0,
    'AC' : 0,
    }

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    # condition, trig_cond = list(conditions_trig.items())[0]
    for condition, trig_cond in conditions_trig.items():

        cond_i_start = np.where(trig.name.values == trig_cond[0])[0]
        cond_i_stop = np.where(trig.name.values == trig_cond[1])[0]

        for i, trig_start in enumerate(cond_i_start):

            trig_stop = cond_i_stop[i]

            count_session[condition] = count_session[condition] + 1 

            raw_chunk = raw_all.copy()
            raw_chunk.crop( tmin = (trig.iloc[trig_start,:].time)/srate , tmax= (trig.iloc[trig_stop,:].time/srate)-0.2 )
            
            raw_chunk.save(sujet + '_' + condition + '_' + str(i+1) + '_' + band_preproc + '.fif')




    df = {'condition' : list(count_session.keys()), 'count' : list(count_session.values())}
    count_session = pd.DataFrame(df, columns=['condition', 'count'])

    if export_info == True :
    
        #### export trig, count_session, cR
        os.chdir(os.path.join(path_prep, sujet, 'info'))
        
        trig.to_excel(sujet + '_trig.xlsx')

        count_session.to_excel(sujet + '_count_session.xlsx')

        cR = pd.DataFrame(ecg_events_time, columns=['cR_time'])
        cR.to_excel(sujet +'_cR_time.xlsx')


    return 




















if __name__ == '__main__':

    ################################
    ######## EXTRACT NLX ########
    ################################

    #### load data
    os.chdir(os.path.join(path_data, sujet))
    raw = mne.io.read_raw_eeglab(os.path.join(path_data, sujet, (sujet + '_complete_iEEG_clean.set')), preload=True)
    data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(raw)
    
    #### load trig
    os.chdir(os.path.join(path_trig, sujet))

    trig_file = open(sujet + '_trig_delay.txt', "r")
    trig_list = trig_file.read()
    trig_file.close()

    trig_list = trig_list.split("\n")[:-1]
    trig_name = []
    trig_time = []
    for i, el in enumerate(trig_list):
        if i == 0:
            continue
        else:
            split = el.split('\t')
            trig_name.append(split[0])
            trig_time.append(float(split[1]))

    #### verify trig
    if debug:
        respi_i = chan_list_aux.index('nasal')
        respi = data_aux[respi_i,:]
        plt.plot(respi)
        plt.vlines( (np.array(trig_time) / 2 ) - 270780, ymin=min(respi), ymax=max(respi), colors='r')
        plt.show()


    #### adjust
    if sujet == 'pat_02459_0912':
        trig_time = list(((np.array(trig_time) / 2) - 829489 ).astype(int))
    if sujet == 'pat_02476_0929':
        trig_time = list(((np.array(trig_time) / 2) - 1395911 ).astype(int))
    if sujet == 'pat_02495_0949':
        trig_time = list(((np.array(trig_time) / 2) - 971115.875 ).astype(int))
    if sujet == 'pat_02718_1201':
        trig_time = list(((np.array(trig_time) / 2) - 270780 ).astype(int))

    #### generate df
    trig_dict = {'name' : trig_name, 'time' : trig_time}
    trig = pd.DataFrame(trig_dict, columns=['name', 'time'])

    #### verify trig
    if debug:
        respi_i = chan_list_aux.index('nasal')
        respi = data_aux[respi_i,:]
        plt.plot(respi)
        plt.vlines(trig['time'].values, ymin=min(respi), ymax=max(respi), colors='r')
        plt.show()


    ################################
    ######## AUX PROCESSING ########
    ################################

    raw_aux, chan_list_aux, ecg_events_time = ecg_detection(data_aux, chan_list_aux, srate)

    if debug:
        #### verif ECG
        ecg_i = chan_list_aux.index('ECG')
        ecg = raw_aux.get_data()[ecg_i,:]
        plt.plot(ecg)
        plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
        plt.vlines(trig['time'].values, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
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
    if sujet == 'pat_02459_0912':
        ecg_events_corrected = [1007705, 1010522, 1120726, 1121095, 1121458, 1121811, 1122169, 1122516, 1122863, 1123210, 1123563, 1123921, 1584942]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()

    if sujet == 'pat_02476_0929':
        ecg_events_corrected = [2.7039e5, 533976, 534261, 534540, 667162, 670598, 670944, 673689, 674002, 674332, 674661]
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()

    if sujet == 'pat_02495_0949':
        ecg_events_corrected = [] # OK
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()   


    if sujet == 'pat_02718_1201':
        ecg_events_corrected = [] # OK
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()





    ################################
    ######## PREPROCESSING ########
    ################################

    data_preproc_lf  = preprocessing_ieeg(data_ieeg, chan_list_ieeg, srate, prep_step_lf)
    data_preproc_hf = preprocessing_ieeg(data_ieeg, chan_list_ieeg, srate, prep_step_hf)
    
    #### verif
    if debug == True:
        compare_pre_post(data_ieeg, data_preproc_lf, 0)



    ################################
    ######## CHOP AND SAVE ########
    ################################

    chop_save_trc(data_preproc_lf, chan_list_ieeg, raw_aux.get_data(), chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc='lf', export_info=True)
    chop_save_trc(data_preproc_hf, chan_list_ieeg, raw_aux.get_data(), chan_list_aux, conditions_trig, trig, srate, ecg_events_time, band_preproc='hf', export_info=False)


