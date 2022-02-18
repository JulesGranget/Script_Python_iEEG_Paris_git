

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.signal
import pandas as pd
import neo
import scipy.io
import xarray as xr

from n0_config import *

debug = False


################################
######## EXTRACT NLX ########
################################


def organize_raw(raw):

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

    # remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean


    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate



def get_events(sujet):

    #### get to events
    os.chdir(os.path.join(path_raw, sujet, 'events', 'mat'))
    
    #### load from matlab
    events_mat = scipy.io.loadmat(f'{sujet}_events.mat')['ts']

    #### generate trigger df
    mask = events_mat[:,-2] >= 0
    mask_i = np.where(mask == True)[0]
    events_mat = events_mat[mask_i, 1:3]
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
def generate_final_raw(raw, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time):

    #### save alldata + stim chan
    data_ieeg = raw.get_data()
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

    return raw_all


















if __name__ == '__main__':

    ################################
    ######## EXTRACT NLX ########
    ################################

    #### load data
    os.chdir(os.path.join(path_raw, sujet, 'raw_data', 'mat'))
    raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)
    
    data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(raw)

    info = mne.create_info(ch_names=chan_list_ieeg, ch_types=['eeg']*data_ieeg.shape[0], sfreq=srate)
    raw_ieeg = mne.io.RawArray(data_ieeg, info)

    #### load trig
    trig = get_events(sujet)

    #### verify trig
    if debug:
        respi_i = chan_list_aux.index('nasal')
        respi = data_aux[respi_i,:]
        times = np.arange(respi.shape[0])/srate
        plt.plot(times, respi)
        plt.vlines(trig['time'].values, ymin=np.min(respi), ymax=np.max(respi), colors='r')
        plt.show()

    ################################
    ######## AUX PROCESSING ########
    ################################

    raw_aux, chan_list_aux, ecg_events_time = ecg_detection(data_aux, chan_list_aux, srate)

    if debug:
        #### verif ECG
        ecg_i = chan_list_aux.index('ECG')
        ecg = raw_aux.get_data()[ecg_i,:]
        times = np.arange(ecg.shape[0])/srate
        plt.plot(times, ecg)
        plt.vlines([int(i)/srate for i in ecg_events_time], ymin=min(ecg), ymax=max(ecg), colors='k')
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

    if sujet == 'pat_03083_1527':
        ecg_events_corrected = [] # OK
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()






    ################################
    ######## PREPROCESSING ########
    ################################

    #### choose preproc
    #band_prep = 'lf'
    #raw_preproc  = preprocessing_ieeg(raw_ieeg, prep_step_lf)

    band_prep = 'hf'
    data_preproc = preprocessing_ieeg(raw_ieeg, prep_step_hf)

    #### verif
    if debug == True:
        compare_pre_post(data_ieeg, data_preproc.get_data(), 0)








    ################################
    ######## IDENTIFY TRIG ########
    ################################

    #### generate raw_all
    raw_all = generate_final_raw(data_preproc, chan_list_ieeg, data_aux, chan_list_aux, srate, ecg_events_time)

    #### make RAM space
    del raw_ieeg
    del data_preproc

    #### adjust trig
    if debug:
        #### all trig
        verif_trig(raw_aux, trig['time'].values)
        
        #### sniff trig
        sniff_allsession = [1400.029144, 2205.759889]
        verif_trig(raw_aux, sniff_allsession)

        raw_sniff = raw_aux.copy()
        raw_sniff.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )

        respi_i = chan_list_aux.index('nasal')
        respi = raw_sniff.get_data()[respi_i,:]
        height = np.max(respi)*0.6

        sniff_peaks = scipy.signal.find_peaks(respi, height=height, threshold=None, distance=srate, rel_height=0.5)[0]

        verif_trig(raw_sniff, sniff_peaks/srate)

        #### all trig
        verif_trig(raw_aux, trig['time'].values)
        
        #### AC trig
        ac_allsession = [3232, 4548]
        ac_starts = [4.07e3, 3.142e4, 4.434e4, 5.679e4, 6.97e4, 8.361e4, 9.837e4, 1.1359e5, 1.2613e5, 1.3871e5, 1.5168e5, 1.6726e5, 2.5769e5, 2.7151e5, 2.8864e5, 3.0265e5, 3.1646e5, 3.2968e5, 3.4421e5, 3.5808e5, 3.7203e5, 3.8770e5, 4.0169e5, 4.9019e5, 5.0365e5, 5.1919e5, 5.3497e5, 5.5291e5, 5.6671e5, 5.8028e5, 5.9466e5, 6.1395e5, 6.2994e5]

        verif_trig(raw_aux, ac_allsession)

        raw_ac = raw_aux.copy()
        raw_ac.crop( tmin = ac_allsession[0] , tmax= ac_allsession[1] )

        respi_i = chan_list_aux.index('nasal')
        respi = raw_ac.get_data()[respi_i,:]
        plt.plot(respi)
        plt.show()

        verif_trig(raw_ac, [int(i)/srate for i in ac_starts])

        #### all trig
        verif_trig(raw_aux, trig['time'].values)
        
        #### AL trig
        al_allsession = [2480, 2954]
        al_starts = [1.186e4, 9.708e4, 1.8708e5]
        al_stops = [4.312e4, 1.3547e5, 2.3087e5]

        verif_trig(raw_aux, al_allsession)

        raw_al = raw_aux.copy()
        raw_al.crop( tmin = al_allsession[0] , tmax= al_allsession[1] )

        respi_i = chan_list_aux.index('nasal')
        respi = raw_al.get_data()[respi_i,:]
        plt.plot(respi)
        plt.show()

        verif_trig(raw_al, [int(i)/srate for i in al_starts])
        verif_trig(raw_al, [int(i)/srate for i in al_stops])


    #### correct values
    if sujet == 'pat_03083_1527':
        vs_starts = [768.008963, 1070.280570]
        
        sniff_allsession = [1400.029144, 2205.759889]
        sniff_peaks = [  3559,   9828,  16200,  22645,  29088,  35963,  42842,  49681, 56832,  63841,  69952,  76961,  84099,  91550,  98626, 105185, 111154, 117757, 124988, 132417, 139805, 147169, 154513, 161800, 253317, 259769, 265514, 269723, 275731, 282273, 288594, 295222, 303245, 309441, 316047, 322543, 328905, 334985, 341225, 349783, 356360, 362542, 367246, 374052, 380279, 387166, 393506, 399891]
        
        ac_allsession = [3232, 4548]
        ac_starts = [4.07e3, 3.142e4, 4.434e4, 5.679e4, 6.97e4, 8.361e4, 9.837e4, 1.1359e5, 1.2613e5, 1.3871e5, 1.5168e5, 1.6726e5, 2.5769e5, 2.7151e5, 2.8864e5, 3.0265e5, 3.1646e5, 3.2968e5, 3.4421e5, 3.5808e5, 3.7203e5, 3.8770e5, 4.0169e5, 4.9019e5, 5.0365e5, 5.1919e5, 5.3497e5, 5.5291e5, 5.6671e5, 5.8028e5, 5.9466e5, 6.1395e5, 6.2994e5]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [2480, 2954]
        al_starts = [1.186e4, 9.708e4, 1.8708e5]
        al_stops = [4.312e4, 1.3547e5, 2.3087e5]





    ################################
    ######## EXPORT DATA ########
    ################################

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
    
    
    #### Export VS
    if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_VS_{band_prep}.fif')) != 1:
        raw_vs_ieeg = raw_all.copy()
        raw_vs_ieeg.crop( tmin = vs_starts[0] , tmax= vs_starts[1] )

        count_session['FR_CV'].append(raw_vs_ieeg.get_data().shape[1]/srate)
        
        raw_vs_ieeg.save(f'{sujet}_FR_CV_{band_prep}.fif')
        del raw_vs_ieeg


    #### generate AL
    if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_AL_1_{band_prep}.fif')) != 1:
        raw_al_ieeg = raw_all.copy()
        raw_al_ieeg.crop( tmin = al_allsession[0] , tmax= al_allsession[1] )
        for trig_i in range(len(al_starts)):
            raw_al_ieeg_i = raw_al_ieeg.copy()
            raw_al_ieeg_i.crop( tmin = int(al_starts[trig_i]/srate) , tmax= int(al_stops[trig_i]/srate) )
            raw_al_ieeg_i.save(f'{sujet}_AL_{trig_i+1}_{band_prep}.fif')
            count_session[f'AL_{trig_i+1}'].append(raw_al_ieeg_i.get_data().shape[1]/srate)
            del raw_al_ieeg_i


    #### generate xr SNIFF
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF.nc")) != 1:
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

        xr_epoch_SNIFF.to_netcdf(f"{sujet}_SNIFF.nc")

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



    #### generate xr AC
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_AC_session_{band_prep}.nc")) != 1:
        raw_ac_ieeg = raw_all.copy()
        raw_ac_ieeg.crop( tmin = ac_allsession[0] , tmax= ac_allsession[1] )
        raw_ac_ieeg.save(f'{sujet}_AC_session_{band_prep}.fif')

        count_session['AC'].append(len(ac_starts))

        del raw_ac_ieeg



    #### export count session
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_count_protocol.xlsx")) != 1:

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


 
