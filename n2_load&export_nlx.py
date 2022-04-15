

from multiprocessing.sharedctypes import Value
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

    #### remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean

    #### filter channel from plot_loca
    os.chdir(os.path.join(path_anatomy, sujet))
    
    plot_loca_df = pd.read_excel(sujet + '_plot_loca.xlsx')

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

    #### whole protocole
    sujet = 'pat_03083_1527'
    #sujet = 'pat_03105_1551'
    #sujet = 'pat_03128_1591'
    #sujet = 'pat_03138_1601'

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

    if sujet == 'pat_03105_1551':
        ecg_events_corrected = [] # OK
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()

    if sujet == 'pat_03128_1591':
        ecg_events_corrected = [] # OK
        ecg_events_time += ecg_events_corrected
        ecg_events_time.sort()


    ################################
    ######## PREPROCESSING ########
    ################################

    #### choose preproc
    band_prep = 'lf'
    data_preproc  = preprocessing_ieeg(raw_ieeg, prep_step_lf)

    #band_prep = 'hf'
    #data_preproc = preprocessing_ieeg(raw_ieeg, prep_step_hf)

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
        sniff_allsession = [2377, 3441]
        verif_trig(raw_aux, sniff_allsession)

        raw_sniff = raw_aux.copy()
        raw_sniff.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )

        respi_i = chan_list_aux.index('nasal')
        respi = raw_sniff.get_data()[respi_i,:]
        height = np.max(respi)*0.6

        sniff_peaks = scipy.signal.find_peaks(respi*-1, height=height, threshold=None, distance=srate, rel_height=0.5)[0]

        verif_trig(raw_sniff, (sniff_peaks/srate)-0.2)

        #### all trig
        verif_trig(raw_aux, trig['time'].values)
        
        #### AC trig
        ac_allsession = [4340.9, 5085.3]
        ac_starts = [9.11e3, 2.836e4, 4.258e4, 5.673e4, 6.825e4, 8.252e4, 9.971e4, 1.1732e5, 1.3326e5, 1.4959e5, 2.1666e5, 2.2981e5, 2.4114e5, 2.5333e5, 2.6586e5, 2.7816e5, 2.9116e5, 3.0356e5, 3.1714e5, 3.3133e5, 3.4529e5, 3.6197e5]
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
        al_allsession = [3614, 4116]
        al_starts = [2.528e4, 1.0628e5, 2.0153e5]
        al_stops = [6.305e4, 1.5602e5, 2.3869e5]

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
        sniff_peaks = [ 3208,   9454,  15835,  22286,  28739,  35610,  42510, 49339,  56480,  63488,  69607,  76597,  83757,  91220, 98287, 104796, 110805, 117415, 124636, 132065, 139446, 146807, 154158, 161457, 252954, 259483, 265212, 269421, 275419, 281938, 288261, 294891, 302913, 309103, 315711, 322221, 328565, 334644, 340894, 349464, 356035, 362207, 366928, 373727, 379926, 386837, 393157, 399566]
        ac_allsession = [3232, 4548]
        ac_starts = [4.07e3, 3.142e4, 4.434e4, 5.679e4, 6.97e4, 8.361e4, 9.837e4, 1.1359e5, 1.2613e5, 1.3871e5, 1.5168e5, 1.6726e5, 2.5769e5, 2.7151e5, 2.8864e5, 3.0265e5, 3.1646e5, 3.2968e5, 3.4421e5, 3.5808e5, 3.7203e5, 3.8770e5, 4.0169e5, 4.9019e5, 5.0365e5, 5.1919e5, 5.3497e5, 5.5291e5, 5.6671e5, 5.8028e5, 5.9466e5, 6.1395e5, 6.2994e5]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [2480, 2954]
        al_starts = [1.186e4, 9.708e4, 1.8708e5]
        al_stops = [4.312e4, 1.3547e5, 2.3087e5]

    if sujet == 'pat_03105_1551':
        vs_starts = [1624.984596, 1926.160561]
        
        sniff_allsession = [2168.62969, 2931.20041]
        sniff_peaks = [  3478,   7939,  14268,  20847,  27259,  32191,  38562, 43492,  50235,  57004,  63607,  69998,  76069,  82459, 88869,  95168, 101639, 108149, 114398, 120516, 126435, 132675, 139195, 145356, 151487, 157244, 163463, 217688, 223227, 228666, 233966, 239356, 244456, 249666, 254765, 259864, 265163, 270363, 275843, 281012, 286301, 291401, 296432, 301852, 307194, 312365, 317645, 322965, 328455, 333705, 338948, 344077, 349138, 354418, 359430, 364820, 370289, 375877]

        ac_allsession = [3605, 4812]
        ac_starts = [5075.0000000000455, 15549.999999999955, 28000, 38250.0, 50799.999999999956, 65200.000000000044, 79700.00000000004, 94700.00000000004, 109799.99999999996, 124900.00000000009, 139049.99999999994, 152900.0000000001, 167500, 236450.00000000006, 249300.00000000017, 262750.0, 277500, 292000, 306599.9999999999, 322750.0, 337050.0000000002, 351150.0000000001, 367000, 380800.0000000002, 442400.0000000001, 456900.0000000001, 470000, 484300.0000000002, 498750.0, 516000, 529699.9999999998, 545449.9999999998, 560550.0000000002, 574599.9999999999, 588099.9999999999]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [2978, 3346]
        al_starts = [1.120e4, 7.999e4, 1.4986e5]
        al_stops = [3.335e4, 9.963e4, 1.6998e5]

    if sujet == 'pat_03128_1591':
        vs_starts = [241.577467,  543.440413]
        
        sniff_allsession = [892.9, 1662.1]
        sniff_peaks = [  2156,   4806,   7599,  10826,  14243,  17907,  21460, 25111,  29285,  33171,  36525,  39501,  42393,  45286, 48362,  51300,  54659,  57948,  61480,  65052,  68917, 72637,  76220,  79756,  83555,  87537,  91452,  95141, 99330, 103160, 107099, 111328, 115157, 119066, 122755, 126605, 130432, 134542, 138413, 142237, 146048, 149727, 153426, 157182, 160644, 164477, 168323, 195436, 210875, 214123, 217593, 220843, 224072, 227533, 231035, 234489, 237311, 239983, 242902, 245693, 248803, 251262, 253882, 256761, 259691, 262720, 265741, 269247, 273571, 276813, 279913, 283290, 286766, 290882, 295511, 301220, 306671, 312049, 316899, 322517, 327340, 332468, 336960, 340938, 344950, 348662, 352873, 356915, 361403, 365827, 370203, 375094]
        ac_allsession = [2270.7, 2952.0]
        ac_starts = [7940.000000000055, 16769.99999999998, 26730.00000000002, 37130.00000000011, 47190.00000000006, 56650.00000000009, 68180.00000000006, 81945.00000000016, 81945.00000000016, 96275.00000000009, 107280.0000000002, 119305.00000000006, 131670.00000000006, 144970.00000000003, 195640.0000000001, 208260.0, 221210.00000000003, 234019.99999999997, 246930.00000000006, 259965.00000000015, 272765.0000000001, 285205.0000000002, 297715.0000000001, 310695.0000000002, 322355.0, 333940.00000000006]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [1787, 2067]
        al_starts = [8149.999999999977, 44250.0, 104350.00000000003]
        al_stops = [29450.000000000044, 68200.00000000004, 135200.00000000006]

    if sujet == 'pat_03138_1601':
        vs_starts = [1718.971136, 2021.155615]
        
        sniff_allsession = [2377, 3441]
        sniff_peaks = [  8453,  15204,  22744,  28425,  36736,  47992,  63959, 74949,  94392, 105881, 121167, 145939, 164163, 202677, 219542, 234655, 253170, 277135, 300756, 317368, 333043, 379714, 395769, 412043, 428842, 438727, 456949, 477525, 497956, 521150]  
            
        ac_allsession = [4340.9, 5085.3]
        ac_starts = [9.11e3, 2.836e4, 4.258e4, 5.673e4, 6.825e4, 8.252e4, 9.971e4, 1.1732e5, 1.3326e5, 1.4959e5, 2.1666e5, 2.2981e5, 2.4114e5, 2.5333e5, 2.6586e5, 2.7816e5, 2.9116e5, 3.0356e5, 3.1714e5, 3.3133e5, 3.4529e5, 3.6197e5]
        ac_starts = [int(i) for i in ac_starts] 


        al_allsession = [3614, 4116]
        al_starts = [2.528e4, 1.0628e5, 2.0153e5]
        al_stops = [6.305e4, 1.5602e5, 2.3869e5]



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
    if os.path.exists(os.path.join(os.getcwd(), f'{sujet}_FR_CV_{band_prep}.fif')) != 1:
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
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_{band_prep}.nc")) != 1:
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

        xr_epoch_SNIFF.to_netcdf(f"{sujet}_SNIFF_{band_prep}.nc")

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
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_SNIFF_session_{band_prep}.fif")) != 1:
        raw_sniff_ieeg = raw_all.copy()
        raw_sniff_ieeg.crop( tmin = sniff_allsession[0] , tmax= sniff_allsession[1] )
        raw_sniff_ieeg.save(f'{sujet}_SNIFF_session_{band_prep}.fif')


        del raw_sniff_ieeg


    #### generate AC
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_AC_session_{band_prep}.fif")) != 1:
        raw_ac_ieeg = raw_all.copy()
        raw_ac_ieeg.crop( tmin = ac_allsession[0] , tmax= ac_allsession[1] )
        raw_ac_ieeg.save(f'{sujet}_AC_session_{band_prep}.fif')

        count_session['AC'].append(len(ac_starts))

        del raw_ac_ieeg



    #### export count session
    os.chdir(os.path.join(path_prep, sujet, 'info'))
    if os.path.exists(os.path.join(os.getcwd(), f"{sujet}_count_session.xlsx")) != 1:

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


 
