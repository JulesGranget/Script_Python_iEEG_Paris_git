

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False


########################
######## DRAFT ########
########################

prms = get_params(sujet)
df_loca = get_loca_df(sujet)
sniff = get_sniff_starts(sujet)

data = load_data('SNIFF', band_prep='hf')
x = data[0,:]
y = data[10,:]

plt.plot(data[-3,:])
plt.show()

plt.plot(x)
plt.plot(y)
plt.show()

nperseg = 5 * prms['srate']
noverlap = nperseg * 0.5
nfft = nperseg

hzPxx, Pxx = scipy.signal.welch(x, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
hzPxx, Pyy = scipy.signal.welch(y, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
hzPxy, Pxy = scipy.signal.csd(x, y, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)


plt.plot(np.log( hzPxy ), np.log( Pxx ), label='Pxx')
plt.plot(np.log( hzPxy ), np.log( Pyy ), label='Pyy')
plt.plot(np.log( hzPxy ), np.log( Pxy ), label='Pxy')
plt.legend()
plt.show()


wavelets, nfrex = get_wavelets('hf', [80, 120])

hzPxx, Pxx_wavelet = scipy.signal.welch(wavelets[10,:], fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
plt.plot(hzPxx, np.log( Pxx_wavelet ))
plt.show()


x_conv = scipy.signal.fftconvolve(x, wavelets[10,:], 'same')
y_conv = scipy.signal.fftconvolve(y, wavelets[10,:], 'same')

plt.plot(x_conv)
plt.plot(y_conv)
plt.show()


hzPxx, Pxx_conv = scipy.signal.welch(x_conv, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
hzPxx, Pyy_conv = scipy.signal.welch(y_conv, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
hzPxy, Pxy_conv = scipy.signal.csd(x_conv, y_conv, fs=prms['srate'], window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)

plt.plot(hzPxy, np.log( Pxx_conv ), label='Pxx')
plt.plot(hzPxy, np.log( Pyy_conv ), label='Pyy')
plt.plot(hzPxy, np.log( Pxy_conv ), label='Pxy')
plt.legend()
plt.show()


x = data[2,:]
y = data[30,:]


stretch_point_TF_epoch = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
epochs = np.zeros(( 2, len(sniff), stretch_point_TF_epoch ), dtype='complex')

for epoch_i, epoch_time in enumerate(sniff):

    _t_start = epoch_time + int(t_start_SNIFF*prms['srate']) 
    _t_stop = epoch_time + int(t_stop_SNIFF*prms['srate'])

    epochs[0, epoch_i, :] = x_conv[_t_start:_t_stop]
    epochs[1, epoch_i, :] = y_conv[_t_start:_t_stop]


#### identify slwin
slwin_len = .5    # in sec
slwin_step = slwin_len*slwin_step_coeff  # in sec
times_epoch = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/prms['srate'])
win_sample = frites.conn.define_windows(times_epoch, slwin_len=slwin_len, slwin_step=slwin_step)[0]
times = np.linspace(t_start_SNIFF, t_stop_SNIFF, len(win_sample))

print('COMPUTE')

dfc = np.zeros(( len(sniff), 3, len(win_sample) ))
ispc_dfc_i = np.zeros(( len(win_sample) ))
pli_dfc_i = np.zeros(( len(win_sample) ))
wpli_dfc_i = np.zeros(( len(win_sample) ))

for sniff_i, _ in enumerate(sniff):

    #slwin_values_i, slwin_values = 0, win_sample[0]
    for slwin_values_i, slwin_values in enumerate(win_sample):
            
        as1 = epochs[0, sniff_i, slwin_values[0]:slwin_values[-1]]
        as2 = epochs[1, sniff_i, slwin_values[0]:slwin_values[-1]]

        # collect "eulerized" phase angle differences
        cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
        
        # compute ISPC and PLI (and average over trials!)
        dfc[sniff_i, 0, slwin_values_i] = np.abs(np.mean(cdd))
        dfc[sniff_i, 1, slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
        dfc[sniff_i, 2, slwin_values_i] = np.abs( np.mean(np.abs(np.imag(cdd))*np.sign(np.imag(cdd))) ) / np.mean(np.abs(np.imag(cdd)))


dfc_mean = np.mean(dfc, axis=0)

plt.pcolormesh(times, range(4), dfc_mean)
plt.vlines(0, ymin=0, ymax=3, color='r')
plt.show()



plt.plot(times, dfc_mean[0, :], label='ispc')
plt.plot(times, dfc_mean[1, :], label='pli')
plt.plot(times, dfc_mean[2, :], label='wpli')
plt.vlines(0, ymin=dfc_mean.min(), ymax=dfc_mean.max(), color='r')
plt.legend()
plt.show()



















def load_data_sujet_test(sujet, cond, band_prep=None):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    if cond == 'FR_CV' :

        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
                load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        raw = mne.io.read_raw_fif(load_list[0], preload=True, verbose='critical')

        data = raw.get_data()

        del raw

    elif cond == 'SNIFF' :

        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if (session_name.find(cond) != -1) and (session_name.find('session') != -1) and ( session_name.find(band_prep) != -1 ):
                load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        raw = mne.io.read_raw_fif(load_list[0], preload=True, verbose='critical')

        data = raw.get_data()
        
        #data = xr.open_dataset(load_list[0])


    elif cond == 'AL' :
    
        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if ( session_name.find(cond) != -1 ) and ( session_name.find(band_prep) != -1 ) and ( session_name.find('session') == -1 ):
                load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data = []
        for load_i in load_list:
            raw = mne.io.read_raw_fif(load_i, preload=True, verbose='critical')
            
            data.append(raw.get_data())

        del raw
    
    
    elif cond == 'AC' :
    
        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if (session_name.find(cond) != -1) and ( session_name.find(band_prep) != -1 ) :
                load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        raw = mne.io.read_raw_fif(load_list[0], preload=True, verbose='critical')

        data = raw.get_data()

        del raw
    
    #### go back to path source
    os.chdir(path_source)

    return data






for sujet in sujet_list:

    prms = get_params(sujet)

    data = load_data_sujet_test(sujet, 'AC', band_prep='lf')
    ac_starts = get_ac_starts(sujet)

    plt.plot(data[-3,:])
    plt.vlines(ac_starts, ymin=data[-3,:].min(), ymax=data[-3,:].max(), color='r')
    plt.show()

    if sujet == 'pat_03083_1527':
        ac_starts = [3.184e4, 4.489e4, 5.726e4, 7.019e4, 8.481e4, 9.888e4, 1.1405e5, 1.2659e5, 1.3910e5, 1.5191e5, 1.6763e5, 2.5844e5,
                    2.7198e5, 3.1681e5, 3.2999e5, 3.4485e5, 3.5846e5, 3.7277e5, 3.8814e5, 4.0222e5, 4.9075e5, 5.0417e5, 5.2017e5,
                    5.3544e5, 5.5329e5, 5.6718e5, 5.8074e5, 5.9496e5, 6.1442e5, 6.3052e5]

    x = data[-3,:]
    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])

    data_chunk = np.zeros(( len(ac_starts), stretch_point_TF_ac ))

    for start_i, start_time in enumerate(ac_starts):

        t_start = int(start_time + t_start_AC*prms['srate'])
        t_stop = int(start_time + t_stop_AC*prms['srate'])

        data_chunk[start_i,:] = x[t_start: t_stop]


    times = np.arange(-5, 15, 1/prms['srate'])
    for i, _ in enumerate(ac_starts):
        plt.plot(times, data_chunk[i,:])
    plt.vlines(0, ymax=data_chunk.max(), ymin=data_chunk.min(), color='r')
    plt.show()

    plt.plot(times, data_chunk.mean(axis=0), label=sujet)
    plt.vlines([0, 10, 11], ymax=data_chunk.mean(axis=0).max(), ymin=data_chunk.mean(axis=0).min(), color='r')
plt.legend()
plt.show()





def zscore(x):

    _x = ( x - np.mean(x) ) / np.std(x)

    return _x

data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate

times = np.arange(0,data_ieeg.shape[1])/srate

select = [1]

plt.plot(times, zscore(data_aux[0,:])+1)
for select_i in select:
    plt.plot(times, zscore(data_ieeg[select_i,:]))
plt.show()












