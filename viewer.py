

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import glob
import neo

from n0_config import *

debug = False


################################
######## PARAMETERS ########
################################

path_data_visu = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\iEEG\\preprocessing'


#sujet = 'CHEe'
#sujet = 'MAZm'
#sujet = 'MUGa' 
sujet = 'GOBc' 
#sujet = 'TREt' 

conditions = ['RD_CV', 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV']

raw_vizu = True

band_preproc = 'lf' # for theta alpha beta
#band_preproc = 'hf' # for gamma


########################################
######## LOAD FOR RAW VIZU ########
########################################

path_data_visu_raw = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Analyses\\iEEG\\preprocessing'

def extract_data_trc(path_data, sujet, aux_chan):

    os.chdir(os.path.join(path_data,sujet))

    # identify number of trc file
    trc_file_names = glob.glob('*.TRC')

    # extract file one by one
    print('EXTRACT TRC')
    data_whole = []
    chan_list_whole = []
    srate_whole = []
    events_name_whole = []
    events_time_whole = []
    #file_i, file_name = 0, trc_file_names[0]
    for file_i, file_name in enumerate(trc_file_names):

        # current file
        print(file_name)

        # extract segment with neo
        reader = neo.MicromedIO(filename=file_name)
        seg = reader.read_segment()
        print('len seg : ' + str(len(seg.analogsignals)))
        
        # extract data
        data_whole_file = []
        chan_list_whole_file = []
        srate_whole_file = []
        events_name_file = []
        events_time_file = []
        #anasig = seg.analogsignals[2]
        for seg_i, anasig in enumerate(seg.analogsignals):
            
            chan_list_whole_file.append(anasig.array_annotations['channel_names'].tolist()) # extract chan
            data_whole_file.append(anasig[:, :].magnitude.transpose()) # extract data
            srate_whole_file.append(int(anasig.sampling_rate.rescale('Hz').magnitude.tolist())) # extract srate

        if srate_whole_file != [srate_whole_file[i] for i in range(len(srate_whole_file))] :
            print('srate different in segments')
            exit()
        else :
            srate_file = srate_whole_file[0]

        # concatenate data
        for seg_i in range(len(data_whole_file)):
            if seg_i == 0 :
                data_file = data_whole_file[seg_i]
                chan_list_file = chan_list_whole_file[seg_i]
            else :
                data_file = np.concatenate((data_file,data_whole_file[seg_i]), axis=0)
                [chan_list_file.append(chan_list_whole_file[seg_i][i]) for i in range(np.size(chan_list_whole_file[seg_i]))]


        # event
        if len(seg.events[0].magnitude) == 0 : # when its VS recorded
            events_name_file = ['CV_start', 'CV_stop']
            events_time_file = [0, len(data_file[0,:])]
        else : # for other sessions
            #event_i = 0
            for event_i in range(len(seg.events[0])):
                events_name_file.append(seg.events[0].labels[event_i])
                events_time_file.append(int(seg.events[0].times[event_i].magnitude * srate_file))

        # fill containers
        data_whole.append(data_file)
        chan_list_whole.append(chan_list_file)
        srate_whole.append(srate_file)
        events_name_whole.append(events_name_file)
        events_time_whole.append(events_time_file)

    # concatenate 
    print('CONCATENATE')
    data = data_whole[0]
    chan_list = chan_list_whole[0]
    events_name = events_name_whole[0]
    events_time = events_time_whole[0]
    srate = srate_whole[0]

    if len(trc_file_names) > 1 :
        #trc_i = 0
        for trc_i in range(len(trc_file_names)): 

            if trc_i == 0 :
                len_trc = np.size(data_whole[trc_i],1)
                continue
            else:
                    
                data = np.concatenate((data,data_whole[trc_i]), axis=1)

                [events_name.append(events_name_whole[trc_i][i]) for i in range(len(events_name_whole[trc_i]))]
                [events_time.append(events_time_whole[trc_i][i] + len_trc) for i in range(len(events_time_whole[trc_i]))]

                if chan_list != chan_list_whole[trc_i]:
                    print('not the same chan list')
                    exit()

                if srate != srate_whole[trc_i]:
                    print('not the same srate')
                    exit()

                len_trc += np.size(data_whole[trc_i],1)

    print('AUX IDENTIFICATION')
    nasal_i = chan_list_file.index(aux_chan.get(sujet).get('nasal'))
    ventral_i = chan_list_file.index(aux_chan.get(sujet).get('ventral'))
    ecg_i = chan_list_file.index(aux_chan.get(sujet).get('ECG'))
    data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)
    chan_list_aux = ['nasal', 'ventral', 'ECG']
    
    chan_to_suppr = [aux_chan.get(sujet).get('nasal'), aux_chan.get(sujet).get('ventral'), aux_chan.get(sujet).get('ECG')] 
    for rmv in chan_to_suppr:
        chan_rmv_i = chan_list_file.index(rmv)
        data = np.delete(data, chan_rmv_i, 0)
        chan_list_file.remove(rmv)

    chan_list = chan_list_file.copy()

    return data, chan_list, data_aux, chan_list_aux

#### extract
data, chan_list, data_aux, chan_list_aux = extract_data_trc(path_data, sujet, aux_chan)

#### plot
chan_name = "B'18"

chan_i = chan_list.index(chan_name)

x = data[chan_i,:int(1e5)]
#y = data[respi_i,:]

x_zscore = (x-np.mean(x))/np.std(x)
#y_zscore = (y-np.mean(y))/np.std(y)

plt.plot(x_zscore, label=chan_name)
#plt.plot(y_zscore+1, label=respi)
plt.legend()
plt.show()








############################
######## LOAD DATA ########
############################

os.chdir(os.path.join(path_data_visu, sujet, 'sections'))

raw_allcond = {}

for cond in conditions:

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_preproc) != -1 ):
            load_i.append(session_i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]

    data = []
    for load_name in load_list:
        data.append(mne.io.read_raw_fif(load_name, preload=True))

    raw_allcond[cond] = data


srate = int(raw_allcond[os.listdir()[0][5:10]][0].info['sfreq'])
chan_list = raw_allcond[os.listdir()[0][5:10]][0].info['ch_names']

############################
######## VIEWER ########
############################


condition = 'RD_CV'
#condition = 'RD_FV'
#condition = 'RD_SV'
#condition = 'RD_AV'
#condition = 'FR_CV'
#condition = 'FR_MV'

chan_name = "J' 1"
respi = 'nasal'

session_i = 0

#### select raw
raw_select = raw_allcond[condition][session_i]






### plot mne
duration = 5
n_chan = 3
start = 0
raw_select.plot(scalings='auto', duration=duration, n_channels=n_chan, start=start) # verify




### plot sig
chan_i = chan_list.index(chan_name)
respi_i = chan_list.index(respi)

x = raw_select.get_data()[chan_i,:]
y = raw_select.get_data()[respi_i,:]

x_zscore = (x-np.mean(x))/np.std(x)
y_zscore = (y-np.mean(y))/np.std(y)

plt.plot(x_zscore, label=chan_name)
plt.plot(y_zscore+1, label=respi)
plt.legend()
plt.show()



### plot PSD
chan_i = chan_list.index(chan_name)
respi_i = chan_list.index(respi)

x = raw_select.get_data()[chan_i,:]
y = raw_select.get_data()[respi_i,:]


nwind = int( 20*srate )
nfft = nwind 
noverlap = np.round(nwind/2) 
hannw = scipy.signal.windows.hann(nwind)

hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)


plt.semilogy(hzPxx,Pxx)
plt.xlim([0,100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.show()








#### filter
raw_filt = raw_select.copy()

filter_length = int(srate*10) # in samples
h_freq = 2
l_freq = None

raw_filt = raw_filt.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hann', fir_design='firwin2')


x = raw_filt.get_data()[chan_i,:]
y = raw_filt.get_data()[respi_i,:]

x_zscore = (x-np.mean(x))/np.std(x)
y_zscore = (y-np.mean(y))/np.std(y)

plt.plot(x_zscore, label=chan_name)
plt.plot(y_zscore+1, label=respi)
plt.legend()
plt.show()



