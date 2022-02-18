

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
from n3_respi_analysis import analyse_resp

from n0_config import *

debug = False


############################
######## LOAD DATA ########
############################

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


srate = int(raw_allcond.get(band_prep_list[0])[os.listdir()[0][5:10]][0].info['sfreq'])
chan_list = raw_allcond.get(band_prep_list[0])[os.listdir()[0][5:10]][0].info['ch_names']
chan_list_ieeg = chan_list[:-3]


########################################
######## LOAD RESPI FEATURES ########
########################################

os.chdir(os.path.join(path_respfeatures, sujet, 'RESPI'))
respfeatures_listdir = os.listdir()

#### remove fig0 and fig1 file
respfeatures_listdir_clean = []
for file in respfeatures_listdir :
    if file.find('fig') == -1 :
        respfeatures_listdir_clean.append(file)

#### get respi features
respfeatures_allcond = {}

for cond in conditions:

    load_i = []
    for session_i, session_name in enumerate(respfeatures_listdir_clean):
        if session_name.find(cond) > 0:
            load_i.append(session_i)
        else:
            continue

    load_list = [respfeatures_listdir_clean[i] for i in load_i]

    data = []
    for load_name in load_list:
        data.append(pd.read_excel(load_name))

    respfeatures_allcond[cond] = data

#### get respi ratio for TF
respi_ratio_allcond = {}

for cond in conditions:

    if len(respfeatures_allcond.get(cond)) == 1:

        mean_cycle_duration = np.mean(respfeatures_allcond.get(cond)[0][['insp_duration', 'exp_duration']].values, axis=0)
        mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

        respi_ratio_allcond[cond] = [ mean_inspi_ratio ]

    elif len(respfeatures_allcond.get(cond)) > 1:

        data_to_short = []

        for session_i in range(len(respfeatures_allcond.get(cond))):   
            
            if session_i == 0 :

                mean_cycle_duration = np.mean(respfeatures_allcond.get(cond)[session_i][['insp_duration', 'exp_duration']].values, axis=0)
                mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
                data_to_short = [ mean_inspi_ratio ]

            elif session_i > 0 :

                mean_cycle_duration = np.mean(respfeatures_allcond.get(cond)[session_i][['insp_duration', 'exp_duration']].values, axis=0)
                mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                data_replace = [(data_to_short[0] + mean_inspi_ratio) / 2]

                data_to_short = data_replace.copy()
        
        # to put in list
        respi_ratio_allcond[cond] = data_to_short 


########################################
######## LOAD LOCALIZATION ########
########################################


def get_electrode_loca():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_ieeg = file_plot_select['Contact'].loc[file_plot_select['Contact selection'] == 1].values.tolist()

    nasal_name = aux_chan.get(sujet).get('nasal')
    chan_list_ieeg.remove(nasal_name)

    ventral_name = aux_chan.get(sujet).get('ventral')
    chan_list_ieeg.remove(ventral_name)

    ecg_name = aux_chan.get(sujet).get('ECG')
    chan_list_ieeg.remove(ecg_name)

    chan_list_ieeg.extend(['nasal', 'ventral', 'ECG'])

    loca_ieeg = []
    for chan_name in chan_list_ieeg[:-3]:
        loca_ieeg.append( file_plot_select['LocalisationDeskian corrected'].loc[file_plot_select['Contact'] == chan_name].values.tolist()[0] )

    dict_loca = {}
    for nchan_i, chan_name in enumerate(chan_list_ieeg[:-3]):
        dict_loca[chan_name] = loca_ieeg[nchan_i]


    return dict_loca

dict_loca = get_electrode_loca()



########################
######## STRETCH ########
########################


#resp_features, nb_point_by_cycle, data = resp_features_CV, srate*2, data_CV[0,:]
def stretch_data(resp_features, nb_point_by_cycle, data):

    # params
    cycle_times = resp_features[['inspi_time', 'expi_time']].values
    mean_cycle_duration = np.mean(resp_features[['insp_duration', 'exp_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,np.size(data))/srate

    # stretch
    clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
            data, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=mean_inspi_ratio)

    nb_cycle = data_stretch_linear.shape[0]//nb_point_by_cycle
    phase = np.arange(nb_point_by_cycle)/nb_point_by_cycle
    data_stretch = data_stretch_linear.reshape(int(nb_cycle), int(nb_point_by_cycle))

    # inspect
    if debug == True:
        for i in range(int(nb_cycle)):
            plt.plot(data_stretch[i])
        plt.show()

        i = 1
        plt.plot(data_stretch[i])
        plt.show()

    return data_stretch, mean_inspi_ratio





########################################
######## PSD AND COH PARAMS ########
########################################


nwind = int( 20*srate ) # window length in seconds*srate
nfft = nwind*5 # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hann(nwind) # hann window






########################################
######## COMPUTE PSD AND COH ########
########################################



#### load surrogates
Cxy_surrogates_allcond = {}
cyclefreq_surrogates_allcond_lf = {}
cyclefreq_surrogates_allcond_hf = {}
os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

for cond in conditions:

    if len(respfeatures_allcond.get(cond)) == 1:

        data_load = []
        data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_Coh.npy'))
        Cxy_surrogates_allcond[cond] = data_load

        data_load = []
        data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_lf.npy'))
        cyclefreq_surrogates_allcond_lf[cond] = data_load

        data_load = []
        data_load.append(np.load(sujet + '_' + cond + '_' + str(1) + '_cyclefreq_hf.npy'))
        cyclefreq_surrogates_allcond_hf[cond] = data_load

    elif len(respfeatures_allcond.get(cond)) > 1:

        data_load = []

        for session_i in range(len(respfeatures_allcond.get(cond))):

            data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy'))
        
        Cxy_surrogates_allcond[cond] = data_load

        data_load = []

        for session_i in range(len(respfeatures_allcond.get(cond))):

            data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_lf.npy'))
        
        cyclefreq_surrogates_allcond_lf[cond] = data_load

        data_load = []

        for session_i in range(len(respfeatures_allcond.get(cond))):

            data_load.append(np.load(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_hf.npy'))
        
        cyclefreq_surrogates_allcond_hf[cond] = data_load

            




#### function
def compute_PxxCxyCyclefreq_for_cond(cond, session_i, nb_point_by_cycle, band_prep):
    
    print(cond)

    chan_i = chan_list.index('nasal')
    respi = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[chan_i,:]

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    Cxy_for_cond = np.zeros(( np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0), len(hzCxy)))
    Pxx_for_cond = np.zeros(( np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0), len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0), nb_point_by_cycle))

    for n_chan in range(np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)):

        print('{:.2f}'.format(n_chan/np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)))

        x = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

        x_stretch, trash = stretch_data(respfeatures_allcond.get(cond)[session_i], nb_point_by_cycle, x)
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond

        

#### compute
Pxx_allcond_lf = {}
Pxx_allcond_hf = {}
Cxy_allcond = {}
cyclefreq_allcond_lf = {}
cyclefreq_allcond_hf = {}

for band_prep in band_prep_list:

    print(band_prep)

    for cond in conditions:

        if ( len(respfeatures_allcond.get(cond)) == 1 ) & (band_prep == 'lf'):

            Pxx_load = []
            Cxy_load = []
            cyclefreq_load = []

            Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(cond, 0, stretch_point_surrogates, band_prep)

            Pxx_load.append(Pxx_for_cond)
            Cxy_load.append(Cxy_for_cond)
            cyclefreq_load.append(cyclefreq_for_cond)

            Pxx_allcond_lf[cond] = Pxx_load
            Cxy_allcond[cond] = Cxy_load
            cyclefreq_allcond_lf[cond] = cyclefreq_load

        elif ( len(respfeatures_allcond.get(cond)) == 1 ) & (band_prep == 'hf') :

            Pxx_load = []
            cyclefreq_load = []

            Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(cond, 0, stretch_point_surrogates, band_prep)

            Pxx_load.append(Pxx_for_cond)
            cyclefreq_load.append(cyclefreq_for_cond)

            Pxx_allcond_hf[cond] = Pxx_load
            cyclefreq_allcond_hf[cond] = cyclefreq_load


        elif (len(respfeatures_allcond.get(cond)) > 1) & (band_prep == 'lf'):

            Pxx_load = []
            Cxy_load = []
            cyclefreq_load = []

            for session_i in range(len(respfeatures_allcond.get(cond))):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(cond, session_i, stretch_point_surrogates, band_prep)

                Pxx_load.append(Pxx_for_cond)
                Cxy_load.append(Cxy_for_cond)
                cyclefreq_load.append(cyclefreq_for_cond)

            Pxx_allcond_lf[cond] = Pxx_load
            Cxy_allcond[cond] = Cxy_load
            cyclefreq_allcond_lf[cond] = cyclefreq_load

        elif (len(respfeatures_allcond.get(cond)) > 1) & (band_prep == 'hf'):

            Pxx_load = []
            cyclefreq_load = []

            for session_i in range(len(respfeatures_allcond.get(cond))):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(cond, session_i, stretch_point_surrogates, band_prep)

                Pxx_load.append(Pxx_for_cond)
                cyclefreq_load.append(cyclefreq_for_cond)

            Pxx_allcond_hf[cond] = Pxx_load
            cyclefreq_allcond_hf[cond] = cyclefreq_load




#### reduce data to one session
respfeatures_allcond_adjust = {} # to conserve respi_allcond for TF

for cond in conditions:

    if len(respfeatures_allcond.get(cond)) == 1:

        respfeatures_allcond_adjust[cond] = respfeatures_allcond[cond].copy()

    elif len(respfeatures_allcond.get(cond)) > 1:

        data_to_short = []

        for session_i in range(len(respfeatures_allcond.get(cond))):
            
            
            if session_i == 0 :

                data_to_short = [
                                respfeatures_allcond.get(cond)[session_i], 
                                Pxx_allcond_lf.get(cond)[session_i],
                                Pxx_allcond_hf.get(cond)[session_i], 
                                Cxy_allcond.get(cond)[session_i], 
                                Cxy_surrogates_allcond.get(cond)[session_i], 
                                cyclefreq_allcond_lf.get(cond)[session_i],
                                cyclefreq_allcond_hf.get(cond)[session_i], 
                                cyclefreq_surrogates_allcond_lf.get(cond)[session_i]
                                ]

            elif session_i > 0 :

                data_replace = [
                                (data_to_short[0] + respfeatures_allcond.get(cond)[session_i]) / 2, 
                                (data_to_short[1] + Pxx_allcond_lf.get(cond)[session_i]) / 2, 
                                (data_to_short[2] + Pxx_allcond_hf.get(cond)[session_i]) / 2, 
                                (data_to_short[3] + Cxy_allcond.get(cond)[session_i]) / 2, 
                                (data_to_short[4] + Cxy_surrogates_allcond.get(cond)[session_i]) / 2,   
                                (data_to_short[5] + cyclefreq_allcond_lf.get(cond)[session_i]) / 2,
                                (data_to_short[6] + cyclefreq_allcond_hf.get(cond)[session_i]) / 2,  
                                (data_to_short[7] + cyclefreq_surrogates_allcond_lf.get(cond)[session_i]) / 2
                                ]

                data_to_short = data_replace.copy()
        
        # to put in list
        data_load = []
        data_load.append(data_to_short[0])
        respfeatures_allcond_adjust[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[1])
        Pxx_allcond_lf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[2])
        Pxx_allcond_hf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[3])
        Cxy_allcond[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[4])
        Cxy_surrogates_allcond[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[5])
        cyclefreq_allcond_lf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[6])
        cyclefreq_allcond_hf[cond] = data_load 

        data_load = []
        data_load.append(data_to_short[7])
        cyclefreq_surrogates_allcond_lf[cond] = data_load 



#### verif if one session only
for cond in conditions :

    verif_size = []

    verif_size.append(len(respfeatures_allcond_adjust[cond]) == 1)
    verif_size.append(len(Pxx_allcond_lf[cond]) == 1)
    verif_size.append(len(Pxx_allcond_hf[cond]) == 1)
    verif_size.append(len(Cxy_allcond[cond]) == 1)
    verif_size.append(len(Cxy_surrogates_allcond[cond]) == 1)
    verif_size.append(len(cyclefreq_allcond_lf[cond]) == 1)
    verif_size.append(len(cyclefreq_allcond_hf[cond]) == 1)
    verif_size.append(len(cyclefreq_surrogates_allcond_lf[cond]) == 1)

    if verif_size.count(False) != 0 :
        print('!!!! PROBLEM VERIF !!!!')

    elif verif_size.count(False) == 0 :
        print('Verif OK')






################################################
######## PLOT & SAVE PSD AND COH ########
################################################

os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

#### for lf
for n_chan in range(len(chan_list_ieeg)):

    session_i = 0       
    
    chan_name = chan_list_ieeg[n_chan]
    print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### plot

    fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    for c, cond in enumerate(conditions):

        #### supress NaN
        keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
        cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
        respi_mean = round(np.mean(cycle_for_mean), 2)
        
        #### plot
        ax = axs[0,c]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')
        ax.set_xlim(0,60)

        ax = axs[1,c]
        ax.plot(hzPxx,Pxx_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.set_xlim(0, 2)
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_lf.get(cond)[session_i][n_chan,:]), color='r')

        ax = axs[2,c]
        ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(hzCxy,Cxy_surrogates_allcond.get(cond)[session_i][n_chan,:], color='c')
        ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

        ax = axs[3,c]
        ax.plot(cyclefreq_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][0, n_chan,:], color='b')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
        ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:] ), colors='r')


    #### save
    fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)




#### for hf
for n_chan in range(len(chan_list_ieeg)):       
    
    session_i = 0

    chan_name = chan_list_ieeg[n_chan]
    print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

    hzPxx = np.linspace(0,srate/2,int(nfft/2+1))

    #### plot

    fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
    plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

    for c, cond in enumerate(conditions):

        #### supress NaN
        keep = np.invert(np.isnan(respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values))
        cycle_for_mean = respfeatures_allcond_adjust.get(cond)[session_i]['cycle_freq'].values[keep]
        respi_mean = round(np.mean(cycle_for_mean), 2)
        
        #### plot
        ax = axs[0,c]
        ax.set_title(cond, fontweight='bold', rotation=0)
        ax.semilogy(hzPxx,Pxx_allcond_hf.get(cond)[session_i][n_chan,:], color='k')
        ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond_hf.get(cond)[session_i][n_chan,:]), color='r')
        ax.set_xlim(45,120)

        ax = axs[1,c]
        ax.plot(cyclefreq_allcond_lf.get(cond)[session_i][n_chan,:], color='k')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][0, n_chan,:], color='b')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
        ax.plot(cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')
        ax.vlines(respi_ratio_allcond.get(cond)[session_i]*stretch_point_surrogates, ymin=np.min( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][2, n_chan,:] ), ymax=np.max( cyclefreq_surrogates_allcond_lf.get(cond)[session_i][1, n_chan,:] ), colors='r')


    #### save
    fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)
















################################
######## LOAD TF ########
################################

#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'TF'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


#### load file with reducing to one TF

tf_stretch_allcond = {}

for cond in conditions:

    tf_stretch_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:
                        tf_stretch_onecond[band] = np.load(file)
                    else:
                        continue
                    
        tf_stretch_allcond[cond] = tf_stretch_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_stretch_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_stretch_onecond.get(band) ) /2
                            tf_stretch_onecond[band] = session_load_tmp

                        else:
                            
                            tf_stretch_onecond[band] = np.load(file)

                    else:

                        continue

        tf_stretch_allcond[cond] = tf_stretch_onecond




#### verif

for cond in conditions:
    if len(tf_stretch_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_stretch_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






################################
######## SAVE TF ########
################################



os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))

for freq_band_i, freq_band in enumerate(freq_band_list): 

    for n_chan in range(len(chan_list_ieeg)):       
        
        chan_name = chan_list_ieeg[n_chan]
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

        time = range(stretch_point_TF)
        frex = np.size(tf_stretch_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)

        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_stretch_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)










################################
######## LOAD ITPC ########
################################


#### load and reduce to all cond
os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

#### generate str to search file
freq_band_str = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    freq_band = freq_band_list[band_prep_i]

    for band, freq in freq_band.items():
        freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])

#### load file with reducing to one TF

tf_itpc_allcond = {}

for cond in conditions:

    tf_itpc_onecond = {}

    if len(respfeatures_allcond.get(cond)) == 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### impose good order in dict
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_stretch_onecond[band] = 0

        #### file load
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):
                    if file.find(freq_band_str.get(band)) != -1:
                        tf_itpc_onecond[ band ] = np.load(file)
                    else:
                        continue
                    
        tf_itpc_allcond[cond] = tf_itpc_onecond

    elif len(respfeatures_allcond.get(cond)) > 1:

        #### generate file to load
        load_file = []
        for file in os.listdir(): 
            if file.find(cond) != -1:
                load_file.append(file)
            else:
                continue

        #### implement count
        for freq_band in freq_band_list:
            for band, freq in freq_band.items():
                tf_itpc_onecond[band] = 0

        #### load file
        for file in load_file:

            for freq_band in freq_band_list:

                for i, (band, freq) in enumerate(freq_band.items()):

                    if file.find(freq_band_str.get(band)) != -1:

                        if np.sum(tf_itpc_onecond.get(band)) != 0:

                            session_load_tmp = ( np.load(file) + tf_itpc_onecond.get(band) ) /2
                            tf_itpc_onecond[band] = session_load_tmp

                        else:
                            
                            tf_itpc_onecond[band] = np.load(file)

                    else:

                        continue

        tf_itpc_allcond[cond] = tf_itpc_onecond


#### verif

for cond in conditions:
    if len(tf_itpc_allcond.get(cond)) != 6:
        print('ERROR COND : ' + cond)

    for freq_band in freq_band_list:

        for band, freq in freq_band.items():
            if len(tf_itpc_allcond.get(cond).get(band)) != len(chan_list_ieeg) :
                print('ERROR FREQ BAND : ' + band)
            






################################
######## SAVE ITPC ########
################################



os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))


for freq_band_i, freq_band in enumerate(freq_band_list): 

    for n_chan in range(len(chan_list_ieeg)):       
        
        chan_name = chan_list_ieeg[n_chan]
        print('{:.2f}'.format(n_chan/len(chan_list_ieeg)))

        time = range(stretch_point_TF)
        frex = np.size(tf_itpc_allcond.get(conditions[0]).get(list(freq_band.keys())[0]),1)

        if freq_band_i == 0:

            #### plot
            fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_lf.jpeg', dpi=600)

        elif freq_band_i == 1:

            #### plot
            fig, axs = plt.subplots(nrows=2, ncols=len(conditions))
            plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))

            for c, cond in enumerate(conditions):
                
                #### plot
                if c == 0:
                        
                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.set_ylabel(band)
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                else:

                    for i, (band, freq) in enumerate(freq_band.items()) :

                        data = tf_itpc_allcond.get(cond).get(band)[n_chan, :, :]
                        frex = np.linspace(freq[0], freq[1], np.size(data,0))
                    
                        if i == 0 :

                            ax = axs[i,c]
                            ax.set_title(cond, fontweight='bold', rotation=0)
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()

                        else :

                            ax = axs[i,c]
                            ax.pcolormesh(time, frex, data, vmin=np.min(data), vmax=np.max(data))
                            ax.vlines(respi_ratio_allcond.get(cond)[0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='r')
                            #plt.show()
                    
                
            #### save
            fig.savefig(sujet + '_' + chan_name + '_hf.jpeg', dpi=600)













































####################################################################

if debug == True:

    #### fig PSD COh all cond

    def PSD_Coh_fig_allcond(cond, session_i, respfeatures_allcond, Pxx_allcond, Cxy_allcond, Cxy_surrogates_allcond, cyclefreq_allcond, cyclefreq_surrogates_allcond):

        for cond in conditions:

            for n_chan in range(np.size(raw_allcond.get(cond)[0].get_data(),0)):       
                
                chan_name = chan_list[n_chan]
                print('{:.2f}'.format(n_chan/np.size(raw_allcond.get(cond)[0].get_data(),0)))

                hzPxx = np.linspace(0,srate/2,int(nfft/2+1))
                hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
                mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
                hzCxy = hzCxy[mask_hzCxy]

                respi_mean = round(np.mean(respfeatures_allcond.get(cond)[session_i]['cycle_freq'].values), 2)

                #### plot
                fig, axs = plt.subplots(nrows=2, ncols=2)
                plt.suptitle(sujet + '_' + chan_name + '_' + dict_loca.get(chan_name))
                        
                ax = axs[0,0]
                ax.set_title('Welch_full', fontweight='bold', rotation=0)
                ax.semilogy(hzPxx,Pxx_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.set_xlim(0,50)

                ax = axs[0,1]
                ax.set_title('Welch_Respi', fontweight='bold', rotation=0)
                ax.plot(hzPxx,Pxx_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.set_xlim(0, 2)
                ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond.get(cond)[session_i][n_chan,:]), color='b')

                ax = axs[1,0]
                ax.set_title('Cxy', fontweight='bold', rotation=0)
                ax.plot(hzCxy,Cxy_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.plot(hzCxy,Cxy_surrogates_allcond.get(cond)[session_i][n_chan,:], color='c')
                ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

                ax = axs[1,1]
                ax.set_title('CycleFreq', fontweight='bold', rotation=0)
                ax.plot(cyclefreq_allcond.get(cond)[session_i][n_chan,:], color='k')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][0, n_chan,:], color='c')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][1, n_chan,:], color='c', linestyle='dotted')
                ax.plot(cyclefreq_surrogates_allcond.get(cond)[session_i][2, n_chan,:], color='c', linestyle='dotted')


    ####################################################################################


