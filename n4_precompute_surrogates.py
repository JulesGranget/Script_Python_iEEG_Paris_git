
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

for band_prep in band_prep_list:

    raw_tmp = {}
    for cond in conditions:

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data = []
        for load_name in load_list:
            data.append(mne.io.read_raw_fif(load_name, preload=True))

        raw_tmp[cond] = data

    raw_allcond[band_prep] = raw_tmp


srate = int(raw_allcond.get(band_prep_list[0])[os.listdir()[0][5:10]][0].info['sfreq'])
chan_list = raw_allcond.get(band_prep_list[0])[os.listdir()[0][5:10]][0].info['ch_names']



########################################
######## LOAD RESPI FEATURES ########
########################################


respfeatures_allcond = {}
band_prep = 'lf'

for cond in conditions:

    if len(raw_allcond.get(band_prep)[cond]) == 1:

        load_data = []
        resp_features, trash, trash = analyse_resp(raw_allcond.get(band_prep)[cond][0].get_data()[-3,:], srate, 0, condition=cond)
        load_data.append(resp_features)
    
    elif len(raw_allcond.get(band_prep)[cond]) > 1:

        load_data = []

        for i in range(len(raw_allcond.get(band_prep)[cond])):

            resp_features, trash, trash = analyse_resp(raw_allcond.get(band_prep)[cond][i].get_data()[-3,:], srate, 0, condition=cond)
            load_data.append(resp_features)

    else:

        continue

    respfeatures_allcond[cond] = load_data



################################
######## STRETCH ########
################################


#resp_features, stretch_point_surrogates, data = resp_features_CV, srate*2, data_CV[0,:]
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
######## PARAMS SURROGATES ########
########################################


nwind = int( 20*srate ) # window length in seconds*srate
nfft = nwind*5 # if no zero padding nfft = nwind
noverlap = np.round(nwind/2) # number of points of overlap here 50%
hannw = scipy.signal.windows.hann(nwind) # hann window






################################################
######## PRECOMPUTE AND SAVE SURROGATES ########
################################################

#### compute

def get_shuffle(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift


def precompute_surrogates_coh(cond, session_i, raw_allcond, band_prep):
    
    print(cond)

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy') == True :
        print('ALREADY COMPUTED')
        return


    respi_i = chan_list.index('nasal')
    respi = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[respi_i,:]

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    surrogates_n_chan = np.zeros((np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0),len(hzCxy)))

    for n_chan in range(np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)):

        chan_name = chan_list[n_chan]

        print('{:.2f}'.format(n_chan/np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)))

        x = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[n_chan,:]
        y = respi

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))
        for surr_i in range(n_surrogates_coh):
            
            if surr_i%100 == 0:
                print(surr_i) 

            x_shift = get_shuffle(x)
            #y_shift = get_shuffle(y)
            hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

            surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
        surrogates_n_chan[n_chan,:] = surrogates_val_tmp_sorted[percentile_i,:]

    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_Coh.npy', surrogates_n_chan)




def precompute_surrogates_cyclefreq(cond, session_i, raw_allcond, respfeatures_allcond, band_prep):
    
    print(cond) 

    if os.path.exists(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy') == True :
        print('ALREADY COMPUTED')
        return

    surrogates_n_chan = np.zeros((3,np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0), stretch_point_surrogates))

    respfeatures_i = respfeatures_allcond[cond][session_i]

    for n_chan in range(np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)):

        chan_name = chan_list[n_chan]

        print('{:.2f}'.format(n_chan/np.size(raw_allcond.get(band_prep).get(cond)[session_i].get_data(),0)))

        x = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[n_chan,:]

        surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq,stretch_point_surrogates))
        for surr_i in range(n_surrogates_cyclefreq):
            
            if surr_i%10 == 0:
                print(surr_i)

            x_shift = get_shuffle(x)
            #y_shift = get_shuffle(y)

            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift)

            x_stretch_mean = np.mean(x_stretch, axis=0)

            surrogates_val_tmp[surr_i,:] = x_stretch_mean

        mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
        percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

        surrogates_n_chan[0,n_chan,:] = mean_surrogate_tmp
        surrogates_n_chan[1,n_chan,:] = surrogates_val_tmp_sorted[percentile_i_up,:]
        surrogates_n_chan[2,n_chan,:] = surrogates_val_tmp_sorted[percentile_i_dw,:]

    np.save(sujet + '_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy', surrogates_n_chan)





################################
######## EXECUTE ########
################################


#### compute and save

os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))


for band_prep in band_prep_list:

    for cond in conditions:

        if len(respfeatures_allcond.get(cond)) == 1:

            precompute_surrogates_cyclefreq(cond, 0, raw_allcond, respfeatures_allcond, band_prep)

            if band_prep == 'lf':
                precompute_surrogates_coh(cond, 0, raw_allcond, band_prep)

        elif len(respfeatures_allcond.get(cond)) > 1:

            for session_i in range(len(respfeatures_allcond.get(cond))):

                precompute_surrogates_cyclefreq(cond, session_i, raw_allcond, respfeatures_allcond, band_prep)

                if band_prep == 'lf':
                    precompute_surrogates_coh(cond, session_i, raw_allcond, band_prep)







