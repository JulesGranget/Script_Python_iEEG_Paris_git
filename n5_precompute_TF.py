
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

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





################################
######## STRETCH TF ########
################################


#resp_features, stretch_point_TF, data = resp_features_CV, srate*2, data_CV[0,:]
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


#condition, resp_features, freq_band, stretch_point_TF = 'CV', list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf(tf, cond, session_i, respfeatures_allcond, stretch_point_TF):

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):

        print('{:.2f}'.format(n_chan/np.size(tf,0)))

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        tf_mean_allchan[n_chan,:,:] = tf_mean

    return tf_mean_allchan

#compute_stretch_tf(condition, resp_features_CV, freq_band, stretch_point_TF)




#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_dB(tf, cond, session_i, respfeatures_allcond, stretch_point_TF):

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    baseline = np.mean(tf, axis=2)
    baseline = np.transpose(baseline)

    db_tf = np.zeros((np.size(tf,0), np.size(tf,1), np.size(tf,2)))
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baseline[fi,n_chan]

            db_tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    tf = db_tf

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), int(stretch_point_TF)))

    for n_chan in range(np.size(tf,0)):

        print('{:.2f}'.format(n_chan/np.size(tf,0)))

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        tf_mean_allchan[n_chan,:,:] = tf_mean


    return tf_mean_allchan


#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_itpc(tf, cond, session_i, respfeatures_allcond, stretch_point_TF):
    
    #### identify number stretch
    x = tf[0,:]
    x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x)
    nb_cycle = np.size(x_stretch, 0)
    
    #### compute tf
    tf_stretch = np.zeros((nb_cycle, np.size(tf,0), int(stretch_point_TF)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        x_stretch, ratio = stretch_data(respfeatures_allcond.get(cond)[session_i], stretch_point_TF, x)
        tf_stretch[:,fi,:] = x_stretch

    return tf_stretch






################################
######## PRECOMPUTE TF ########
################################


def precompute_tf(cond, session_i, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list):

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    #### select prep to load
    #band_prep_i, band_prep = 0, 'hf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[:-3,:]

        freq_band = freq_band_list[band_prep_i]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            if os.path.exists(sujet + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            if freq[0] == .05 :
                wavetime = np.arange(-60,60,1/srate) 
            elif freq[0] == 2 :
                wavetime = np.arange(-2,2,1/srate)
            else :
                wavetime = np.arange(-.5,.5,1/srate)

            #### select nfrex
            if band_prep == 'lf':
                nfrex = nfrex_lf
            if band_prep == 'hf':
                nfrex = nfrex_hf

            #### compute wavelets
            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()


            tf_allchan = np.zeros((np.size(data,0),nfrex,np.size(data,1)))
            for n_chan in range(np.size(data,0)):

                print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 


                tf_allchan[n_chan,:,:] = tf


            #### stretch
            print('STRETCH')
            tf_allband_stretched = compute_stretch_tf_dB(tf_allchan, cond, session_i, respfeatures_allcond, stretch_point_TF)
            
            #### save
            print('SAVE')
            np.save(sujet + '_tf_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', tf_allband_stretched)







def precompute_tf_itpc(cond, session_i, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list):

    os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

    #### select prep to load
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #### select data without aux chan
        data = raw_allcond.get(band_prep).get(cond)[session_i].get_data()[:-3,:]

        freq_band = freq_band_list[band_prep_i]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            if os.path.exists(sujet + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            if freq[0] == .05 :
                wavetime = np.arange(-60,60,1/srate) 
            elif freq[0] == 2 :
                wavetime = np.arange(-2,2,1/srate)
            else :
                wavetime = np.arange(-.5,.5,1/srate)

            #### select nfrex
            if band_prep == 'lf':
                nfrex = nfrex_lf
            if band_prep == 'hf':
                nfrex = nfrex_hf

            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()


            tf_allchan = np.zeros((np.size(data,0),nfrex,np.size(data,1)), dtype='complex')
            for n_chan in range(np.size(data,0)):

                print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')


                tf_allchan[n_chan,:,:] = tf

            #### stretch & itpc

            itpc_allchan = np.zeros((np.size(tf_allchan,0),np.size(tf_allchan,1),stretch_point_TF))
            print('STRETCH & ITPC')
            for n_chan in range(len(chan_list)-3):

                print("{:.2f}".format(n_chan/(len(chan_list)-3)))

                tf_nchan = tf_allchan[n_chan,:,:]

                #### stretch
                tf_stretch = compute_stretch_tf_itpc(tf_nchan, cond, session_i, respfeatures_allcond, stretch_point_TF)

                #### ITPC
                tf_angle = np.angle(tf_stretch)
                tf_cangle = np.exp(1j*tf_angle) 
                itpc = np.abs(np.mean(tf_cangle,0)) 

                itpc_allchan[n_chan,:,:] = itpc

                if debug == True:
                    time = range(stretch_point_TF)
                    frex = range(nfrex)
                    plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
                    plt.show()

            #### save
            print('SAVE')
            np.save(sujet + '_itpc_' + str(freq[0]) + '_' + str(freq[1]) + '_' + cond + '_' + str(session_i+1) + '.npy', itpc_allchan)

            tf_allchan = []
            itpc_allchan = []











########################################
######## EXECUTE AND SAVE ########
########################################

#### compute and save tf
for cond in conditions:

    print(cond)

    if len(respfeatures_allcond[cond]) == 1:
 
        precompute_tf(cond, 0, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list)
        precompute_tf_itpc(cond, 0, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list)
    
    elif len(respfeatures_allcond[cond]) > 1:

        for session_i in range(len(respfeatures_allcond[cond])):

            precompute_tf(cond, session_i, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list)
            precompute_tf_itpc(cond, session_i, ncycle, srate_dw, respfeatures_allcond, freq_band_list, band_prep_list)










































################################################################################################################################################
################################################################################################################################################





if debug == True:

    ################################################
    ######## COMPUTE DIFFERENTIAL TF ########
    ################################################


    #condition1, condition2 = 'FV', 'SV'
    def precompute_differential_tf(condition1, condition2, freq_band):

        for band_prep_i, band_prep in enumerate(band_prep_list):

            #band, freq = list(freq_band.items())[3] 
            for band, freq in freq_band.items():

                print(band, freq)


                os.chdir(path_prep+'\\'+'Diff')    
                if os.path.exists('tf_precompute_'+band+'_'+condition1+'min'+condition2+'.npy'):
                    print(band+' : already done')
                    continue


                #load data
                print('import1')
                os.chdir(path_prep+'\\'+condition1)
                tf_load1 = np.load('tf_precompute_'+band+'_'+condition1+'.npy')

                print('import2')
                os.chdir(path_prep+'\\'+condition2)
                tf_load2 = np.load('tf_precompute_'+band+'_'+condition2+'.npy')

                #reshape if both tf are not the same size
                if np.size(tf_load1,2) > np.size(tf_load2,2):
                    tf_load1 = tf_load1[:,:,:np.size(tf_load2,2)] 
                elif np.size(tf_load1,2) < np.size(tf_load2,2):
                    tf_load2 = tf_load2[:,:,:np.size(tf_load1,2)] 

                #diff
                tf_diff = tf_load1 - tf_load2

                #### select nfrex
                if band_prep == 'lf':
                    nfrex = nfrex_lf
                if band_prep == 'hf':
                    nfrex = nfrex_hf

                if debug == True :
                    time = np.arange(0,np.size(tf_diff,2))/srate
                    frex = np.linspace(freq[0],freq[1],nfrex)
                    for n_chan in range(np.size(tf_diff,0)):
                        chan_name = chan_list[n_chan]

                        plt.pcolormesh(time,frex,tf_diff[n_chan,:,:],vmin=np.min(tf_diff[n_chan,:,:]),vmax=np.max(tf_diff[n_chan,:,:]))
                        #plt.pcolormesh(time,frex,tf_load1[n_chan,:,:],vmin=np.min(tf_load1[n_chan,:,:]),vmax=np.max(tf_load1[n_chan,:,:]))
                        #plt.pcolormesh(time,frex,tf_load2[n_chan,:,:],vmin=np.min(tf_load2[n_chan,:,:]),vmax=np.max(tf_load2[n_chan,:,:]))
                        plt.xlabel('Time (s)')
                        plt.ylabel('Frequency (Hz)')
                        plt.title(chan_name)
                        plt.show()

                os.chdir(path_prep+'\\'+'Diff')
                np.save('tf_precompute_'+band+'_'+condition1+'min'+condition2+'.npy', tf_diff)

                tf_load1 = []
                tf_load2 = []
                tf_diff = []


