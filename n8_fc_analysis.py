

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


def get_loca_df():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_ieeg = file_plot_select['Contact'].loc[file_plot_select['Select'] == 1].values.tolist()

    nasal_name = aux_chan.get(sujet).get('nasal')
    chan_list_ieeg.remove(nasal_name)

    ventral_name = aux_chan.get(sujet).get('ventral')
    chan_list_ieeg.remove(ventral_name)

    ecg_name = aux_chan.get(sujet).get('ECG')
    chan_list_ieeg.remove(ecg_name)

    ROI_ieeg = []
    lobes_ieeg = []
    for chan_name in chan_list_ieeg:
        ROI_ieeg.append( file_plot_select['Localisation_corrected'].loc[file_plot_select['Contact'] == chan_name].values.tolist()[0] )
        lobes_ieeg.append( file_plot_select['Lobes_corrected'].loc[file_plot_select['Contact'] == chan_name].values.tolist()[0] )

    dict_loca = {'name' : chan_list_ieeg,
                'ROI' : ROI_ieeg,
                'lobes' : lobes_ieeg
                }

    df_loca = pd.DataFrame(dict_loca, columns=dict_loca.keys())

    return df_loca

def get_mni_loca():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_ieeg = file_plot_select['Contact'].loc[file_plot_select['Select'] == 1].values.tolist()

    nasal_name = aux_chan.get(sujet).get('nasal')
    chan_list_ieeg.remove(nasal_name)

    ventral_name = aux_chan.get(sujet).get('ventral')
    chan_list_ieeg.remove(ventral_name)

    ecg_name = aux_chan.get(sujet).get('ECG')
    chan_list_ieeg.remove(ecg_name)

    mni_loc = file_plot_select['MNI']

    dict_mni = {}
    for chan_name in chan_list_ieeg:
        mni_nchan = file_plot_select['MNI'].loc[file_plot_select['Contact'] == chan_name].values[0]
        mni_nchan = mni_nchan[1:-1]
        mni_nchan_convert = [float(mni_nchan.split(',')[0]), float(mni_nchan.split(',')[1]), float(mni_nchan.split(',')[2])]
        dict_mni[chan_name] = mni_nchan_convert

    return dict_mni


df_loca = get_loca_df()
dict_mni = get_mni_loca()


#######################################
############# ISPC #############
#######################################

#raw, freq = raw_allcond.get(band_prep).get(cond)[session_i], [2, 10]
def compute_fc_metrics(raw, freq):
    
    #### wavelets computation
    frex  = np.linspace(freq[0],freq[1],nfrex)

    if freq[0] == .05 :
        wavetime = np.arange(-60,60,1/srate) 
    elif freq[0] == 2 :
        wavetime = np.arange(-2,2,1/srate)
    else :
        wavetime = np.arange(-.5,.5,1/srate)

    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    for fi in range(0,nfrex):
    
        s = ncycle / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    #### load data
    #data = raw.get_data()[:-3,:]
    data = raw.get_data()[:-3,:10]

    #### compute all convolution
    convolutions = {}

    print('CONV')

    for nchan in range(np.size(data,0)) :

        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')
        nchan_name = chan_list_ieeg[nchan]

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan_name] = nchan_conv

    #### compute metrics
    pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
    ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

    print('COMPUTE')

    for seed in range(np.size(data,0)) :

        seed_name = chan_list_ieeg[seed]

        for nchan in range(np.size(data,0)) :

            nchan_name = chan_list_ieeg[nchan]

            if nchan == seed : 
                continue
                
            else :

                # initialize output time-frequency data
                ispc = np.zeros((nfrex))
                pli  = np.zeros((nfrex))

                # compute metrics
                for fi in range(nfrex):
                    
                    as1 = convolutions.get(seed_name)[fi,:]
                    as2 = convolutions.get(nchan_name)[fi,:]

                    # collect "eulerized" phase angle differences
                    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                    
                    # compute ISPC and PLI (and average over trials!)
                    ispc[fi] = np.abs(np.mean(cdd))
                    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

                pli_mat[seed,nchan] = np.mean(ispc,0)
                ispc_mat[seed,nchan] = np.mean(pli,0)

    return pli_mat, ispc_mat



#### compute pli, itpc

pli_allband = {}
ispc_allband = {}

for band_prep_i, band_prep in enumerate(band_prep_list):

    for band in freq_band_list[band_prep_i].keys():

        if band == 'whole' :

            continue

        else: 

            freq = freq_band_fc_analysis.get(band)

            pli_allcond = {}
            ispc_allcond = {}

            for cond_i, cond in enumerate(conditions) :

                print(band, cond)

                if len(raw_allcond.get(band_prep).get(cond)) == 1:

                    pli_mat, ispc_mat = compute_fc_metrics(raw_allcond.get(band_prep).get(cond)[0], freq)
                    pli_allcond[cond] = [pli_mat]
                    ispc_allcond[cond] = [ispc_mat]

                elif len(raw_allcond.get(band_prep).get(cond)) > 1:

                    load_ispc = []
                    load_pli = []

                    for session_i in range(len(raw_allcond.get(band_prep).get(cond))):

                        pli_mat, ispc_mat = compute_fc_metrics(raw_allcond.get(band_prep).get(cond)[session_i], freq)
                        load_ispc.append(ispc_mat)
                        load_pli.append(pli_mat)

                    pli_allcond[cond] = load_pli
                    ispc_allcond[cond] = load_ispc

            pli_allband[band] = pli_allcond
            ispc_allband[band] = ispc_allcond

#### verif

if debug == True:

    for band, freq in freq_band_fc_analysis.items():

        for cond_i, cond in enumerate(conditions) :

            print(band, cond, len(pli_allband.get(band).get(cond)))
            print(band, cond, len(ispc_allband.get(band).get(cond)))



#### reduce to one cond

#### generate dict to fill
ispc_allband_reduced = {}
pli_allband_reduced = {}

for band, freq in freq_band_fc_analysis.items():

    ispc_allband_reduced[band] = {}
    pli_allband_reduced[band] = {}

    for cond_i, cond in enumerate(conditions) :

        ispc_allband_reduced.get(band)[cond] = []
        pli_allband_reduced.get(band)[cond] = []

#### fill
for band_prep_i, band_prep in enumerate(band_prep_list):

    for band, freq in freq_band_list[band_prep_i].items():

        if band == 'whole' :

            continue

        else:

            for cond_i, cond in enumerate(conditions) :

                if len(raw_allcond.get(band_prep).get(cond)) == 1:

                    ispc_allband_reduced.get(band)[cond] = ispc_allband.get(band).get(cond)[0]
                    pli_allband_reduced.get(band)[cond] = pli_allband.get(band).get(cond)[0]

                elif len(raw_allcond.get(band_prep).get(cond)) > 1:

                    load_ispc = []
                    load_pli = []

                    for session_i in range(len(raw_allcond.get(band_prep).get(cond))):

                        if session_i == 0 :

                            load_ispc.append(ispc_allband.get(band).get(cond)[session_i])
                            load_pli.append(pli_allband.get(band).get(cond)[session_i])

                        else :
                        
                            load_ispc = (load_ispc[0] + ispc_allband.get(band).get(cond)[session_i]) / 2
                            load_pli = (load_pli[0] + pli_allband.get(band).get(cond)[session_i]) / 2

                    pli_allband_reduced.get(band)[cond] = load_pli
                    ispc_allband_reduced.get(band)[cond] = load_ispc




########################
######## SAVE FIG ########
########################

#### sort matrix

def sort_ispc(mat):

    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    #### for rows
    for i_before_sort, i_sort in enumerate(df_sorted.index.values):
        mat_sorted[i_before_sort,:] = mat[i_sort,:]

    #### for columns
    for i_before_sort, i_sort in enumerate(df_sorted.index.values):
        mat_sorted[:,i_before_sort] = mat[:,i_sort]

    return mat_sorted



#### ISPC
os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC'))

df_sorted = df_loca.sort_values(['lobes', 'ROI'])
chan_name_sorted = df_sorted['ROI'].values.tolist()

    #### count for cond subpolt
if len(conditions) == 1:
    nrows, ncols = 1, 0
elif len(conditions) == 2:
    nrows, ncols = 1, 1
elif len(conditions) == 3:
    nrows, ncols = 2, 1
elif len(conditions) == 4:
    nrows, ncols = 2, 2
elif len(conditions) == 5:
    nrows, ncols = 3, 2
elif len(conditions) == 6:
    nrows, ncols = 3, 3



#band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
for band, freq in freq_band_fc_analysis.items():

    fig = plt.figure(facecolor='black')
    for cond_i, cond in enumerate(conditions):
        mne.viz.plot_connectivity_circle(sort_ispc(ispc_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
    plt.suptitle('ISPC_' + band, color='w')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    #fig.show()

    fig.savefig(sujet + '_ISPC_' + band, dpi = 600)


#### PLI

os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI'))

#band_prep_i, band_prep, nchan, band, freq = 0, 'lf', 0, 'theta', [2, 10]
for band, freq in freq_band_fc_analysis.items():

    fig = plt.figure(facecolor='black')
    for cond_i, cond in enumerate(conditions):
        mne.viz.plot_connectivity_circle(sort_ispc(pli_allband_reduced.get(band).get(cond)), node_names=chan_name_sorted, n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, cond_i+1))
    plt.suptitle('PLI_' + band, color='w')
    fig.set_figheight(10)
    fig.set_figwidth(12)
    #fig.show()

    fig.savefig(sujet + '_PLI_' + band, dpi = 600)









################################################################################################################################################
################################################################################################################################################


########################
######## MNE ########
########################


if debug == True:
        
    montage = mne.channels.make_dig_montage(dict_mni)
    trans = mne.channels.compute_native_head_t(montage)

    #### install pyvista, mayavi, 
    raw = raw_allcond.get('lf').get('RD_CV')[0]
    raw.set_montage(montage)
    fig = mne.viz.plot_alignment(raw.info, show_axes=True, surfaces='auto')


    mne.sys_info()

