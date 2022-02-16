

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config import *


debug = False



########################################
######## GENERATE FOLDERS ########
########################################


#os.getcwd()
def create_folder(folder_name, construct_token):
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)
        print('create : ' + folder_name)
        construct_token += 1
    return construct_token

def generate_folder_structure(sujet):

    construct_token = 0

    os.chdir(path_general)
    
    construct_token = create_folder('Analyses', construct_token)
    construct_token = create_folder('Data', construct_token)
    construct_token = create_folder('Mmap', construct_token)

    #### Analyses
    os.chdir(os.path.join(path_general, 'Analyses'))
    construct_token = create_folder('preprocessing', construct_token)
    construct_token = create_folder('precompute', construct_token)
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('results', construct_token)
    construct_token = create_folder('protocole', construct_token)
    
        #### preprocessing
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'preprocessing', sujet))
    construct_token = create_folder('sections', construct_token)
    construct_token = create_folder('info', construct_token)
    construct_token = create_folder('baseline', construct_token)

        #### precompute
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', sujet))
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)

        #### anatomy
    os.chdir(os.path.join(path_general, 'Analyses', 'anatomy'))
    construct_token = create_folder(sujet, construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet))
    construct_token = create_folder('RESPI', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
    construct_token = create_folder('HRV', construct_token)

            #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### PSD_Coh
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'PSD_Coh'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'ITPC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

            #### FC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC'))
    construct_token = create_folder('PLI', construct_token)
    construct_token = create_folder('ISPC', construct_token)

                #### PLI
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC', 'PLI'))
    construct_token = create_folder('figures', construct_token)
    construct_token = create_folder('matrix', construct_token)

                #### ISPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC', 'ISPC'))
    construct_token = create_folder('figures', construct_token)
    construct_token = create_folder('matrix', construct_token)

    #### Data
    os.chdir(os.path.join(path_general, 'Data'))
    construct_token = create_folder('raw_data', construct_token)

        #### raw_data
    os.chdir(os.path.join(path_general, 'Data', 'raw_data'))    
    construct_token = create_folder(sujet, construct_token)
    
            #### anatomy
    os.chdir(os.path.join(path_general, 'Data', 'raw_data', sujet))    
    construct_token = create_folder('anatomy', construct_token)

    return construct_token



############################
######## LOAD DATA ########
############################

def extract_chanlist_srate_conditions(conditions_allsubjects):

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    dirlist_subject = os.listdir()

    conditions = []
    for cond in conditions_allsubjects:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                conditions.append(cond)
                break

    #### extract data
    band_prep = band_prep_list[0]
    cond = conditions[0]

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(session_i)
        else:
            continue

    load_name = [os.listdir()[i] for i in load_i][0]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    srate = int(raw.info['sfreq'])
    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # on enlève : nasal, ventral, ECG, ECG_cR

    #### go back to path source
    os.chdir(path_source)

    return conditions, chan_list, chan_list_ieeg, srate


def extract_chanlist_srate_conditions_for_sujet(sujet_tmp, conditions_allsubjects):

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))
    dirlist_subject = os.listdir()

    conditions = []
    for cond in conditions_allsubjects:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                conditions.append(cond)
                break

    #### extract data
    band_prep = band_prep_list[0]
    cond = conditions[0]

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(session_i)
        else:
            continue

    load_name = [os.listdir()[i] for i in load_i][0]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    srate = int(raw.info['sfreq'])
    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # on enlève : nasal, ventral, ECG, ECG_cR

    #### go back to path source
    os.chdir(path_source)

    return conditions, chan_list, chan_list_ieeg, srate



def load_data(band_prep, cond, session_i):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]
    load_name = load_list[session_i]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    data = raw.get_data() 

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return data


def load_data_sujet(sujet_tmp, band_prep, cond, session_i):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond) != -1 ) & ( session_name.find(band_prep) != -1 ):
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]
    load_name = load_list[session_i]

    raw = mne.io.read_raw_fif(load_name, preload=True, verbose='critical')

    data = raw.get_data() 

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return data

def get_srate(sujet):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw = mne.io.read_raw_fif(sujet + '_FR_CV_1_lf.fif', preload=True, verbose='critical')
    
    srate = int(raw.info['sfreq'])

    #### go back to path source
    os.chdir(path_source)

    #### free memory
    del raw

    return srate



########################################
######## LOAD RESPI FEATURES ########
########################################

def load_respfeatures(conditions):

    path_source = os.getcwd()
    
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
    
    #### go back to path source
    os.chdir(path_source)

    return respfeatures_allcond




def get_all_respi_ratio(conditions, respfeatures_allcond):
    
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

    return respi_ratio_allcond


################################
######## STRETCH ########
################################


#resp_features, stretch_point_surrogates, data = resp_features_CV, srate*2, data_CV[0,:]
def stretch_data(resp_features, nb_point_by_cycle, data, srate):

    # params
    cycle_times = resp_features[['inspi_time', 'expi_time']].values
    mean_cycle_duration = np.mean(resp_features[['insp_duration', 'exp_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,np.size(data))/srate

    # stretch
    if stretch_TF_auto:
        clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                data, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=mean_inspi_ratio)
    else:
        clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                data, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=ratio_stretch_TF)

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
######## LOAD LOCALIZATION ########
########################################


def get_electrode_loca():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_txt = open(sujet + '_chanlist_ieeg.txt', 'r')
    chan_list_txt_readlines = chan_list_txt.readlines()
    chan_list_ieeg_trc = [i.replace('\n', '') for i in chan_list_txt_readlines]

    if sujet[:3] == 'pat':
        chan_list_ieeg_csv = chan_list_ieeg_trc.copy()
    else:
        chan_list_ieeg_csv, trash = modify_name(chan_list_ieeg_trc)

    loca_ieeg = []
    for chan_name in chan_list_ieeg_csv:
        loca_ieeg.append( str(file_plot_select['localisation_corrected'][file_plot_select['plot'] == chan_name].values.tolist()[0]) )

    dict_loca = {}
    for nchan_i, chan_name in enumerate(chan_list_ieeg_trc):
        dict_loca[chan_name] = loca_ieeg[nchan_i]


    return dict_loca



def get_loca_df():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_txt = open(sujet + '_chanlist_ieeg.txt', 'r')
    chan_list_txt_readlines = chan_list_txt.readlines()
    chan_list_ieeg_trc = [i.replace('\n', '') for i in chan_list_txt_readlines]

    if sujet[:3] == 'pat':
        chan_list_ieeg_csv = chan_list_ieeg_trc.copy()
    else:
        chan_list_ieeg_csv, trash = modify_name(chan_list_ieeg_trc)
        chan_list_ieeg_csv.sort()

    ROI_ieeg = []
    lobes_ieeg = []
    for chan_name in chan_list_ieeg_csv:
        ROI_ieeg.append( file_plot_select['localisation_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )
        lobes_ieeg.append( file_plot_select['lobes_corrected'].loc[file_plot_select['plot'] == chan_name].values.tolist()[0] )

    dict_loca = {'name' : chan_list_ieeg_trc,
                'ROI' : ROI_ieeg,
                'lobes' : lobes_ieeg
                }

    df_loca = pd.DataFrame(dict_loca, columns=dict_loca.keys())

    return df_loca


def get_mni_loca():

    os.chdir(os.path.join(path_anatomy, sujet))

    file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')

    chan_list_txt = open(sujet + '_chanlist_ieeg.txt', 'r')
    chan_list_txt_readlines = chan_list_txt.readlines()
    chan_list_ieeg = [i.replace('\n', '') for i in chan_list_txt_readlines]
    chan_list_ieeg, trash = modify_name(chan_list_ieeg)
    chan_list_ieeg.sort()

    mni_loc = file_plot_select['MNI']

    dict_mni = {}
    for chan_name in chan_list_ieeg:
        mni_nchan = file_plot_select['MNI'].loc[file_plot_select['plot'] == chan_name].values[0]
        mni_nchan = mni_nchan[1:-1]
        mni_nchan_convert = [float(mni_nchan.split(',')[0]), float(mni_nchan.split(',')[1]), float(mni_nchan.split(',')[2])]
        dict_mni[chan_name] = mni_nchan_convert

    return dict_mni


########################################
######## CHANGE NAME CSV TRC ########
########################################


def modify_name(chan_list):
    
    chan_list_modified = []
    chan_list_keep = []

    for nchan in chan_list:

        #### what we remove
        if nchan.find("+") != -1:
            continue

        if np.sum([str.isalpha(str_i) for str_i in nchan]) >= 2 and nchan.find('p') == -1:
            continue

        if nchan.find('ECG') != -1:
            continue

        if nchan.find('.') != -1:
            continue

        if nchan.find('*') != -1:
            continue

        #### what we do to chan we keep
        else:

            nchan_mod = nchan.replace(' ', '')
            nchan_mod = nchan_mod.replace("'", 'p')

            if nchan_mod.find('p') != -1:
                split = nchan_mod.split('p')
                letter_chan = split[0]

                if len(split[1]) == 1:
                    num_chan = '0' + split[1] 
                else:
                    num_chan = split[1]

                chan_list_modified.append(letter_chan + 'p' + num_chan)
                chan_list_keep.append(nchan)
                continue

            if nchan_mod.find('p') == -1:
                letter_chan = nchan_mod[0]

                split = nchan_mod[1:]

                if len(split) == 1:
                    num_chan = '0' + split
                else:
                    num_chan = split

                chan_list_modified.append(letter_chan + num_chan)
                chan_list_keep.append(nchan)
                continue


    return chan_list_modified, chan_list_keep



