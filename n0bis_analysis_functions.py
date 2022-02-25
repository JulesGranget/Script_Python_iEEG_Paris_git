

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import subprocess
import sys
import stat
import xarray as xr

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

        #### precompute
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute'))
    construct_token = create_folder(sujet, construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', sujet))
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('Baselines', construct_token)
    construct_token = create_folder('FC', construct_token)

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
    construct_token = create_folder('ERP', construct_token)

            #### ERP
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'ERP'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)
    

            #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

                #### summary
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'TF', 'summary'))
    construct_token = create_folder('AL', construct_token)

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
    construct_token = create_folder(sujet, construct_token)

        #### sujet
    os.chdir(os.path.join(path_general, 'Data', sujet))    
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('events', construct_token)
    construct_token = create_folder('raw_data', construct_token)
    
            #### events
    os.chdir(os.path.join(path_general, 'Data', sujet, 'events'))    
    construct_token = create_folder('mat', construct_token)
    construct_token = create_folder('ncs', construct_token)

            #### raw_data
    os.chdir(os.path.join(path_general, 'Data', sujet, 'raw_data'))    
    construct_token = create_folder('mat', construct_token)
    construct_token = create_folder('ncs', construct_token)

    return construct_token


################################
######## SLURM EXECUTE ########
################################


#name_script, name_function, params = 'test', 'slurm_test',  ['Pilote', 2]
def execute_function_in_slurm(name_script, name_function, params):

    python = sys.executable

    #### params to print in script
    params_str = ""
    for params_i in params:
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if params_i == params[0] :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for params_i in params:

        str_i = str(params_i)

        if params_i == params[0] :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run_function_{name_function}_{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
        
    subprocess.Popen(['sbatch', f'{slurm_script_name}', f'-cpus-per-task={n_core_slurms}', f'-mem={mem_crnl_cluster}']) 

    # wait subprocess to lauch before removing
    #time.sleep(3)
    #os.remove(slurm_script_name)

    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')






#name_script, name_function, params = 'n9_fc_analysis', 'compute_pli_ispc_allband', [sujet]
def execute_function_in_slurm_bash(name_script, name_function, params):

    scritp_path = os.getcwd()
    
    python = sys.executable

    #### params to print in script
    params_str = ""
    for i, params_i in enumerate(params):
        if isinstance(params_i, str):
            str_i = f"'{params_i}'"
        else:
            str_i = str(params_i)

        if i == 0 :
            params_str = params_str + str_i
        else:
            params_str = params_str + ' , ' + str_i

    #### params to print in script name
    params_str_name = ''
    for i, params_i in enumerate(params):

        str_i = str(params_i)

        if i == 0 :
            params_str_name = params_str_name + str_i
        else:
            params_str_name = params_str_name + '_' + str_i

    #### remove all txt that block name save
    for txt_remove_i in ["'", "[", "]", "{", "}", ":", " ", ","]:
        if txt_remove_i == " " or txt_remove_i == ",":
            params_str_name = params_str_name.replace(txt_remove_i, '_')
        else:
            params_str_name = params_str_name.replace(txt_remove_i, '')
    
    #### script text
    lines = [f'#! {python}']
    lines += ['import sys']
    lines += [f"sys.path.append('{path_main_workdir}')"]
    lines += [f'from {name_script} import {name_function}']
    lines += [f'{name_function}({params_str})']

    cpus_per_task = n_core_slurms
    mem = mem_crnl_cluster
        
    #### write script and execute
    os.chdir(path_slurm)
    slurm_script_name =  f"run__{name_function}__{params_str_name}.py" #add params
        
    with open(slurm_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()
    
    #### script text
    lines = ['#!/bin/bash']
    lines += [f'#SBATCH --job-name={name_function}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_crnl_cluster}']
    lines += [f'srun {python} {os.path.join(path_slurm, slurm_script_name)}']
        
    #### write script and execute
    slurm_bash_script_name =  f"bash__{name_function}__{params_str_name}.batch" #add params
        
    with open(slurm_bash_script_name, 'w') as f:
        f.writelines('\n'.join(lines))
        os.fchmod(f.fileno(), mode = stat.S_IRWXU)
        f.close()

    #### execute bash
    print(f'#### slurm submission : from {name_script} execute {name_function}({params})')
    subprocess.Popen(['sbatch', f'{slurm_bash_script_name}']) 

    # wait subprocess to lauch before removing
    #time.sleep(4)
    #os.remove(slurm_script_name)
    #os.remove(slurm_bash_script_name)

    #### get back to original path
    os.chdir(scritp_path)







############################
######## LOAD DATA ########
############################




def get_params(sujet):

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions()
    respi_ratio_allcond = get_all_respi_ratio(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    params = {'conditions' : conditions, 'chan_list' : chan_list, 'chan_list_ieeg' : chan_list_ieeg, 'srate' : srate, 
    'nwind' : nwind, 'nfft' : nfft, 'noverlap' : noverlap, 'hannw' : hannw, 'respi_ratio_allcond' : respi_ratio_allcond}

    return params






def extract_chanlist_srate_conditions():

    path_source = os.getcwd()
    
    #### select conditions to keep
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    conditions = conditions_allsubjects

    #### extract data
    band_prep = band_prep_list[0]
    cond = 'FR_CV'

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



def load_data(cond, band_prep=None):

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



def get_ac_starts(sujet):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))

    with open(f'{sujet}_AC_starts.txt') as f:
        ac_starts_txt = f.readlines()
        f.close()

    ac_starts = [int(i.replace('\n', '')) for i in ac_starts_txt]

    os.chdir(path_source)

    return ac_starts




def get_sniff_starts(sujet):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))

    with open(f'{sujet}_SNIFF_starts.txt') as f:
        ac_starts_txt = f.readlines()
        f.close()

    ac_starts = [int(i.replace('\n', '')) for i in ac_starts_txt]

    os.chdir(path_source)

    return ac_starts



########################################
######## LOAD RESPI FEATURES ########
########################################

def load_respfeatures(sujet):

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

    for cond in ['FR_CV']:

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





def get_all_respi_ratio(sujet):
    
    respfeatures_allcond = load_respfeatures(sujet)
    
    respi_ratio_allcond = {}

    for session_eeg in range(3):

        respi_ratio_allcond = {}

        for cond in ['FR_CV']:

            if len(respfeatures_allcond[cond]) == 1:

                mean_cycle_duration = np.mean(respfeatures_allcond[cond][0][['insp_duration', 'exp_duration']].values, axis=0)
                mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                respi_ratio_allcond[cond] = [ mean_inspi_ratio ]

            elif len(respfeatures_allcond[cond]) > 1:

                data_to_short = []

                for session_i in range(len(respfeatures_allcond[cond])):   
                    
                    if session_i == 0 :

                        mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                        mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
                        data_to_short = [ mean_inspi_ratio ]

                    elif session_i > 0 :

                        mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                        mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                        data_replace = [(data_to_short[0] + mean_inspi_ratio) / 2]

                        data_to_short = data_replace.copy()
                
                # to put in list
                respi_ratio_allcond[cond] = data_to_short 

    return respi_ratio_allcond


################################
######## STRETCH ########
################################


#resp_features, nb_point_by_cycle, data, srate = respfeatures_i, stretch_point_surrogates, x_shift, srate
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



def get_loca_df(sujet):

    path_source = os.getcwd()

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

    os.chdir(path_source)

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



