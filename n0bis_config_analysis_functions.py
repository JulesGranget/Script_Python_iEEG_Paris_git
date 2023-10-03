

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
import joblib

from n0_config_params import *


debug = False




########################
######## DEBUG ########
########################


def debug_memory():

    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                            key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))












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
    construct_token = create_folder('allplot', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'precompute', sujet))
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('baselines', construct_token)
    construct_token = create_folder('DFC', construct_token)
    construct_token = create_folder('ERP', construct_token)

        #### anatomy
    os.chdir(os.path.join(path_general, 'Analyses', 'anatomy'))
    construct_token = create_folder(sujet, construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allplot', construct_token)

            #### allplot
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot'))
    construct_token = create_folder('df', construct_token)
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('anatomy', construct_token)

                #### allcond
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond'))
    construct_token = create_folder('DFC', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('TF', construct_token)

                    #### TF/ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'TF'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'ITPC'))
    construct_token = create_folder('ROI', construct_token)
    construct_token = create_folder('Lobes', construct_token)

                    #### DFC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'allcond', 'DFC'))
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('summary', construct_token)

            #### sujet
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet))
    construct_token = create_folder('RESPI', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('DFC', construct_token)
    construct_token = create_folder('df', construct_token)
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

                #### ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'ITPC'))
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('allcond', construct_token)

                #### DFC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'DFC'))
    construct_token = create_folder('allcond', construct_token)
    construct_token = create_folder('summary', construct_token)
    construct_token = create_folder('verif', construct_token)

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
    lines += [f'#SBATCH --job-name={name_function}_{params_str_name}']
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





#name_script, name_function, params = 'n9_fc_analysis', 'compute_pli_ispc_allband', [sujet]
def execute_function_in_slurm_bash_mem_choice(name_script, name_function, params, mem_required):

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
    lines += [f'#SBATCH --job-name={name_function}_{params_str_name}']
    lines += [f'#SBATCH --output=%slurm_{name_function}_{params_str_name}.log']
    lines += [f'#SBATCH --cpus-per-task={n_core_slurms}']
    lines += [f'#SBATCH --mem={mem_required}']
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




def get_params(sujet, electrode_recording_type):

    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)
    respi_ratio_allcond = get_all_respi_ratio(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    params = {'conditions' : conditions, 'chan_list' : chan_list, 'chan_list_ieeg' : chan_list_ieeg, 'srate' : srate, 
    'nwind' : nwind, 'nfft' : nfft, 'noverlap' : noverlap, 'hannw' : hannw, 'respi_ratio_allcond' : respi_ratio_allcond}

    return params






def get_chanlist(sujet, electrode_recording_type):

    path_source = os.getcwd()
    
    #### extract data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    if electrode_recording_type == 'monopolaire':
        file_to_load = f'{sujet}_FR_CV.fif'
    if electrode_recording_type == 'bipolaire':
        file_to_load = f'{sujet}_FR_CV_bi.fif'

    raw = mne.io.read_raw_fif(file_to_load, preload=True, verbose='critical')

    chan_list = raw.info['ch_names']
    chan_list_ieeg = chan_list[:-4] # we remove : nasal, ventral, ECG, ECG_cR

    #### go back to path source
    os.chdir(path_source)

    return chan_list, chan_list_ieeg





def load_data(sujet, cond, electrode_recording_type):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    if cond == 'FR_CV' :

        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if electrode_recording_type == 'bipolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') != -1 ):
                    load_i.append(i)
            elif electrode_recording_type == 'monopolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') == -1 ):
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
            if electrode_recording_type == 'bipolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') != -1 ):
                    load_i.append(i)
            elif electrode_recording_type == 'monopolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') == -1 ):
                    load_i.append(i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i if os.listdir()[i].find('.nc') == -1]

        raw = mne.io.read_raw_fif(load_list[0], preload=True, verbose='critical')

        data = raw.get_data()
        
        #data = xr.open_dataset(load_list[0])


    elif cond == 'AL' :
    
        load_i = []
        for i, session_name in enumerate(os.listdir()):
            if electrode_recording_type == 'bipolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') != -1 ):
                    load_i.append(i)
            elif electrode_recording_type == 'monopolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') == -1 ):
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
            if electrode_recording_type == 'bipolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') != -1 ):
                    load_i.append(i)
            elif electrode_recording_type == 'monopolaire':
                if ( session_name.find(cond) != -1 ) & ( session_name.find('bi') == -1 ):
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

    os.chdir(os.path.join(path_precompute, sujet, 'ERP'))

    ac_starts = np.load(f'{sujet}_AC_select.npy')

    os.chdir(path_source)

    return ac_starts




def get_sniff_starts(sujet):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_precompute, sujet, 'ERP'))

    sniff_starts = np.load(f'{sujet}_SNIFF_select.npy')

    os.chdir(path_source)

    return sniff_starts




def get_ac_starts_uncleaned(sujet):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))

    with open(f'{sujet}_AC_starts.txt') as f:
        ac_starts_txt = f.readlines()
        f.close()

    ac_starts = [int(i.replace('\n', '')) for i in ac_starts_txt]

    os.chdir(path_source)

    return ac_starts




def get_sniff_starts_uncleaned(sujet):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_prep, sujet, 'info'))

    with open(f'{sujet}_SNIFF_starts.txt') as f:
        ac_starts_txt = f.readlines()
        f.close()

    ac_starts = [int(i.replace('\n', '')) for i in ac_starts_txt]

    os.chdir(path_source)

    return ac_starts






################################
######## WAVELETS ########
################################


def get_wavelets():

    #### compute wavelets
    wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

    # create Morlet wavelet family
    for fi in range(nfrex):
        
        s = cycles[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    if debug:

        plt.plot(np.sum(np.abs(wavelets),axis=1))
        plt.show()

        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets)[0,:])
        plt.show()

    return wavelets



def get_wavelets_dfc(freq):

    #### compute wavelets
    wavelets = np.zeros((nfrex_dfc, len(wavetime)), dtype=complex)
    frex_dfc = np.linspace(freq[0], freq[-1], nfrex_dfc)

    # create Morlet wavelet family
    for fi in range(nfrex_dfc):
        
        s = cycles[fi] / (2*np.pi*frex_dfc[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex_dfc[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    if debug:

        plt.plot(np.sum(np.abs(wavelets),axis=1))
        plt.show()

        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets)[0,:])
        plt.show()

    return wavelets











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
                data_to_short_count = 0

                for session_i in range(len(respfeatures_allcond[cond])):   
                    
                    if session_i == 0 :

                        mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                        mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
                        data_to_short = [ mean_inspi_ratio ]

                    elif session_i > 0 :

                        mean_cycle_duration = np.mean(respfeatures_allcond[cond][session_i][['insp_duration', 'exp_duration']].values, axis=0)
                        mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()

                        data_replace = [(data_to_short[0] + mean_inspi_ratio)]
                        data_to_short_count += 1

                        data_to_short = data_replace.copy()
                
                # to put in list
                respi_ratio_allcond[cond] = data_to_short / data_to_short_count

    return respi_ratio_allcond











########################################
######## MI ANALYSIS FUNCTIONS ########
########################################



def shuffle_CycleFreq(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift
    

def shuffle_Cxy(x):
   half_size = x.shape[0]//2
   ind = np.random.randint(low=0, high=half_size)
   x_shift = x.copy()
   
   x_shift[ind:ind+half_size] *= -1
   if np.random.rand() >=0.5:
       x_shift *= -1

   return x_shift


def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(np.where(a != 0, a * np.log(a), 0))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def Shannon_MI(a):
    a = np.asarray(a, dtype = float)
    N = a.size
    kl_divergence_shannon = np.log(N) - Shannon_Entropy(a)
    return kl_divergence_shannon / np.log(N)



def get_MVL(x):
    _phase = np.arange(0, x.shape[0])*2*np.pi/x.shape[0]
    complex_vec = x*np.exp(1j*_phase)

    MVL = np.abs(np.mean(complex_vec))
    
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.scatter(complex_vec.real, complex_vec.imag)
        ax.scatter(np.mean(complex_vec.real), np.mean(complex_vec.imag), linewidth=3, color='r')
        plt.show()

    return MVL










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


#resp_features, nb_point_by_cycle, data, srate = respfeatures_allcond[cond][0], stretch_point_TF, tf[0,:,:], srate
def stretch_data_tf(resp_features, nb_point_by_cycle, data, srate):

    # params
    cycle_times = resp_features[['inspi_time', 'expi_time']].values
    mean_cycle_duration = np.mean(resp_features[['insp_duration', 'exp_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,data.shape[1])/srate

    # stretch
    if stretch_TF_auto:
        clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                data.T, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=mean_inspi_ratio)
    else:
        clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                data.T, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=ratio_stretch_TF)
                
    #### clean
    mask = resp_features[resp_features['select'] == 1].index.values
    cycle_clean = mask[np.isin(mask, cycles)]

    #### reshape
    if np.iscomplex(data[0,0]):
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ), dtype='complex')
    else:
        data_stretch = np.zeros(( cycle_clean.shape[0], data.shape[0], nb_point_by_cycle ), dtype=data.dtype)

    for cycle_i, cycle_val in enumerate(cycle_clean):
        data_stretch[cycle_i, :, :] = data_stretch_linear.T[:, nb_point_by_cycle*(cycle_val):nb_point_by_cycle*(cycle_val+1)]

    # inspect
    if debug == True:
        plt.pcolormesh(data_stretch_linear.T)
        plt.show()

        plt.pcolormesh(np.mean(data_stretch, axis=0))
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



def get_loca_df(sujet, electrode_recording_type):

    path_source = os.getcwd()

    os.chdir(os.path.join(path_anatomy, sujet))

    if electrode_recording_type == 'monopolaire':
        file_plot_select = pd.read_excel(sujet + '_plot_loca.xlsx')
    if electrode_recording_type == 'bipolaire':
        file_plot_select = pd.read_excel(sujet + '_plot_loca_bi.xlsx')

    chan_list_ieeg_trc = file_plot_select['plot'][file_plot_select['select'] == 1].values.tolist()

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








########################################
######## SCRIPT ADVANCEMENT ########
########################################


def print_advancement(i, i_final, steps=[25, 50, 75]):

    steps_i = {}
    for step in steps:

        step_i = 0
        while (step_i/i_final*100) < step:
            step_i += 1

        steps_i[step] = step_i

    for step, step_i in steps_i.items():

        if i == step_i:
            print(f'{step}%', flush=True)






################################
######## NORMALIZATION ########
################################


def zscore(x):

    x_zscore = (x - x.mean()) / x.std()

    return x_zscore




def zscore_mat(x):

    _zscore_mat = (x - x.mean(axis=1).reshape(-1,1)) / x.std(axis=1).reshape(-1,1)

    return _zscore_mat



def rscore(x):

    mad = np.median( np.abs(x-np.median(x)) ) # median_absolute_deviation

    rzscore_x = (x-np.median(x)) * 0.6745 / mad

    return rzscore_x
    



def rscore_mat(x):

    mad = np.median(np.abs(x-np.median(x, axis=1).reshape(-1,1)), axis=1) # median_absolute_deviation

    _rscore_mat = (x-np.median(x, axis=1).reshape(-1,1)) * 0.6745 / mad.reshape(-1,1)

    return _rscore_mat





#tf_conv = tf
def norm_tf(sujet, tf_conv, electrode_recording_type, norm_method):

    path_source = os.getcwd()

    chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

    if norm_method not in ['rscore', 'zscore']:

        #### load baseline
        os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

        if electrode_recording_type == 'monopolaire':
            baselines = xr.open_dataarray(f'{sujet}_baselines.nc')
        if electrode_recording_type == 'bipolaire':
            baselines = xr.open_dataarray(f'{sujet}_baselines_bi.nc')

        if debug:

            plt.plot(baselines.values[0,:,0], label='mean')
            plt.plot(baselines.values[0,:,2], label='median')
            plt.legend()
            plt.show()

            plt.plot(baselines.values[0,:,1], label='std')
            plt.plot(baselines.values[0,:,3], label='mad')
            plt.legend()
            plt.show()

    if norm_method == 'dB':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = 10*np.log10(tf_conv[n_chan_i,:,:] / baselines.loc[n_chan, :, 'median'].values.reshape(-1,1))

    if norm_method == 'zscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'mean'].values.reshape(-1,1)) / baselines.loc[n_chan,:,'std'].values.reshape(-1,1)
                
    if norm_method == 'rscore_baseline':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = (tf_conv[n_chan_i,:,:] - baselines.loc[n_chan,:,'median'].values.reshape(-1,1)) * 0.6745 / baselines.loc[n_chan,:,'mad'].values.reshape(-1,1)

    if norm_method == 'zscore':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = zscore_mat(tf_conv[n_chan_i,:,:])
                
    if norm_method == 'rscore':

        for n_chan_i, n_chan in enumerate(chan_list_ieeg):

            tf_conv[n_chan_i,:,:] = rscore_mat(tf_conv[n_chan_i,:,:])


    #### verify baseline
    if debug:

        plt.pcolormesh(tf_conv[0,:,:], vmin=np.percentile(tf_conv[0,:,:], 1), vmax=np.percentile(tf_conv[0,:,:], 99))
        plt.show()

        nchan = 0
        nchan_name = chan_list_ieeg[nchan]

        data_plot = tf_conv[nchan,:,:]
        plt.pcolormesh(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99))
        plt.show()

        fig, axs = plt.subplots(ncols=2)
        axs[0].set_title('mean std')
        axs[0].plot(baselines.loc[nchan_name,:,'mean'], label='mean')
        axs[0].plot(baselines.loc[nchan_name,:,'std'], label='std')
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[1].set_title('median mad')
        axs[1].plot(baselines.loc[nchan_name,:,'median'], label='median')
        axs[1].plot(baselines.loc[nchan_name,:,'mad'], label='mad')
        axs[1].legend()
        axs[1].set_yscale('log')
        plt.show()

        tf_test = tf_conv[nchan,:,:int(tf_conv.shape[-1]/10)].copy()

        fig, axs = plt.subplots(nrows=6)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        percentile_sel = 0

        vmin = np.percentile(tf_test.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_test.reshape(-1),100-percentile_sel)
        im = axs[0].pcolormesh(tf_test, vmin=vmin, vmax=vmax)
        axs[0].set_title('raw')
        fig.colorbar(im, ax=axs[0])

        tf_baseline = 10*np.log10(tf_test / baselines.loc[chan_list_ieeg[nchan], :, 'median'].values.reshape(-1,1))
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[1].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[1].set_title('db')
        fig.colorbar(im, ax=axs[1])

        tf_baseline = (tf_test - baselines.loc[chan_list_ieeg[nchan],:,'mean'].values.reshape(-1,1)) / baselines.loc[chan_list_ieeg[nchan],:,'std'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[2].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[2].set_title('zscore')
        fig.colorbar(im, ax=axs[2])

        tf_baseline = (tf_test - baselines.loc[chan_list_ieeg[nchan],:,'median'].values.reshape(-1,1)) / baselines.loc[chan_list_ieeg[nchan],:,'mad'].values.reshape(-1,1)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[3].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[3].set_title('rscore')
        fig.colorbar(im, ax=axs[3])

        tf_baseline = zscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[4].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[4].set_title('zscore_mat')
        fig.colorbar(im, ax=axs[4])

        tf_baseline = rscore_mat(tf_test)
        vmin = np.percentile(tf_baseline.reshape(-1),percentile_sel)
        vmax = np.percentile(tf_baseline.reshape(-1),100-percentile_sel)
        im = axs[5].pcolormesh(tf_baseline, vmin=vmin, vmax=vmax)
        axs[5].set_title('rscore_mat')
        fig.colorbar(im, ax=axs[5])

        plt.show()

    os.chdir(path_source)

    return tf_conv






def get_mad(data, axis=0):

    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data - med), axis=axis)

    return mad