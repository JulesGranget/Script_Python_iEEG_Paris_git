

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
import frites

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
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('Baselines', construct_token)
    construct_token = create_folder('FC', construct_token)

        #### anatomy
    os.chdir(os.path.join(path_general, 'Analyses', 'anatomy'))
    construct_token = create_folder(sujet, construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results'))
    construct_token = create_folder(sujet, construct_token)
    construct_token = create_folder('allplot', construct_token)

            #### allplot
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot'))
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('anatomy', construct_token)
    construct_token = create_folder('FC', construct_token)

                #### FC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FC'))
    construct_token = create_folder('DFC', construct_token)

                    #### DFC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'FC', 'DFC'))
    construct_token = create_folder('SNIFF', construct_token)
    construct_token = create_folder('AC', construct_token)
    construct_token = create_folder('AL', construct_token)

                #### TF
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'TF'))
    construct_token = create_folder('Lobes', construct_token)
    construct_token = create_folder('ROI', construct_token)
                
                ####ITPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', 'allplot', 'ITPC'))
    construct_token = create_folder('Lobes', construct_token)
    construct_token = create_folder('ROI', construct_token)

        #### results
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet))
    construct_token = create_folder('RESPI', construct_token)
    construct_token = create_folder('TF', construct_token)
    construct_token = create_folder('PSD_Coh', construct_token)
    construct_token = create_folder('ITPC', construct_token)
    construct_token = create_folder('FC', construct_token)
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
    construct_token = create_folder('DFC', construct_token)

                #### PLI
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC', 'PLI'))
    construct_token = create_folder('figures', construct_token)
    construct_token = create_folder('matrix', construct_token)

                #### ISPC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC', 'ISPC'))
    construct_token = create_folder('figures', construct_token)
    construct_token = create_folder('matrix', construct_token)

                #### DFC
    os.chdir(os.path.join(path_general, 'Analyses', 'results', sujet, 'FC', 'DFC'))
    construct_token = create_folder('SNIFF', construct_token)
    construct_token = create_folder('AC', construct_token)
    construct_token = create_folder('AL', construct_token)

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
    lines += [f'#SBATCH --job-name={name_function}']
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




def get_params(sujet):

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    respi_ratio_allcond = get_all_respi_ratio(sujet)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    params = {'conditions' : conditions, 'chan_list' : chan_list, 'chan_list_ieeg' : chan_list_ieeg, 'srate' : srate, 
    'nwind' : nwind, 'nfft' : nfft, 'noverlap' : noverlap, 'hannw' : hannw, 'respi_ratio_allcond' : respi_ratio_allcond}

    return params






def extract_chanlist_srate_conditions(sujet):

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

#band_prep, cond, session_i = 'lf', cond, 0
def load_data_sujet(sujet_tmp, band_prep, cond, session_i):

    path_source = os.getcwd()
    
    os.chdir(os.path.join(path_prep, sujet_tmp, 'sections'))

    if cond == 'SNIFF':
        cond_search = 'SNIFF_session'
    else:
        cond_search = cond

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if ( session_name.find(cond_search) != -1 ) & ( session_name.find(band_prep) != -1 ):
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




################################
######## WAVELETS ########
################################


def get_wavelets(band_prep, freq):

    #### get params
    prms = get_params(sujet)

    #### select wavelet parameters
    if band_prep == 'wb':
        wavetime = np.arange(-2,2,1/prms['srate'])
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex) 

    if band_prep == 'lf':
        wavetime = np.arange(-2,2,1/prms['srate'])
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/prms['srate'])
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

    #### compute wavelets
    frex  = np.linspace(freq[0],freq[1],nfrex)
    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    # create Morlet wavelet family
    for fi in range(0,nfrex):
        
        s = ncycle_list[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    return wavelets, nfrex


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








################################
######## PLI ISPC DFC ######## 
################################

#cond, band_prep, band, freq = 'SNIFF', 'hf', 'l_gamma', [50, 80]
def get_pli_ispc_dfc(sujet, cond, band_prep, band, freq):

        data = load_data(cond, band_prep='hf')

        if cond == 'SNIFF':
            epochs_starts = get_sniff_starts(sujet)

        if cond == 'AC':
            epochs_starts = get_ac_starts(sujet)
        
        prms = get_params(sujet)

        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_DFC_pli_ispc_{band}_{cond}.nc')):
            print('ALREADY DONE')
            return

        wavelets, nfrex = get_wavelets(band_prep, freq)

        os.chdir(path_memmap)
        convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

        print('CONV')

        #nchan = 0
        def convolution_x_wavelets_nchan(nchan_i, nchan):

            # print_advancement(nchan_i, len(prms['chan_list_ieeg']), steps=[25, 50, 75])
            
            nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

            x = data[nchan_i,:]

            for fi in range(nfrex):

                nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

            convolutions[nchan_i,:,:] = nchan_conv

            return

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(prms['chan_list_ieeg']))

        #### free memory
        del data

        #### epoch convolutions
        if cond == 'SNIFF':
            t_start_epoch, t_stop_epoch = t_start_SNIFF, t_stop_SNIFF
        if cond == 'AC':
            t_start_epoch, t_stop_epoch = t_start_AC, t_stop_AC

        #### generate matrix epoch
        os.chdir(path_memmap)
        stretch_point_TF_epoch = int(np.abs(t_start_epoch)*prms['srate'] +  t_stop_epoch*prms['srate'])
        epochs = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_fc_epochs.dat', dtype=np.complex128, mode='w+', shape=( len(prms['chan_list_ieeg']), len(epochs_starts), nfrex, stretch_point_TF_epoch ))

        def chunk_epochs_in_signal(nchan_i, nchan):
            
            for epoch_i, epoch_time in enumerate(epochs_starts):

                _t_start = epoch_time + int(t_start_epoch*prms['srate']) 
                _t_stop = epoch_time + int(t_stop_epoch*prms['srate'])

                epochs[nchan_i, epoch_i, :, :] = convolutions[nchan_i, :, _t_start:_t_stop]

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(chunk_epochs_in_signal)(nchan_i, nchan) for nchan_i, nchan in enumerate(prms['chan_list_ieeg']))
        
        #### identify roi in data
        df_loca = get_loca_df(sujet)
        df_sorted = df_loca.sort_values(['lobes', 'ROI'])
        index_sorted = df_sorted.index.values
        chan_name_sorted = df_sorted['ROI'].values.tolist()

        roi_in_data = []
        rep_count = 0
        for i, name_i in enumerate(chan_name_sorted):
            if i == 0:
                roi_in_data.append(name_i)
                continue
            else:
                if name_i == chan_name_sorted[i-(rep_count+1)]:
                    rep_count += 1
                    continue
                if name_i != chan_name_sorted[i-(rep_count+1)]:
                    roi_in_data.append(name_i)
                    rep_count = 0
                    continue

        #### compute index
        pairs_possible = []
        for pair_A_i, pair_A in enumerate(roi_in_data):
            for pair_B_i, pair_B in enumerate(roi_in_data[pair_A_i:]):
                if pair_A == pair_B:
                    continue
                pairs_possible.append(f'{pair_A}-{pair_B}')

        pairs_to_compute = []
        for pair_A_i, pair_A in enumerate(prms['chan_list_ieeg']):
            for pair_B_i, pair_B in enumerate(prms['chan_list_ieeg']):
                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue
                pairs_to_compute.append(f'{pair_A}-{pair_B}')

        #### identify slwin
        slwin_len = slwin_dict[band]    # in sec
        slwin_step = slwin_len*slwin_step_coeff  # in sec
        times_epoch = np.arange(t_start_epoch, t_stop_epoch, 1/prms['srate'])
        win_sample = frites.conn.define_windows(times_epoch, slwin_len=slwin_len, slwin_step=slwin_step)[0]
        times = np.linspace(t_start_epoch, t_stop_epoch, len(win_sample))

        print('COMPUTE')   

        #pair_to_compute = pairs_to_compute[0]
        def compute_ispc_pli(pair_to_compute_i, pair_to_compute):

            # print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

            pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
            pair_A_i, pair_B_i = prms['chan_list_ieeg'].index(pair_A), prms['chan_list_ieeg'].index(pair_B)

            ispc_dfc_i = np.zeros(( len(win_sample) ))
            pli_dfc_i = np.zeros(( len(win_sample) ))

            #slwin_values_i, slwin_values = 0, win_sample[0]
            for slwin_values_i, slwin_values in enumerate(win_sample):
                    
                as1 = epochs[pair_A_i, :, :, slwin_values[0]:slwin_values[-1]]
                as2 = epochs[pair_B_i, :, :, slwin_values[0]:slwin_values[-1]]

                # collect "eulerized" phase angle differences
                cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                
                # compute ISPC and PLI (and average over trials!)
                ispc_dfc_i[slwin_values_i] = np.abs(np.mean(cdd))
                pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))

            return ispc_dfc_i, pli_dfc_i

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))
        
        #### compute metrics
        pli_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))
        ispc_mat = np.zeros((len(pairs_to_compute),np.size(win_sample,0)))

        #### load in mat    
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
                    
            ispc_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][0]
            pli_mat[pair_to_compute_i,:] = compute_ispc_pli_res[pair_to_compute_i][1]

        #### free memory
        del compute_ispc_pli_res

        #### remove conv
        os.chdir(path_memmap)
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_fc_convolutions.dat')
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_fc_epochs.dat')
        
        #### generate mat results
        mat_pli_time = np.zeros(( len(pairs_possible), pli_mat.shape[-1] ))
        mat_ispc_time = np.zeros(( len(pairs_possible), ispc_mat.shape[-1] ))

        #### fill mat
        name_modified = np.array([])
        count_pairs = np.zeros(( len(pairs_possible) ))
        for pair_i in pairs_to_compute:
            pair_A, pair_B = pair_i.split('-')
            pair_A_name, pair_B_name = df_loca['ROI'][df_loca['name'] == pair_A].values[0], df_loca['ROI'][df_loca['name'] == pair_B].values[0]
            pair_name_i = f'{pair_A_name}-{pair_B_name}'
            name_modified = np.append(name_modified, pair_name_i)
        
        for pair_name_i, pair_name in enumerate(pairs_possible):
            pair_name_inv = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
            mask = (name_modified == pair_name) | (name_modified == pair_name_inv)
            count_pairs[pair_name_i] = int(np.sum(mask))
            mat_pli_time[pair_name_i,:] = np.mean(pli_mat[mask,:], axis=0)
            mat_ispc_time[pair_name_i,:] = np.mean(ispc_mat[mask,:], axis=0)

        #### save
        os.chdir(os.path.join(path_precompute, sujet, 'FC'))
        dict_xr_pli = {'mat_type' : ['pli', 'ispc'], 'pairs' : pairs_possible, 'times' : times}
        data_export = np.concatenate( [mat_pli_time.reshape(1, mat_pli_time.shape[0], mat_pli_time.shape[1]), 
                                        mat_ispc_time.reshape(1, mat_ispc_time.shape[0], mat_ispc_time.shape[1])], axis=0 )
        xr_export = xr.DataArray(data_export, coords=dict_xr_pli.values(), dims=dict_xr_pli.keys())
        xr_export.to_netcdf(f'{sujet}_DFC_pli_ispc_{band}_{cond}.nc')



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
            print(f'{step}%')







