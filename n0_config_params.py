
import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

#### whole protocole
sujet = 'pat_03083_1527'
# sujet = 'pat_03105_1551'
# sujet = 'pat_03128_1591'
# sujet = 'pat_03138_1601'


#### FR_CV only
#sujet = 'pat_02459_0912'
#sujet = 'pat_02476_0929'
#sujet = 'pat_02495_0949'

#sujet = 'DEBUG'

#### whole protocole
sujet_list = ['pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601']

#### FR_CV
sujet_list_FR_CV = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'MUGa', 'BANc', 'KOFs', 'LEMl', 'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949']

conditions_allsubjects = ['FR_CV', 'SNIFF', 'AL', 'AC']

conditions_compute_TF = ['FR_CV', 'AC', 'SNIFF']

conditions_FC = ['FR_CV', 'AL']

band_prep_list = ['lf', 'hf']
freq_band_list = [{'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]}, {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}]
freq_band_whole = {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

freq_band_list_precompute = [{'theta_1' : [2,10], 'theta_2' : [4,8], 'alpha_1' : [8,12], 'alpha_2' : [8,14], 'beta_1' : [12,40], 'beta_2' : [10,40], 'whole_1' : [2,50]}, {'l_gamma_1' : [50, 80], 'h_gamma_1' : [80, 120]}]

freq_band_dict = {'wb' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40]},
                'lf' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_dict_FC = {'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40]},
                'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_dict_FC_function = {'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }



########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Home'
    path_main_workdir = 'C:\\Users\\jules\\Desktop\\Codage Informatique\\Script_Python_iEEG_Paris'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Paris'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Paris'
    n_core = 4

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    path_main_workdir = 'C:\\Users\\jules\\Desktop\\Script_Python_iEEG_Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/Script_Python_iEEG_Paris_git/'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/Mmap'
    n_core = 6

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon/'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/iEEG_Lyon_VJ/Mmap'
    n_core = 10

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/Script_Python_iEEG_Lyon'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/Script_Python_iEEG_Paris_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10

path_raw = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 
path_anatomy = os.path.join(path_general, 'Analyses', 'anatomy') 

path_slurm = os.path.join(path_general, 'Script_slurm')

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10

################################################
######## ELECTRODES REMOVED BEFORE LOCA ######## 
################################################

electrodes_to_remove = {

'pat_03083_1527' : [],
'pat_03105_1551' : [],
'pat_03128_1591' : [],
'pat_03138_1601' : [],

}



################################
######## PREP INFO ######## 
################################


conditions_trig = {
'FR_CV' : ['31', '32'], # FreeVentilation Comfort Ventilation
'SNIFF' : ['51', '52'], # RespiDriver Comfort Ventilation
'AC' : ['11', '12'], # RespiDriver Fast Ventilation  
'AL' : ['61', '62'], # RespiDriver Slow Ventilation
}


aux_chan = {
'pat_03083_1527' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1', 'EMG' : 'EMG1'}, # OK
'pat_03105_1551' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1', 'EMG' : 'EMG1'}, # OK
'pat_03128_1591' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1', 'EMG' : 'EMG1'}, # OK
'pat_03138_1601' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1', 'EMG' : 'EMG1'},

'DEBUG' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK
}


prep_step_lf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': False},
}

prep_step_hf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': True, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': False},
}





################################
######## RESPI PARAMS ########
################################ 

#### INSPI DOWN
sujet_respi_adjust = {
'pat_03083_1527' : 'normal',
'pat_03105_1551' : 'inverse',
'pat_03128_1591' : 'normal',
'pat_03138_1601' : 'normal',
}





################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'pat_03083_1527' : 'inverse',
'pat_03105_1551' : 'normal',
'pat_03128_1591' : 'normal',
'pat_03138_1601' : 'normal',
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']








########################################
######## PARAMS ERP ########
########################################

t_start_SNIFF = -3
t_stop_SNIFF = 3


t_start_AC = -5
t_stop_AC = 15





########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 2

def get_params_spectral_analysis(srate):
    nwind = int( 50*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
remove_zero_pad = 5

#### stretch
stretch_point_surrogates = 1000

#### coh
n_surrogates_coh = 1000
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 1000
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01


#### n bin for MI computation
MI_n_bin = 18



################################
######## PRECOMPUTE TF ########
################################

#### stretch
stretch_point_TF = 1000
stretch_TF_auto = False
ratio_stretch_TF = 0.40
resampled_points_AL = 10000

#### TF & ITPC
nfrex_hf = 50
nfrex_lf = 50
nfrex_wb = 50
ncycle_list_lf = [7, 15]
ncycle_list_hf = [20, 30]
ncycle_list_wb = [7, 30]
srate_dw = 10



################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi






################################
######## FC ANALYSIS ########
################################

#### ROI for DFC
ROI_for_DFC_df =    ['orbitofrontal', 'cingulaire ant rostral', 'cingulaire ant caudal', 'cingulaire post',
                    'insula ant', 'insula post', 'parahippocampique', 'amygdala', 'hippocampus']

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['beta', 'l_gamma', 'h_gamma']

#### cond definition
cond_FC_DFC = ['FR_CV', 'AL', 'SNIFF', 'AC']

#### down sample for AL
dw_srate_fc_AL = 10

#### n points for AL interpolation
n_points_AL_interpolation = 10000




################################
######## DF EXTRACTION ########
################################

stretch_point_IE = [300, 500]
stretch_point_EI = [900, 100]
stretch_point_I = [100, 300]
stretch_point_E = [600, 800]

sniff_extract_pre = [-1, 0]
sniff_extract_post = [0, 1]

AC_extract_pre = [-5, 0]
AC_extract_post = [0, 10]

AL_coeff_pre = .10 


################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)





