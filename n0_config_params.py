
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

#### data analysis type
electrode_recording_type = 'monopolaire'
# electrode_recording_type = 'bipolaire'

#### whole protocole
# sujet = 'pat_03083_1527'
# sujet = 'pat_03105_1551'
# sujet = 'pat_03128_1591'
# sujet = 'pat_03138_1601'
sujet = 'pat_03146_1608'
# sujet = 'pat_03174_1634'


#### FR_CV only
#sujet = 'pat_02459_0912'
#sujet = 'pat_02476_0929'
#sujet = 'pat_02495_0949'

#sujet = 'DEBUG'

AL_n = 3
srate = 500

#### whole protocole
sujet_list = ['pat_03083_1527', 'pat_03105_1551', 'pat_03128_1591', 'pat_03138_1601', 'pat_03146_1608', 'pat_03174_1634']

#### FR_CV
sujet_list_FR_CV = ['CHEe', 'GOBc', 'MAZm', 'TREt', 'MUGa', 'BANc', 'KOFs', 'LEMl', 'pat_02459_0912', 'pat_02476_0929', 'pat_02495_0949']

conditions = ['FR_CV', 'SNIFF', 'AL', 'AC']

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
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]},
                'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}}



########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_VPN'
    path_main_workdir = 'Z:\\multisite\\DATA_MANIP\\EEG_Paris_J\\Script_Python_iEEG_Paris_git'
    path_general = 'Z:\\multisite\\DATA_MANIP\\iEEG_Paris_J'
    path_memmap = 'Z:\\multisite\\DATA_MANIP\\iEEG_Paris_J\\Mmap'
    n_core = 4

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    path_main_workdir = 'C:\\Users\\jules\\Desktop\\Script_Python_iEEG_Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules' or PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J/Script_Python_iEEG_Paris_git/'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J/'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J/Mmap'
    n_core = 6

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Lyon_VJ/Script_Python_iEEG_Lyon/'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Lyon_VJ/'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Lyon_VJ/Mmap'
    n_core = 10

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J/Script_Python_iEEG_Lyon'
    path_general = '/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J/Script_Python_iEEG_Paris_git'
    path_general = '/crnldata/cmo/Projets/IntraEEG_Respi_Cardiaque/NBuonviso202001_ieeg_respi_nrec_jules/iEEG_Paris_J'
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
mem_crnl_cluster_offset = int(10e9)
n_core_slurms = 10




################################################
######## ELECTRODES REMOVED BEFORE LOCA ######## 
################################################

electrodes_to_remove = {

'pat_03083_1527' : [],
'pat_03105_1551' : [],
'pat_03128_1591' : [],
'pat_03138_1601' : [],
'pat_03146_1608' : [],
'pat_03174_1634' : []

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
'pat_03146_1608' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'},
'pat_03174_1634' : {'nasal': 'PRES1', 'ventral' : 'BELT1', 'ECG' : 'ECG1'},

'DEBUG' : {'nasal': 'p20+', 'ventral' : 'p19+', 'ECG' : 'ECG'}, # OK
}


prep_step_lf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': True},
}

prep_step_hf = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': True, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'average_reref' : {'execute': True},
}

prep_step_wb = {
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'average_reref' : {'execute': True},
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
'pat_03146_1608' : 'normal',
'pat_03174_1634' : 'normal'
}





################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'pat_03083_1527' : 'inverse',
'pat_03105_1551' : 'normal',
'pat_03128_1591' : 'normal',
'pat_03138_1601' : 'normal',
'pat_03146_1608' : 'normal',
'pat_03174_1634' : 'normal'
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']








########################################
######## PARAMS ERP ########
########################################

t_start_SNIFF = -3
t_stop_SNIFF = 3


t_start_AC = -12
t_stop_AC = 24


SNIFF_length = 3
AC_length = 12

SNIFF_lm_time = [-2, -0.5]






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
stretch_point_TF = 500
stretch_TF_auto = False
ratio_stretch_TF = 0.5
resampled_points_AL = 1000

#### TF & ITPC
nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
srate_dw = 10
wavetime = np.arange(-3,3,1/srate)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')
Pxx_wavelet_norm = 1000

stretch_point_TF_ac_resample = 1500
stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
stretch_point_TF_sniff_resampled = 1000

#### AL chunk
AL_chunk_pre_post_time = 12 #seconde

#### plot
tf_plot_percentile_scale = 1 #for one side
tf_plot_percentile_scale_AL = 5

#### STATS
n_surrogates_tf = 1000
norm_method = 'rscore'# 'zscore', 'dB'
tf_percentile_sel_stats_dw = 5 
tf_percentile_sel_stats_up = 95 
tf_stats_percentile_cluster = 95
tf_stats_percentile_cluster_allplot = 99
phase_stats =   {'AC' : ['pre', 're', 'post'],
                'SNIFF' : ['pre', 'post'],
                'AL' : ['pre', 'post']}


################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi






################################
######## FC ANALYSIS ########
################################

#### nfrex
nfrex_dfc = 50

#### ROI for DFC
ROI_for_DFC_df =    ['orbitofrontal', 'cingulaire ant rostral', 'cingulaire ant caudal', 'cingulaire post',
                    'insula ant', 'insula post', 'parahippocampique', 'amygdala', 'hippocampus']
ROI_for_DFC_plot =    ['orbitofrontal', 'cingulaire ant rostral', 'cingulaire ant caudal', 'cingulaire post',
                    'insula ant', 'insula post', 'parahippocampique', 'amygdala', 'hippocampus', 'temporal inf',
                    'temporal med', 'temporal sup']

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma']

#### cond definition
cond_FC_DFC = ['FR_CV', 'AL', 'SNIFF', 'AC']

#### down sample for AL
dw_srate_fc_AL = 10

#### down sample for AC
dw_srate_fc_AC = 50

#### n points for AL interpolation
n_points_AL_interpolation = 10000
n_points_AL_chunk = 1000

#### for df computation
percentile_graph_metric = 75





################################
######## DF EXTRACTION ########
################################

stretch_point_IE = [300, 500]
stretch_point_EI = [900, 100]
stretch_point_I = [0, int(stretch_point_TF/2)]
stretch_point_E = [int(stretch_point_TF/2), stretch_point_TF]

sniff_extract_pre = [-1, 0]
sniff_extract_resp_evnmt = [0, 1]
sniff_extract_post = [1, 2]

AC_extract_pre = [-AC_length, 0]
AC_extract_resp_evnmt_1 = [0, AC_length/2]
AC_extract_resp_evnmt_2 = [AC_length/2, AC_length]
AC_extract_post = [AC_length, AC_length*2]

n_points_AL_interpolation = 10500
n_phase_extraction_AL = 2
n_points_AL_chunk = 1000

srate_dw_stats = 100

AL_extract_time = 12




################################
######## HRV ANALYSIS ########
################################

srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)





