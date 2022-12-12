
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







def organize_raw(sujet, raw):

    #### extract chan_list
    chan_list_clean = []
    chan_list = raw.info['ch_names']
    srate = int(raw.info['sfreq'])
    [chan_list_clean.append(nchan[23:]) for nchan in chan_list]

    #### extract data
    data = raw.get_data()

    #### identify aux chan
    nasal_i = chan_list_clean.index(aux_chan[sujet]['nasal'])
    ventral_i = chan_list_clean.index(aux_chan[sujet]['ventral'])
    ecg_i = chan_list_clean.index(aux_chan[sujet]['ECG'])

    data_aux = np.vstack((data[nasal_i,:], data[ventral_i,:], data[ecg_i,:]))

    if debug:
        plt.plot(data_aux[0,:])
        plt.plot(data_aux[1,:])
        plt.plot(data_aux[2,:])
        plt.show()

    #### remove from data
    data_ieeg = data.copy()

    # remove other aux
    for aux_name in aux_chan[sujet].keys():

        aux_i = chan_list_clean.index(aux_chan[sujet][aux_name])
        data_ieeg = np.delete(data_ieeg, aux_i, axis=0)
        chan_list_clean.remove(aux_chan[sujet][aux_name])

    chan_list_aux = [aux_i for aux_i in list(aux_chan[sujet]) if aux_i != 'EMG']
    chan_list_ieeg = chan_list_clean


    return data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate




#trc_filename = 'LYONNEURO_2021_GOBc_RESPI.TRC'
def extract_chanlist(sujet):

    print('#### EXTRACT ####')

    os.chdir(os.path.join(path_raw, sujet, 'raw_data', 'mat'))

    if sujet == 'pat_03146_1608':

        raw_1 = mne.io.read_raw_eeglab(f'{sujet}_1_allchan.set', preload=True)
        raw_2 = mne.io.read_raw_eeglab(f'{sujet}_2_allchan.set', preload=True)

        raw_2.info['ch_names'] = raw_1.info['ch_names']

        raw = mne.concatenate_raws([raw_1, raw_2], preload=True)

        del raw_1, raw_2

    else:

        raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)
    
    data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(sujet, raw)

    return chan_list_ieeg



def generate_plot_loca(chan_list_ieeg):


    #### open loca file
    os.chdir(os.path.join(path_raw, sujet, 'anatomy', 'anat_adjusted'))

    parcellisation = pd.read_excel(f'{sujet}_anatomical_localizations.xlsx')

    #### open nomenclature
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    #### identify if parcellisation miss plots 
    miss_in_parcellisation = []
    chan_list_ieeg_in_parcel = []
    for plot in chan_list_ieeg:
        if plot in parcellisation['Plot'].values:
            chan_list_ieeg_in_parcel.append(plot)
        else:
            miss_in_parcellisation.append(plot)

    print('miss in parcellisation : ', miss_in_parcellisation)

    #### export missed plots
    os.chdir(os.path.join(path_anatomy, sujet))
    miss_plot_textfile = open(sujet + "_miss_in_parcelisation.txt", "w")
    for element in miss_in_parcellisation:
        miss_plot_textfile.write(element + "\n")
    miss_plot_textfile.close()

    keep_plot_textfile = open(sujet + "_plot_in_parcelisation.txt", "w")
    for element in chan_list_ieeg_in_parcel:
        keep_plot_textfile.write(element + "\n")
    keep_plot_textfile.close()

    #### categories to fill
    columns = ['subject', 'plot', 'MNI', 'freesurfer_destrieux', 'correspondance_ROI', 'correspondance_lobes', 'comparison', 'abscent',	
                'noisy_signal', 'inSOZ', 'not_in_atlas', 'select', 'localisation_corrected', 'lobes_corrected']

    #### find correspondances
    freesurfer_destrieux = [ parcellisation['FreesurferDesikan'][parcellisation['Plot'] == nchan].values.tolist()[0] for nchan in chan_list_ieeg_in_parcel]

    correspondance_ROI = []
    correspondance_lobes = []
    for parcel_i in freesurfer_destrieux:
        if parcel_i[0] == 'c':
            parcel_i_chunk = parcel_i[7:]
        elif parcel_i[0] == 'L':
            parcel_i_chunk = parcel_i[5:]
        elif parcel_i[0] == 'R':
            parcel_i_chunk = parcel_i[6:]
        elif parcel_i[0] == 'W':
            parcel_i_chunk = 'Cerebral-White-Matter'
        else:
            parcel_i_chunk = parcel_i

        if parcel_i_chunk == 'unknown':
            parcel_i_chunk = 'Unknown'
        
        correspondance_ROI.append(nomenclature['Our correspondances'][nomenclature['Labels'] == parcel_i_chunk].values[0])
        correspondance_lobes.append(nomenclature['Lobes'][nomenclature['Labels'] == parcel_i_chunk].values[0])

    #### generate MNI info
    MNI_loca = []
    for nchan in chan_list_ieeg_in_parcel:
        MNI_i = [parcellisation['MNI_x'][parcellisation['Plot'] == nchan].values[0], parcellisation['MNI_y'][parcellisation['Plot'] == nchan].values[0], parcellisation['MNI_z'][parcellisation['Plot'] == nchan].values[0]] 
        MNI_loca.append(MNI_i)

    #### generate df
    electrode_select_dict = {}
    for ncol in columns:
        if ncol == 'subject':
            electrode_select_dict[ncol] = [sujet] * len(chan_list_ieeg_in_parcel)
        elif ncol == 'plot':
            electrode_select_dict[ncol] = chan_list_ieeg_in_parcel
        elif ncol == 'MNI':
            electrode_select_dict[ncol] = MNI_loca
        elif ncol == 'freesurfer_destrieux':
            electrode_select_dict[ncol] = freesurfer_destrieux
        elif ncol == 'localisation_corrected' or ncol == 'lobes_corrected':
            electrode_select_dict[ncol] = [0] * len(chan_list_ieeg_in_parcel)
        elif ncol == 'correspondance_ROI':
            electrode_select_dict[ncol] = correspondance_ROI
        elif ncol == 'correspondance_lobes':
            electrode_select_dict[ncol] = correspondance_lobes
        else :
            electrode_select_dict[ncol] = [0] * len(chan_list_ieeg_in_parcel)


    #### generate df and save
    electrode_select_df = pd.DataFrame(electrode_select_dict, columns=columns)

    os.chdir(os.path.join(path_anatomy, sujet))
    
    electrode_select_df.to_excel(sujet + '_plot_loca.xlsx')

    return
        







################################
######## BIPOLARIZATION ########
################################


def bipolarize_anatomy_name(plot_name_sel):

    #### bipolarize anatomy localization
    correspondance_bipol = []

    for plot_i, plot_name in enumerate(plot_name_sel):

        if plot_i == len(plot_name_sel)-1:

            continue

        correspondance_bipol.append(f'{plot_name_sel[plot_i]}-{plot_name_sel[plot_i+1]}')

    return correspondance_bipol






def bipolarize_anatomy_localization(anat_selection):

    #### bipolarize anatomy localization
    correspondance_ROI_bipol = []
    #plot_i, plot_name = 0, correspondance_ROI[0]
    for plot_i, plot_name in enumerate(anat_selection):

        if plot_i == len(anat_selection)-1:
            
            continue

        if anat_selection[plot_i+1] == plot_name:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] not in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] in ['WM', 'unknown', 'Unknown', 'ventricule']:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] in ['WM', 'unknown', 'Unknown', 'ventricule']:
            
            correspondance_ROI_bipol.append(plot_name)
            continue

        if anat_selection[plot_i] in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] not in ['WM', 'unknown', 'Unknown', 'ventricule']:

            correspondance_ROI_bipol.append(anat_selection[plot_i+1])
            continue

        if anat_selection[plot_i] not in ['WM', 'unknown', 'Unknown', 'ventricule'] and anat_selection[plot_i+1] not in ['WM', 'unknown', 'Unknown', 'ventricule']:

            correspondance_ROI_bipol.append(anat_selection[plot_i])
            continue

    return correspondance_ROI_bipol

        




#chan_list_ieeg = chan_list_ncs
def generate_plot_loca_bipolaire(chan_list_ieeg):

    #### open loca file
    os.chdir(os.path.join(path_raw, sujet, 'anatomy', 'anat_adjusted'))

    parcellisation = pd.read_excel(f'{sujet}_anatomical_localizations.xlsx')

    #### open nomenclature
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    #### identify plots 
    chan_list_ieeg_in_parcel = []
    for plot in chan_list_ieeg:
        if plot in parcellisation['Plot'].values:
            chan_list_ieeg_in_parcel.append(plot)

    #### categories to fill
    columns = ['subject', 'plot', 'MNI', 'freesurfer_destrieux', 'correspondance_ROI', 'correspondance_lobes', 'comparison', 'abscent',	
                'noisy_signal', 'inSOZ', 'not_in_atlas', 'select', 'localisation_corrected', 'lobes_corrected']

    #### find correspondances
    freesurfer_destrieux = [ parcellisation['FreesurferDesikan'][parcellisation['Plot'] == nchan].values.tolist()[0] for nchan in chan_list_ieeg_in_parcel]

    correspondance_ROI = []
    correspondance_lobes = []
    for parcel_i in freesurfer_destrieux:
        if parcel_i[0] == 'c':
            parcel_i_chunk = parcel_i[7:]
        elif parcel_i[0] == 'L':
            parcel_i_chunk = parcel_i[5:]
        elif parcel_i[0] == 'R':
            parcel_i_chunk = parcel_i[6:]
        elif parcel_i[0] == 'W':
            parcel_i_chunk = 'Cerebral-White-Matter'
        else:
            parcel_i_chunk = parcel_i

        if parcel_i_chunk == 'unknown':
            parcel_i_chunk = 'Unknown'
        
        correspondance_ROI.append(nomenclature['Our correspondances'][nomenclature['Labels'] == parcel_i_chunk].values[0])
        correspondance_lobes.append(nomenclature['Lobes'][nomenclature['Labels'] == parcel_i_chunk].values[0])

    #### generate MNI info
    MNI_loca = []
    for nchan in chan_list_ieeg_in_parcel:
        MNI_i = [parcellisation['MNI_x'][parcellisation['Plot'] == nchan].values[0], parcellisation['MNI_y'][parcellisation['Plot'] == nchan].values[0], parcellisation['MNI_z'][parcellisation['Plot'] == nchan].values[0]] 
        MNI_loca.append(MNI_i)

    #### generate df
    electrode_select_dict = {}
    for ncol in columns:
        if ncol == 'subject':
            electrode_select_dict[ncol] = [sujet] * len(chan_list_ieeg_in_parcel)
        elif ncol == 'plot':
            electrode_select_dict[ncol] = chan_list_ieeg_in_parcel
        elif ncol == 'MNI':
            electrode_select_dict[ncol] = MNI_loca
        elif ncol == 'freesurfer_destrieux':
            electrode_select_dict[ncol] = freesurfer_destrieux
        elif ncol == 'localisation_corrected' or ncol == 'lobes_corrected':
            electrode_select_dict[ncol] = [0] * len(chan_list_ieeg_in_parcel)
        elif ncol == 'correspondance_ROI':
            electrode_select_dict[ncol] = correspondance_ROI
        elif ncol == 'correspondance_lobes':
            electrode_select_dict[ncol] = correspondance_lobes
        else :
            electrode_select_dict[ncol] = [0] * len(chan_list_ieeg_in_parcel)


    #### generate df and bipolarize
    df = pd.DataFrame(electrode_select_dict, columns=columns)

    for row_i in df.index:
        plot_name, plot_i = df['plot'][row_i].split('_')[0], df['plot'][row_i].split('_')[-1]
        if len(plot_i) == 1:
            df['plot'][row_i] = f'{plot_name}_0{plot_i}'

    df = df.sort_values('plot')
    df.index = range(df.index.shape[0])

    #### separate electrodes
    plot_name_bip = []
    plot_ROI_bip = []
    plot_Lobes_bip = []

    verif_count = 0
    verif_count_name_bip = 0
    verif_count_anat_ROI_bip = 0
    verif_count_anat_Lobes_bip = 0

    plot_list_unique = np.unique(np.array([plot_i.split('_')[0] for plot_i in df['plot'].values]))
    #plot_unique_i = plot_list_unique[1]
    for plot_unique_i in plot_list_unique:
        plot_selection_i = np.array([plot_i for plot_i, plot_name in enumerate(df['plot'].values) if plot_name.find(plot_unique_i) != -1])
        
        verif_count += plot_selection_i.shape[0]-1
        
        plot_name_sel = df['plot'][plot_selection_i].values
        plot_name_sel_bipol = bipolarize_anatomy_name(plot_name_sel)

        anat_selection_ROI = df['correspondance_ROI'][plot_selection_i].values
        anat_selection_Lobes = df['correspondance_lobes'][plot_selection_i].values

        anat_selection_ROI_bi = bipolarize_anatomy_localization(anat_selection_ROI)
        anat_selection_Lobes_bi = bipolarize_anatomy_localization(anat_selection_Lobes)

        verif_count_name_bip += len(plot_name_sel_bipol)
        verif_count_anat_ROI_bip += len(anat_selection_ROI_bi)
        verif_count_anat_Lobes_bip += len(anat_selection_Lobes_bi)

        plot_name_bip.extend(plot_name_sel_bipol)
        plot_ROI_bip.extend(anat_selection_ROI_bi)
        plot_Lobes_bip.extend(anat_selection_Lobes_bi)

    #### verif bipol
    if verif_count != verif_count_anat_ROI_bip or verif_count != verif_count_anat_Lobes_bip or verif_count != verif_count_name_bip:
        raise ValueError('!! WARNING !! bipolarization issue')

    #### update df for bipol
    df = df.iloc[:verif_count, :]
    df = df.drop('MNI', axis=1)
    df = df.drop('freesurfer_destrieux', axis=1)
    df['plot'] = plot_name_bip
    df['correspondance_ROI'] = plot_ROI_bip
    df['correspondance_lobes'] = plot_Lobes_bip

    #### export
    os.chdir(os.path.join(path_anatomy, sujet))
    
    df.to_excel(sujet + '_plot_loca_bi.xlsx')

    return
        


        






################################
######## EXECUTE ########
################################


if __name__== '__main__':

    #### whole protocole
    sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    # sujet = 'pat_03138_1601'
    # sujet = 'pat_03146_1608'
    # sujet = 'pat_03174_1634'

    construct_token = generate_folder_structure(sujet)

    if construct_token != 0 :
        
        print('Folder structure has been generated')
        print('Lauch the script again for electrode selection')

    else:

        os.chdir(os.path.join(path_anatomy, sujet))
        if os.path.exists(sujet + '_plot_loca.xlsx'):
            print('#### ALREADY COMPUTED ####')
            exit()

        #### execute
        chan_list_ncs = extract_chanlist(sujet)
        generate_plot_loca(chan_list_ncs)
        generate_plot_loca_bipolaire(chan_list_ncs)







