
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from n0_config import *
from n0bis_analysis_functions import *


debug = False







def organize_raw(raw):

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
def extract_chanlist():

    print('#### EXTRACT ####')

    os.chdir(os.path.join(path_raw, sujet, 'raw_data', 'mat'))
    raw = mne.io.read_raw_eeglab(f'{sujet}_allchan.set', preload=True)
    
    data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate = organize_raw(raw)

    return chan_list_ieeg



def generate_plot_loca(chan_list_trc):


    #### open loca file
    os.chdir(os.path.join(path_raw, sujet, 'anatomy'))

    parcellisation = pd.read_excel('_' + sujet + '.xlsx')

    new_cols = parcellisation.iloc[1,:].values.tolist()
    parcellisation.columns = new_cols

    parcellisation = parcellisation.drop(labels=[0, 1], axis=0)
    new_indexs = range(len(parcellisation['contact'].values))
    parcellisation.index = new_indexs

    #### open nomenclature
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')

    #### identify if parcellisation miss plots
    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    plot_csv_extract = []
    for i in range(len(parcellisation.index)):
        if len(str(parcellisation.iloc[i,0])) <= 4 and str(parcellisation.iloc[i,0]) != 'nan':
            plot_csv_extract.append(parcellisation.iloc[i,0])

    plot_csv, test2 = modify_name(plot_csv_extract)

    chan_list_modified, chan_list_keep = modify_name(chan_list_trc)
    
    miss_plot = []
    chan_list_trc_rmw = []
    for nchan_i, nchan in enumerate(chan_list_modified):
        if nchan in plot_csv :
            chan_list_trc_rmw.append(chan_list_keep[nchan_i])
            continue
        else:
            miss_plot.append(nchan)
            

    #### export missed plots
    os.chdir(os.path.join(path_anatomy, sujet))
    miss_plot_textfile = open(sujet + "_miss_in_csv.txt", "w")
    for element in miss_plot:
        miss_plot_textfile.write(element + "\n")
    miss_plot_textfile.close()

    keep_plot_textfile = open(sujet + "_trcplot_in_csv.txt", "w")
    for element in chan_list_trc_rmw:
        keep_plot_textfile.write(element + "\n")
    keep_plot_textfile.close()

    #### categories to fill
    columns = ['subject', 'plot', 'MNI', 'freesurfer_destrieux', 'correspondance_ROI', 'correspondance_lobes', 'comparison', 'abscent',	
                'noisy_signal', 'inSOZ', 'not_in_atlas', 'select', 'localisation_corrected', 'lobes_corrected']

    #### find correspondances
    freesurfer_destrieux = [ parcellisation['Freesurfer'][parcellisation['contact'] == nchan].values.tolist()[0] for nchan in plot_csv_extract]
    if debug:
        for nchan in plot_csv_extract:
            print(nchan)
            parcellisation['Freesurfer'][parcellisation['contact'] == nchan].values.tolist()[0]

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
        
        correspondance_ROI.append(nomenclature['Our correspondances'][nomenclature['Labels'] == parcel_i_chunk].values[0])
        correspondance_lobes.append(nomenclature['Lobes'][nomenclature['Labels'] == parcel_i_chunk].values[0])

    #### generate df
    electrode_select_dict = {}
    for ncol in columns:
        if ncol == 'subject':
            electrode_select_dict[ncol] = [sujet] * len(plot_csv_extract)
        elif ncol == 'plot':
            electrode_select_dict[ncol] = plot_csv
        elif ncol == 'MNI':
            electrode_select_dict[ncol] = [ parcellisation['MNI'][parcellisation['contact'] == nchan].values.tolist()[0] for nchan in plot_csv_extract]
        elif ncol == 'freesurfer_destrieux':
            electrode_select_dict[ncol] = freesurfer_destrieux
        elif ncol == 'localisation_corrected' or ncol == 'lobes_corrected':
            electrode_select_dict[ncol] = [0] * len(plot_csv_extract)
        elif ncol == 'correspondance_ROI':
            electrode_select_dict[ncol] = correspondance_ROI
        elif ncol == 'correspondance_lobes':
            electrode_select_dict[ncol] = correspondance_lobes
        else :
            electrode_select_dict[ncol] = [1] * len(plot_csv_extract)


    #### generate df and save
    electrode_select_df = pd.DataFrame(electrode_select_dict, columns=columns)

    os.chdir(os.path.join(path_anatomy, sujet))
    
    electrode_select_df.to_excel(sujet + '_plot_loca.xlsx')

    return
        


if __name__== '__main__':

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
        chan_list_trc = extract_chanlist()
        generate_plot_loca(chan_list_trc)







