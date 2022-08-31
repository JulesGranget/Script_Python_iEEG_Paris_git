


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import seaborn as sns

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False



################################
######## COMPUTE ########
################################


def count_all_plot_location():

    os.chdir(os.path.join(path_anatomy, 'nomenclature'))

    nomenclature_df = pd.read_excel('Freesurfer_Parcellisation_Destrieux.xlsx')
    
    #### fill dict with anat names
    anat_loca_dict = {}
    anat_lobe_dict = {}
    anat_loca_list = nomenclature_df['Our correspondances'].values
    anat_lobe_list_non_sorted = nomenclature_df['Lobes'].values
    for i in range(len(anat_loca_list)):
        anat_loca_dict[anat_loca_list[i]] = 0
        anat_lobe_dict[anat_lobe_list_non_sorted[i]] = 0

    anat_loca_dict_FR_CV = anat_loca_dict.copy()
    anat_lobe_dict_FR_CV = anat_lobe_dict.copy()
    
    anat_ROI_noselect_dict = anat_loca_dict.copy()
    anat_lobe_noselect_dict = anat_lobe_dict.copy()

    anat_ROI_noselect_dict_FR_CV = anat_loca_dict.copy()
    anat_lobe_noselect_dict_FR_CV = anat_lobe_dict.copy()

    #### for FR_CV
    #sujet_i = sujet_list[0]
    for sujet_i in sujet_list:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        chan_list_ieeg_csv = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        count_verif = 0

        for nchan in chan_list_ieeg_csv:

            loca_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            anat_loca_dict_FR_CV[loca_tmp] = anat_loca_dict_FR_CV[loca_tmp] + 1
            anat_lobe_dict_FR_CV[lobe_tmp] = anat_lobe_dict_FR_CV[lobe_tmp] + 1
            count_verif += 1

        #### verif count
        if count_verif != len(chan_list_ieeg_csv):
            print('ERROR : anatomical count is not correct, count != len chan_list')
            exit()


    #### for whole protocole all subjects
    #sujet_i = sujet_list[0]
    for sujet_i in sujet_list:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')

        chan_list_ieeg_csv = plot_loca_df['plot'][plot_loca_df['select'] == 1].values

        count_verif = 0

        for nchan in chan_list_ieeg_csv:

            loca_tmp = plot_loca_df['localisation_corrected'][plot_loca_df['plot'] == nchan].values[0]
            lobe_tmp = plot_loca_df['lobes_corrected'][plot_loca_df['plot'] == nchan].values[0]
            
            anat_loca_dict[loca_tmp] = anat_loca_dict[loca_tmp] + 1
            anat_lobe_dict[lobe_tmp] = anat_lobe_dict[lobe_tmp] + 1
            count_verif += 1

        #### verif count
        if count_verif != len(chan_list_ieeg_csv):
            print('ERROR : anatomical count is not correct, count != len chan_list')
            exit()


    #### for all plot, i. e. not included FR_CV
    df_all_plot_noselect_FR_CV = pd.DataFrame(columns=plot_loca_df.columns)
    for sujet_i in sujet_list:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        df_all_plot_noselect_FR_CV = pd.concat([df_all_plot_noselect_FR_CV, plot_loca_df])

    df_all_plot_noselect_FR_CV.index = np.arange(df_all_plot_noselect_FR_CV.index.shape[0])

    for i in df_all_plot_noselect_FR_CV.index.values:

        ROI_tmp = df_all_plot_noselect_FR_CV['localisation_corrected'][i]
        lobe_tmp = df_all_plot_noselect_FR_CV['lobes_corrected'][i]
        sujet_tmp = df_all_plot_noselect_FR_CV['subject'][i]
        
        if sujet_tmp in sujet_list:
            anat_ROI_noselect_dict_FR_CV[ROI_tmp] = anat_ROI_noselect_dict_FR_CV[ROI_tmp] + 1
            anat_lobe_noselect_dict_FR_CV[lobe_tmp] = anat_lobe_noselect_dict_FR_CV[lobe_tmp] + 1

    df_data_ROI_FR_CV = {'ROI' : list(anat_loca_dict.keys()), 'ROI_Count_No_Included' : list(anat_ROI_noselect_dict_FR_CV.values()), 'ROI_Count_Included' : list(anat_loca_dict_FR_CV.values())}
    df_data_Lobes_FR_CV = {'Lobes' : list(anat_lobe_dict.keys()), 'Lobes_Count_No_Included' : list(anat_lobe_noselect_dict_FR_CV.values()), 'Lobes_Count_Included' : list(anat_lobe_dict_FR_CV.values())}

    df_ROI_count_FR_CV = pd.DataFrame(df_data_ROI_FR_CV)
    df_lobes_count_FR_CV = pd.DataFrame(df_data_Lobes_FR_CV)


    #### for all plot, i. e. not included
    df_all_plot_noselect = pd.DataFrame(columns=plot_loca_df.columns)
    for sujet_i in sujet_list:

        os.chdir(os.path.join(path_anatomy, sujet_i))
        plot_loca_df = pd.read_excel(sujet_i + '_plot_loca.xlsx')
        df_all_plot_noselect = pd.concat([df_all_plot_noselect, plot_loca_df])

    df_all_plot_noselect.index = np.arange(df_all_plot_noselect.index.shape[0])

    for i in df_all_plot_noselect.index.values:

        ROI_tmp = df_all_plot_noselect['localisation_corrected'][i]
        lobe_tmp = df_all_plot_noselect['lobes_corrected'][i]
        sujet_tmp = df_all_plot_noselect['subject'][i]
        
        if sujet_tmp in sujet_list:
            anat_ROI_noselect_dict[ROI_tmp] = anat_ROI_noselect_dict[ROI_tmp] + 1
            anat_lobe_noselect_dict[lobe_tmp] = anat_lobe_noselect_dict[lobe_tmp] + 1

    df_data_ROI = {'ROI' : list(anat_loca_dict.keys()), 'ROI_Count_No_Included' : list(anat_ROI_noselect_dict.values()), 'ROI_Count_Included' : list(anat_loca_dict.values())}
    df_data_Lobes = {'Lobes' : list(anat_lobe_dict.keys()), 'Lobes_Count_No_Included' : list(anat_lobe_noselect_dict.values()), 'Lobes_Count_Included' : list(anat_lobe_dict.values())}

    df_ROI_count = pd.DataFrame(df_data_ROI)
    df_lobes_count = pd.DataFrame(df_data_Lobes)

    #### save df
    os.chdir(os.path.join(path_results, 'allplot', 'anatomy'))

    if os.path.exists('ROI_count.xlsx'):
        os.remove('ROI_count.xlsx')

    if os.path.exists('Lobes_count.xlsx'):
        os.remove('Lobes_count.xlsx')  

    df_ROI_count.to_excel('ROI_count.xlsx')
    df_lobes_count.to_excel('Lobes_count.xlsx')

    df_ROI_count_FR_CV.to_excel('FR_CV_ROI_count.xlsx')
    df_lobes_count_FR_CV.to_excel('FR_CV_Lobes_count.xlsx')

    #### save fig whole protocol FR_CV
    sns.catplot(x="ROI_Count_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count_FR_CV, height=10, aspect=1)
    plt.savefig('FR_CV_ROI_count_whole_protocol_included.png', dpi=150)
    plt.close()
    
    sns.catplot(x="Lobes_Count_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count_FR_CV, height=10, aspect=1)
    plt.savefig('FR_CV_Lobes_Count_whole_protocol_Included.png', dpi=150)
    plt.close()

    sns.catplot(x="ROI_Count_No_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count_FR_CV, height=10, aspect=1)
    plt.savefig('FR_CV_ROI_Count_whole_protocol_No_Included.png', dpi=150)
    plt.close()
    
    sns.catplot(x="Lobes_Count_No_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count_FR_CV, height=10, aspect=1)
    plt.savefig('FR_CV_Lobes_Count_whole_protocol_No_Included.png', dpi=150)
    plt.close()


    #### save fig whole protocol all
    sns.catplot(x="ROI_Count_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count, height=10, aspect=1)
    plt.savefig('ROI_count_whole_protocol_included.png', dpi=150)
    plt.close()
    
    sns.catplot(x="Lobes_Count_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count, height=10, aspect=1)
    plt.savefig('Lobes_Count_whole_protocol_Included.png', dpi=150)
    plt.close()

    sns.catplot(x="ROI_Count_No_Included", y="ROI", kind='bar', palette="pastel", edgecolor=".6", data=df_ROI_count, height=10, aspect=1)
    plt.savefig('ROI_Count_whole_protocol_No_Included.png', dpi=150)
    plt.close()
    
    sns.catplot(x="Lobes_Count_No_Included", y="Lobes", kind='bar', palette="pastel", edgecolor=".6", data=df_lobes_count, height=10, aspect=1)
    plt.savefig('Lobes_Count_whole_protocol_No_Included.png', dpi=150)
    plt.close()
    

################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    count_all_plot_location()
