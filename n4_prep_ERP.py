

import os
import numpy as np
import matplotlib.pyplot as plt
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






########################################
######## COMPUTE FUNCTIONS ########
########################################



def compute_chunk_AC(data, ac_starts, prms):

    srate = prms['srate']

    #### chunk
    stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
    data_stretch = np.zeros((len(ac_starts), data.shape[0], int(stretch_point_TF_ac)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(ac_starts):

            t_start = int(start_time + t_start_AC*srate)
            t_stop = int(start_time + t_stop_AC*srate)

            data_stretch[start_i, nchan, :] = x[t_start: t_stop]

    return data_stretch




def compute_chunk_SNIFF(data, sniff_starts, prms):

    srate = prms['srate']

    #### chunk
    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*srate +  t_stop_SNIFF*srate)
    data_stretch = np.zeros((len(sniff_starts), data.shape[0], int(stretch_point_TF_sniff)))

    #nchan = 0
    for nchan in range(data.shape[0]):

        x = data[nchan,:]

        for start_i, start_time in enumerate(sniff_starts):

            t_start = int(start_time + t_start_SNIFF*srate)
            t_stop = int(start_time + t_stop_SNIFF*srate)

            data_stretch[start_i, nchan, :] = x[t_start: t_stop]

    return data_stretch









################################
######## ERP ANALYSIS ########
################################


def compute_ERP(sujet, electrode_recording_type):

    prms = get_params(sujet, electrode_recording_type)

    data_stretch_allcond = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        data_stretch_allcond[band_prep] = {}

        #cond = 'SNIFF'
        for cond in ['AC', 'SNIFF']:

            #### select data without aux chan
            data = load_data(sujet, cond, electrode_recording_type)
            data = data[:-3,:]

            #### stretch or chunk
            if cond == 'AC':
                ac_starts = get_ac_starts_uncleaned(sujet)
                data_stretch = compute_chunk_AC(data, ac_starts, prms)

            if cond == 'SNIFF':
                sniff_starts = get_sniff_starts_uncleaned(sujet)
                data_stretch = compute_chunk_SNIFF(data, sniff_starts, prms)

            data_stretch_allcond[band_prep][cond] = data_stretch

            #### verif
            if debug:

                plt.plot(data[-4,:])
                plt.show()

    return data_stretch_allcond








def plot_ERP(sujet, data_stretch_allcond, electrode_recording_type):

    print('ERP PLOT')

    os.chdir(os.path.join(path_precompute, sujet, 'ERP'))

    prms = get_params(sujet, electrode_recording_type)
    df_loca = get_loca_df(sujet, electrode_recording_type)

    #nchan_i, nchan = len(prms['chan_list'][:-3])-1, prms['chan_list'][:-3][-1]
    for nchan_i, nchan in enumerate(prms['chan_list'][:-3]):

        if nchan == 'nasal':
            chan_loca = 'nasal'    
        else:
            chan_loca = df_loca['ROI'][df_loca['name'] == nchan].values[0]

        #band_prep_i, band_prep = 0, 'lf'
        for band_prep_i, band_prep in enumerate(band_prep_list):

            fig, axs = plt.subplots(nrows=3)

            if electrode_recording_type == 'monopolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}')
            if electrode_recording_type == 'bipolaire':
                plt.suptitle(f'{nchan}_{chan_loca}_{band_prep}_bi')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #cond_i, cond = 0, 'FR_CV'
            for cond_i, cond in enumerate(['FR_CV', 'AC', 'SNIFF']):

                data_stretch = data_stretch_allcond[band_prep][cond]

                ax = axs[cond_i]
                ax.set_title(f'{cond} : {data_stretch.shape[0]}', fontweight='bold')

                if cond == 'FR_CV':
                    time_vec = np.arange(stretch_point_TF)

                if cond == 'AC':
                    stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
                    time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)

                if cond == 'SNIFF':
                    stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
                    time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)

                ax.plot(time_vec, data_stretch.mean(axis=0)[nchan_i,:], color='b')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:], color='k', linestyle='--')
                ax.plot(time_vec, data_stretch.std(axis=0)[nchan_i,:]*-1, color='k', linestyle='--')

                max_plot = np.stack((data_stretch.mean(axis=0)[nchan_i,:], data_stretch.std(axis=0)[nchan_i,:])).max()

                if cond == 'FR_CV':
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=max_plot*-1, ymax=max_plot, colors='g')
                if cond == 'AC':
                    ax.vlines([0, 12], ymin=max_plot*-1, ymax=max_plot, colors='g')
                if cond == 'SNIFF':
                    ax.vlines(0, ymin=max_plot*-1, ymax=max_plot, colors='g')

            #plt.show()

            #### save
            if electrode_recording_type == 'monopolaire':
                fig.savefig(f'{sujet}_{nchan}_{chan_loca}_{band_prep}.jpeg', dpi=150)
            if electrode_recording_type == 'bipolaire':
                fig.savefig(f'{sujet}_{nchan}_{chan_loca}_{band_prep}_bi.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()




################################
######## ERP CLEANING ######## 
################################


def save_erp_cleaning(sujet):

    prms = get_params(sujet, electrode_recording_type)

    data_stretch_allcond = compute_ERP(sujet, electrode_recording_type) 

    #cond = 'AC'
    for cond in ['AC', 'SNIFF']:     

        if cond == 'AC':
            stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
            time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
            vlines_plot = [0, 12]
            erp_starts = get_ac_starts_uncleaned(sujet)

        if cond == 'SNIFF':
            stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
            time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
            vlines_plot = [0]
            erp_starts = get_sniff_starts_uncleaned(sujet)

        data_erp = zscore_mat(data_stretch_allcond['lf'][cond][:, -1, :])

        #### import exclusion and adjust
        erp_include = np.ones((len(erp_starts)), dtype='bool')
        erp_to_exclude = get_erp_to_exclude(sujet)

        try:
            erp_include[erp_to_exclude[cond]] = False
        except:
            pass

        erp_starts_filt = np.array(erp_starts)[erp_include]

        #### save fig
        os.chdir(os.path.join(path_precompute, sujet, 'ERP'))
        plt.figure(figsize=(10,8))
        for erp_time in erp_starts_filt:

            erp_i = erp_starts.index(erp_time)

            plt.plot(time_vec, data_erp[erp_i, :], alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max(), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(f'{cond} : {erp_starts_filt.shape[0]}')
        # plt.show()
        plt.savefig(f'{sujet}_{cond}_select.png')
        plt.close()

        for jitter_i, erp_time in enumerate(erp_starts_filt):

            erp_i = erp_starts.index(erp_time)

            plt.plot(time_vec, data_erp[erp_i, :]+jitter_i+1, alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max()+(len(erp_starts)), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(f'{cond} : {erp_starts_filt.shape[0]}')
        # plt.show()
        plt.savefig(f'{sujet}_{cond}_select_jitter.png')
        plt.close()

        #### save values
        np.save(f'{sujet}_{cond}_select', erp_starts_filt)





def get_erp_to_exclude(sujet):

    if sujet == 'pat_03083_1527':
        erp_to_exclude =    {'AC' : np.array([-1]),
                            'SNIFF' : np.array([])} 
    if sujet == 'pat_03105_1551':
        erp_to_exclude =    {'AC' : np.array([13, 18]),
                            'SNIFF' : np.array([])} 
    if sujet == 'pat_03128_1591':
        erp_to_exclude =    {'AC' : np.array([6, 12, 13, 23]),
                            'SNIFF' : np.array([])} 
    if sujet == 'pat_03138_1601':
        erp_to_exclude =    {'AC' : np.array([0, 3, 11]),
                            'SNIFF' : np.array([])} 
    if sujet == 'pat_03146_1608':
        erp_to_exclude =    {'AC' : np.array([6]),
                            'SNIFF' : np.array([])} 
    if sujet == 'pat_03174_1634':
        erp_to_exclude =    {'AC' : np.array([10, 24, 25]),
                            'SNIFF' : np.array([0])} 

    return erp_to_exclude






################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    ######## PREPARE ########

    #### whole protocole
    # sujet = 'pat_03083_1527'
    # sujet = 'pat_03105_1551'
    # sujet = 'pat_03128_1591'
    # sujet = 'pat_03138_1601'
    # sujet = 'pat_03146_1608'
    # sujet = 'pat_03174_1634'

    electrode_recording_type = 'monopolaire'

    print(sujet, electrode_recording_type)

    #### compute data
    data_stretch_allcond = compute_ERP(sujet, electrode_recording_type) 

    #### artifact rejection
    if debug:

        prms = get_params(sujet, electrode_recording_type)

        nchan = -1

            #### load data
        cond = 'AC'
        
        cond = 'SNIFF'
        
        if cond == 'AC':
            stretch_point_TF_ac = int(np.abs(t_start_AC)*prms['srate'] +  t_stop_AC*prms['srate'])
            time_vec = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
            vlines_plot = [0, 12]
            erp_starts = get_ac_starts_uncleaned(sujet)

        if cond == 'SNIFF':
            stretch_point_TF_sniff = int(np.abs(t_start_SNIFF)*prms['srate'] +  t_stop_SNIFF*prms['srate'])
            time_vec = np.linspace(t_start_SNIFF, t_stop_SNIFF, stretch_point_TF_sniff)
            vlines_plot = [0]
            erp_starts = get_sniff_starts_uncleaned(sujet)

        data_erp = zscore_mat(data_stretch_allcond['lf'][cond][:, nchan, :])

            #### verif erp on sig
        respi = load_data(sujet, cond, electrode_recording_type)[-3,:]
        plt.plot(respi)
        plt.vlines(erp_starts, ymax=respi.max(), ymin=respi.min(), color='r')
        plt.show()

            #### all erp
        for erp_i, erp_time in enumerate(erp_starts):

            plt.plot(time_vec, data_erp[erp_i, :], alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max(), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(cond)
        plt.show()

            #### all erp jitter
        for erp_i, erp_time in enumerate(erp_starts):

            plt.plot(time_vec, data_erp[erp_i, :]+erp_i+1, alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max()+(len(erp_starts)), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(cond)
        plt.show()

            #### every erp
        for erp_i, erp_time in enumerate(erp_starts):

            plt.plot(time_vec, data_erp[erp_i, :])
            plt.vlines(vlines_plot, ymax=data_erp.max(), ymin=data_erp.min(), color='r')
            plt.title(f'{erp_i}/{len(erp_starts)-1}')
            plt.show()

            #### exclude
        erp_include = np.ones((len(erp_starts)), dtype='bool')
        erp_i_exclude = [0, 3, 11]

        erp_include[erp_i_exclude] = False

        erp_starts_filt = np.array(erp_starts)[erp_include]

            #### verify all erp
        for erp_time in erp_starts_filt:

            erp_i = erp_starts.index(erp_time)

            plt.plot(time_vec, data_erp[erp_i, :], alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max(), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(cond)
        plt.show()

            #### verify all erp jitter
        for jitter_i, erp_time in enumerate(erp_starts_filt):

            erp_i = erp_starts.index(erp_time)

            plt.plot(time_vec, data_erp[erp_i, :]+jitter_i+1, alpha=0.3)

        plt.vlines(vlines_plot, ymax=data_erp.max()+(len(erp_starts)), ymin=data_erp.min(), color='k')
        plt.plot(time_vec, data_erp.mean(axis=0), color='r')
        plt.title(cond)
        plt.show()



    ######## PROCESS ########

    electrode_recording_type = 'monopolaire'

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)
        save_erp_cleaning(sujet)


