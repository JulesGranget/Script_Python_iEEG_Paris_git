

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import xarray as xr
from frites.dataset import SubjectEphy, DatasetEphy
from frites.conn import conn_te, conn_covgc
from frites.workflow import WfMi


from n0_config import *
from n0bis_analysis_functions import *

debug = False






########################################
######## PROCESS SNIFF ERP ########
########################################


def process_sniff_ERP():

    print('SNIFF COMPUTE')

    cond = 'SNIFF'

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions()
    df_loca = get_loca_df(sujet)

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    band_prep = 'lf'

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find(cond) != -1) and (session_name.find('nc') != -1) :
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]
    
    xr_sniff = xr.open_dataarray(load_list[0])
    chan_list = xr_sniff['chan_list'].data

    #### mean on sniffs
    xr_sniff_mean = xr_sniff.mean('sniffs')

    #### chdir
    os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))

    #nchan = xr_sniff['chan_list'].values[0]
    for nchan in xr_sniff['chan_list'].values:

        chan_loca = df_loca['ROI'][df_loca['name'] == nchan].values[0]

        fig, ax = plt.subplots()
        ax.plot(xr_sniff_mean['times'].data, xr_sniff_mean.sel(chan_list=nchan).data)
        ax.vlines(0, ymin=np.min(xr_sniff_mean.sel(chan_list=nchan).data) ,ymax=np.max(xr_sniff_mean.sel(chan_list=nchan).data), colors='r')
        ax.set_title(f'{sujet}_{nchan}_{chan_loca}')
        #plt.show()
        fig.savefig(f'{sujet}_{nchan}_{chan_loca}.jpeg', dpi=600)


            

########################################
######## SNIFF CONNECTIVITY ########
########################################


def process_sniff_connectivity():

    cond = 'SNIFF'

    print('SNIFF COMPUTE')

    #### load params

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions()

    #### load data

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    band_prep = 'lf'

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find(cond) != -1) and (session_name.find('nc') != -1) :
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]
    
    xr_sniff = xr.open_dataarray(load_list[0])
    xr_sniff = xr_sniff.transpose('sniffs', 'chan_list', 'times')

    #### compute covgc

    dt = SubjectEphy(xr_sniff, roi='chan_list', times='times')
    print(dt)

    lag = 10
    win = 100
    t0 = np.arange(win, 1600, lag) #dt, 
    gc = conn_covgc(dt, win, lag, t0, step=10, times='times', roi='roi', method='gauss', n_jobs=6)
    
    gc = gc.mean('trials')


    #### plot

    #### for one chan
    roi_pairs = gc['roi'].data
    roi_direction = gc['direction'].data

    nchan = chan_list[0]
    mask = [i for i, name in enumerate(roi_pairs) if name[:len(nchan)+1] == f'{nchan}-']
    nchan_pairs = roi_pairs[mask]

    n_visu = 1
    for i in range(int(nchan_pairs.shape[0]/n_visu)+1):

        try:
            roi_p = nchan_pairs[i*n_visu:i*n_visu+n_visu]
        except:
            roi_p = nchan_pairs[i*n_visu:]
    
        plt.figure(figsize=(10, 8))
        for r in roi_p:
            plt.plot(gc.times.data, gc.sel(roi=r, direction='x->y').T, label=r[len(nchan)+1:])
        plt.legend()
        plt.show()

    #### for all chan
    gc_mean = np.mean(gc.sel(direction='x->y').mean('roi').data)
    gc_std = np.std(gc.sel(direction='x->y').mean('roi').data)

    pairs_signi = []

    for r in roi_pairs:

        mean_ = np.mean(gc.sel(roi=r, direction='x->y').data)
        std_ = np.std(gc.sel(roi=r, direction='x->y').data)

        if (np.max(gc.sel(roi=r, direction='x->y').data) >= (mean_ + 3.5*std_)) or (np.min(gc.sel(roi=r, direction='x->y').data) <= (mean_ - 3.5*std_)):

            pairs_signi.append(r)

    
    r = pairs_signi[0]
    for r in pairs_signi:
        plt.title(r)
        plt.plot(gc.times.data, gc.sel(roi=r, direction='x->y').T, label=r[len(nchan)+1:])
        plt.legend()
        plt.show()


    #### MUTUAL INFORMATION
    # define an electrophysiological dataset
    ds = DatasetEphy([xr_sniff], y='sniffs', times='times', roi='chan_list')
    # define a workflow of mutual information
    wf = WfMi(mi_type='cd', inference='rfx')
    # run the workflow
    mi, pv = wf.fit(ds, n_perm=200, n_jobs=6, random_state=0)














    mask = np.where(gc.sel(direction='x->y').data >= gc_mean + 40*gc_std)[0]
    mask = np.where(gc.sel(direction='x->y').data <= gc_mean - 40*gc_std)[0]

    plt.plot(gc.sel(direction='x->y').data[mask][0])
    plt.show()
    len(mask)
    
    plt.plot(gc.sel(direction='x->y').mean('roi').data)
    plt.plot(gc.sel(direction='x->y').std('roi').data)
    plt.show()

    roi_pairs = gc['roi'].data
    roi_direction = gc['direction'].data

    nchan = chan_list[0]
    mask = [i for i, name in enumerate(roi_pairs) if name[:len(nchan)+1] == f'{nchan}-']
    nchan_pairs = roi_pairs[mask]

    n_visu = 1
    for i in range(int(nchan_pairs.shape[0]/n_visu)+1):

        try:
            roi_p = nchan_pairs[i*n_visu:i*n_visu+n_visu]
        except:
            roi_p = nchan_pairs[i*n_visu:]
    
        plt.figure(figsize=(10, 8))
        for r in roi_p:
            plt.plot(gc.times.data, gc.sel(roi=r, direction='x->y').T, label=r[len(nchan)+1:])
        plt.legend()
        plt.show()

    

    n_visu = 10
    for i in range(20):

        try:
            roi_p = gc['roi'].data[i*n_visu:i*n_visu+n_visu]
        except:
            roi_p = gc['roi'].data[i*n_visu:]

        plt.title(i)
        plt.figure(figsize=(10, 8))
        plt.subplot(311)
        for r in roi_p:
            plt.plot(gc.times.data, gc.sel(roi=r, direction='x->y').T, label=r.replace('-', ' -> '))
        plt.legend()
        plt.subplot(312)
        for r in roi_p:
            plt.plot(gc.times.data, gc.sel(roi=r, direction='y->x').T, label=r.replace('-', ' <- '))
        plt.legend()
        plt.subplot(313)
        for r in roi_p:
            plt.plot(gc.times.data, gc.sel(roi=r, direction='x.y').T, label=r.replace('-', ' . '))
        plt.legend()
        plt.xlabel('Time')
        plt.show()


################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    cond = 'SNIFF'

    #process_sniff_ERP()

    execute_function_in_slurm_bash('n9_sniff_analysis', 'process_sniff_ERP', [])
        









