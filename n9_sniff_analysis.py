

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne

from mne_connectivity import spectral_connectivity
from mne_connectivity.viz import circular_layout, plot_connectivity_circle

import pandas as pd
import respirationtools
import joblib
import xarray as xr

from frites.dataset import SubjectEphy, DatasetEphy
from frites.conn import conn_covgc, conn_dfc
from frites.workflow import WfMi
from frites.conn import define_windows


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
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
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
######## PLOT FUNCTION ########
########################################

def plot_mi(mi, pv, roi_i):
    # figure definition
    r = mi['roi'].data[roi_i]
    fig, gs  = plt.subplots()

    n_r = np.where(mi['roi'].data == r)[0]
    # select mi and p-values for a single roi
    mi_r, pv_r = mi.sel(roi=r), pv.sel(roi=r)
    # set to nan when it's not significant
    mi_r_s = mi_r.copy()
    mi_r_s[pv_r >= .05] = np.nan

    # significant = red; non-significant = black
    plt.plot(mi['times'].data, mi_r, lw=1, color='k')
    plt.plot(mi['times'].data, mi_r_s, lw=3, color='red')
    plt.xlabel('Times'), plt.ylabel('MI (bits)')
    plt.title(f"ROI={r}")
    plt.axvline(0, lw=2, color='b')

    return plt.gcf()














########################################
######## SNIFF CONNECTIVITY ########
########################################


def zscore(sig):

    sig_clean = (sig - np.mean(sig)) / np.std(sig)

    return sig_clean


def process_sniff_connectivity():

    cond = 'SNIFF'

    print('SNIFF COMPUTE')

    #### load params

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    sniff_starts = get_sniff_starts(sujet)
    times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)

    #### load data

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    sniff_starts = get_sniff_starts(sujet)
    times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find('hf.nc') != -1) :
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]

    df_loca = get_loca_df(sujet)

    xr_sniff = xr.open_dataarray(load_list[0])
    xr_sniff = xr_sniff.transpose('sniffs', 'chan_list', 'times')
    xr_sniff['sniffs'] = [0]*len(sniff_starts)
    xr_sniff['chan_list'] = df_loca['ROI'].values

    slwin_len = .3    # 100ms window length
    slwin_step = .02  # 80ms between consecutive windows
    win_sample = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]

    # compute the DFC for each subject
    dfc = conn_dfc(xr_sniff.data, win_sample, times=times, roi=chan_list[:-4], n_jobs=6, verbose=False)
    # reset trials dimension
    dfc['trials'] = xr_sniff['sniffs'].data

    dfc_suj = xr.concat(dfc, 'trials').groupby('trials').mean('trials')



    dfc_search = dfc_suj.data.reshape(dfc_suj.shape[1], dfc_suj.shape[2])
    pair_signi = []
    for pair_i in range(dfc_search.shape[0]):
        x = dfc_search[pair_i,:]
        x_mean = np.mean(x)
        x_std = np.std(x)
        thresh = x_mean + 3*x_std
        if len(np.where(dfc_search[pair_i,:] >= thresh)[0]) == 0:
            continue
        else:
            pair_signi.append(pair_i)

    for pair_i in pair_signi:
        dfc_suj[0,pair_i,:].plot()
        plt.show()


    #### segment time
    time_pre = 0
    time_post = .5
    time_list = ['pre', 'inst', 'post']
    times_pre = dfc_suj['times'].values[np.where(dfc_suj['times'].data < time_pre)[0]]
    times_inst = dfc_suj['times'].values[np.where((dfc_suj['times'].data > time_pre) & (dfc_suj['times'].data < time_post))[0]]
    times_post = dfc_suj['times'].values[np.where(dfc_suj['times'].data > time_post)[0]]

    if debug:
        for pair_i in dfc_suj['roi'].data[:10]:
            for time_sel_i, time_sel in enumerate([times_pre, times_inst, times_post]):
                dfc_suj.sel(trials=0, roi=pair_i, times=time_sel).mean().values
                plt.plot(dfc_suj.sel(trials=0, roi=pair_i, times=time_sel).values)
                plt.show()
            

    #### generate matrix
    mat_dfc = np.zeros(( 3, len(chan_list[:-4]), len(chan_list[:-4]) ))
    #pair_i = dfc_suj['roi'].values[0]
    for time_sel_i, time_sel in enumerate([times_pre, times_inst, times_post]):
        for pair_i in dfc_suj['roi'].data:
            pair_A, pair_B = pair_i.split('-')
            pair_A_i, pair_B_i = chan_list.index(pair_A), chan_list.index(pair_B)
            mat_dfc[time_sel_i, pair_A_i, pair_B_i] = dfc_suj.sel(trials=0, roi=pair_i, times=time_sel).max().values
            mat_dfc[time_sel_i, pair_B_i, pair_A_i] = dfc_suj.sel(trials=0, roi=pair_i, times=time_sel).max().values


    #### sorting
    def sort_mat(mat):

        mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
        for i_before_sort_r, i_sort_r in enumerate(df_sorted.index.values):
            for i_before_sort_c, i_sort_c in enumerate(df_sorted.index.values):
                mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

        return mat_sorted

    #### verify sorting
    #mat = pli_allband_reduced.get(band).get(cond)
    #mat_sorted = sort_mat(mat)
    #plt.matshow(mat_sorted)
    #plt.show()

    #### prepare sort
    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    chan_name_sorted_mat = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        if i == 0:
            chan_name_sorted_mat.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_mat.append('')
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)]:
                chan_name_sorted_mat.append(name_i)
                rep_count = 0
                continue


    #### plot
    fig, axs = plt.subplots(ncols=3)
    for c in range(3):
        ax = axs[c]
        ax.set_title(time_list[c])
        ax.matshow(sort_mat(mat_dfc[c, :, :]), vmin=np.min(dfc_suj.values), vmax=np.max(dfc_suj.values))
        if c == 0:
            ax.set_yticks(np.arange(len(chan_list[:-4])))
            ax.set_yticklabels(chan_name_sorted_mat)
    plt.show()













    dt = SubjectEphy(xr_sniff, roi='chan_list', times='times')
    print(dt)

    lag = 10
    win = 100
    t0 = np.arange(win, 1600, lag) #dt, 
    gc = conn_covgc(dt, win, lag, t0, step=10, times='times', roi='roi', method='gauss', n_jobs=6)
    
    gc = gc.mean('trials')


    #### plot
    #### for all chan
    gc_mean = gc.mean('trials').T
    roi_pairs = gc_mean['roi'].data
    roi_direction = gc_mean['direction'].data[:-1]


    #for nchan in chan_list_ieeg:
    def get_max_signi_nchan(nchan):

        pairs_max = []
        nchan = chan_list_ieeg[nchan]

        print(nchan)

        mask = [i for i, name in enumerate(roi_pairs) if name[:len(nchan)+1] == f'{nchan}-']
        nchan_pairs = roi_pairs[mask]

        for r in nchan_pairs:

            for r_dir in roi_direction:

                mean_r = np.mean(gc_mean.sel(roi=r, direction=r_dir).data)
                std_r = np.std(gc_mean.sel(roi=r, direction=r_dir).data)

                z_max = (np.max(gc_mean.sel(roi=r, direction=r_dir).data) - mean_r)/std_r

                pairs_max.append(z_max)

        return pairs_max

    n_core = 15
    n_nchan = len(chan_list_ieeg)
    pair_max_allchan = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_max_signi_nchan)(nchan) for nchan in range(n_nchan))


    pair_max = [] 

    for nchan in range(len(pair_max_allchan)):

        for r in range(len(pair_max_allchan[nchan])):
            
            pair_max.append(pair_max_allchan[nchan][r])

    pair_max = np.array(pair_max)
    mask = len(np.where(pair_max >= 4)[0])

    std_choose = 4


    #for nchan in chan_list_ieeg:
    def get_pair_signi_nchan(nchan_i):

        pairs_signi = []
        nchan = chan_list_ieeg[nchan_i]

        print(nchan)

        #name = roi_pairs[0]
        mask = [i for i, name in enumerate(roi_pairs) if name[:len(nchan)+1] == f'{nchan}-']
        nchan_pairs = roi_pairs[mask]

        for r in nchan_pairs:

            for r_dir in roi_direction:

                mean_r = np.mean(gc_mean.sel(roi=r, direction=r_dir).data)
                std_r = np.std(gc_mean.sel(roi=r, direction=r_dir).data)

                if np.max(gc_mean.sel(roi=r, direction=r_dir).data) >= (mean_r + std_choose*std_r):

                    pairs_signi.append([r, r_dir])

        return pairs_signi

    n_nchan = 30
    n_core = 15
    n_nchan = len(chan_list_ieeg)
    pair_signi_allchan = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_pair_signi_nchan)(nchan_i) for nchan_i in range(n_nchan))

    pair_signi = []

    for nchan in range(len(pair_signi_allchan)):

        for pair_i in range(len(pair_signi_allchan[nchan])):
            
            pair_signi.append(pair_signi_allchan[nchan][pair_i])

    
    os.chdir('/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/brouillon')

    n_visu = 20
    
    for pair_i in range(n_visu):
        r = pair_signi[pair_i][0]
        dir = pair_signi[pair_i][1]
        plt.title(f'{r}_{dir}')
        plt.plot(gc_mean.times.data, gc_mean.sel(roi=r, direction=dir))
        plt.savefig(f'{r}.png')
        plt.close()




    #### SPECTRAL CONN
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    xr_sniff_SC = xr.open_dataarray(load_list[0])
    xr_sniff_SC = xr_sniff_SC.transpose('sniffs', 'chan_list', 'times')

    mne_sniff_info = mne.create_info(list(xr_sniff_SC['chan_list'].data), srate, ch_types=['seeg']*len(xr_sniff_SC['chan_list']))
    mne_sniff = mne.EpochsArray(xr_sniff_SC.data, info=mne_sniff_info)

    metrics = ['coh', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
    for metric_i in metrics:

        fmin = 8.
        fmax = 13.
        n_core = 15
        con = spectral_connectivity(mne_sniff, method=metric_i, mode='multitaper', sfreq=srate, fmin=fmin, fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=n_core)

        conmat = con.get_data(output='dense')[:, :, 0]

        os.chdir('/crnldata/cmo/multisite/DATA_MANIP/iEEG_Paris_J/brouillon')

        fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
        plot_connectivity_circle(conmat, chan_list_ieeg, title=f'{metric_i}', fig=fig)
        fig.savefig(f'{metric_i}.png')

        fig, ax = plt.subplots()
        ax.matshow(conmat)
        fig.savefig(f'mat_{metric_i}.png')



    #### MUTUAL INFORMATION
    # define an electrophysiological dataset
    xr_sniff_MI = xr.open_dataarray(load_list[0])
    xr_sniff_MI = xr_sniff_MI.transpose('sniffs', 'chan_list', 'times')

    mne_sniff_info = mne.create_info(list(xr_sniff_MI['chan_list'].data), srate, ch_types=['seeg']*len(xr_sniff_MI['chan_list']))
    mne_sniff = mne.EpochsArray(xr_sniff_MI.data, info=mne_sniff_info)



    con = spectral_connectivity(mne_sniff, method='pli', mode='multitaper', sfreq=srate, fmin=8, fmax=12, faverage=True, tmin=0, mt_adaptive=False, n_jobs=1)

    con.plot_circle()



    xr_sniff_MI['sniffs'] = np.array(['free']*len(xr_sniff['sniffs'].data), dtype='object')
    se = SubjectEphy(mne_sniff)
    ds = DatasetEphy([xr_sniff_MI], y='sniffs', times='times', roi='chan_list')
    # define a workflow of mutual information
    wf = WfMi(mi_type='cd', inference='ffx')
    # run the workflow
    mi, pv = wf.fit(ds, n_perm=200, n_jobs=10, random_state=0)














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
        









