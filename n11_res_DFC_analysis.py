

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
from frites.conn import conn_covgc, conn_dfc
from frites.workflow import WfMi
from frites.conn import define_windows


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False






########################################
######## PROCESS SNIFF ERP ########
########################################


def process_sniff_ERP():

    print('######## SNIFF ERP ########')

    cond = 'SNIFF'

    respfeatures_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    df_loca = get_loca_df(sujet)

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    band_prep = 'lf'

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find(cond) != -1) and (session_name.find('lf.nc') != -1) :
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
        # plt.show()
        fig.savefig(f'{sujet}_{nchan}_{chan_loca}.jpeg', dpi=150)
        plt.close()








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



def reduce_functionnal_mat(mat, df_sorted):

    index_sorted = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    #### which roi in data
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
            
    indexes_to_compute = {}
    for pair_i, pair_name in enumerate(pairs_possible):    
        pair_A, pair_B = pair_name.split('-')[0], pair_name.split('-')[-1]
        x_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == pair_A]
        y_to_mean = [i for i, roi in enumerate(chan_name_sorted) if roi == pair_B]
        
        coord = []
        for x_i in x_to_mean:
            for y_i in y_to_mean:
                if x_i == y_i:
                    continue
                else:
                    coord.append([x_i, y_i])
                    coord.append([y_i, x_i])

        indexes_to_compute[pair_name] = coord

    #### reduce mat
    mat_reduced = np.zeros((len(roi_in_data), len(roi_in_data) ))

    for roi_i_x, roi_name_x in enumerate(roi_in_data):        
        for roi_i_y, roi_name_y in enumerate(roi_in_data):
            if roi_name_x == roi_name_y:
                continue
            else:
                pair_i = f'{roi_name_x}-{roi_name_y}'
                if (pair_i in indexes_to_compute) == False:
                    pair_i = f'{roi_name_y}-{roi_name_x}'
                pair_i_data = []
                for coord_i in indexes_to_compute[pair_i]:
                    pair_i_data.append(mat[coord_i[0],coord_i[-1]])
                mat_reduced[roi_i_x, roi_i_y] = np.mean(pair_i_data)

    #### verif
    if debug:
        mat_test = np.zeros(( len(chan_name_sorted), len(chan_name_sorted) )) 
        for roi_i, roi_name in enumerate(pairs_possible): 
            i_test = indexes_to_compute[roi_name]
            for pixel in i_test:
                mat_test[pixel[0],pixel[-1]] = 1

        plt.matshow(mat_test)
        plt.show()

    return mat_reduced




def sort_mat(mat, index_sorted):

    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    for i_before_sort_r, i_sort_r in enumerate(index_sorted):
        for i_before_sort_c, i_sort_c in enumerate(index_sorted):
            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    return mat_sorted






def process_dfc_connectivity(cond):


    print(f'######## {cond} FC ########')

    #### load params
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)
    df_loca = get_loca_df(sujet)

    if cond == 'SNIFF':
        sniff_starts = get_sniff_starts(sujet)
        times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
        str_load_file = 'hf.nc'

    if cond == 'AC':
        ac_starts = get_ac_starts(sujet)
        stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
        times = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
        str_load_file = 'AC_session_hf'

    #### load data 
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    load_i = []
    for i, session_name in enumerate(os.listdir()):
        if (session_name.find(str_load_file) != -1) :
            load_i.append(i)
        else:
            continue

    load_list = [os.listdir()[i] for i in load_i]

    if cond == 'SNIFF':
        xr_sniff = xr.open_dataarray(load_list[0])
        xr_sniff = xr_sniff.transpose('sniffs', 'chan_list', 'times')
        xr_sniff['sniffs'] = [0]*len(sniff_starts)
        xr_sniff['chan_list'] = df_loca['ROI'].values

        xr_to_compute = xr_sniff

    if cond == 'AC':
        raw = mne.io.read_raw_fif(load_list[0])
        data = raw.get_data()

        data_AC = np.zeros((len(chan_list_ieeg),len(ac_starts),int(stretch_point_TF_ac)))

        for nchan_i, nchan in enumerate(chan_list_ieeg):

                x = data[nchan_i,:]

                for start_i, start_time in enumerate(ac_starts):

                    t_start = int(start_time + t_start_AC*srate)
                    t_stop = int(start_time + t_stop_AC*srate)

                    data_AC[nchan_i,start_i,:] = x[t_start: t_stop]

        coords = {'chan_list' : chan_list_ieeg, 'AC' : [0]*len(ac_starts), 'time' : times}

        xr_ac = xr.DataArray(data=data_AC, coords=coords)
        xr_ac = xr_ac.transpose('AC', 'chan_list', 'time')
        xr_ac['chan_list'] = df_loca['ROI'].values

        xr_to_compute = xr_ac



    #### params for dfc
    slwin_len = .5    # in sec
    slwin_step = .5*.1  # in sec
    win_sample = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]

    #### compute dfc
    dfc = conn_dfc(xr_to_compute.data, win_sample, times=times, roi=chan_list[:-4], n_jobs=n_core_slurms, verbose=False)

    #### rearange results and mean
    if cond == 'SNIFF':
        dfc['trials'] = xr_to_compute['sniffs'].data
    if cond == 'AC':
        dfc['trials'] = xr_to_compute['AC'].data

    dfc_mean = xr.concat(dfc, 'trials').groupby('trials').mean('trials')

    if debug:
        plt.matshow(dfc_mean[0,:100,:])
        plt.show()


    #### ARANGE ANAT DATA ####
    df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    index_sorted = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    #### identify roi in data
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






    #### SEGMENT PERIODS ####  

    if cond == 'SNIFF':

        #### segment time
        time_pre = -0.5
        time_inspi = 0
        time_post = 0.5
        time_final = 1
        time_list = ['pre', 'inst', 'post']
        times_pre = dfc_mean['times'].values[np.where((dfc_mean['times'].data > time_pre) & (dfc_mean['times'].data < time_inspi))[0]]
        times_inst = dfc_mean['times'].values[np.where((dfc_mean['times'].data > time_inspi) & (dfc_mean['times'].data < time_post))[0]]
        times_post = dfc_mean['times'].values[np.where((dfc_mean['times'].data > time_post) & (dfc_mean['times'].data < time_final))[0]]

        if debug:
            for pair_i in dfc_mean['roi'].data[:10]:
                for time_sel_i, time_sel in enumerate([times_pre, times_inst, times_post]):
                    dfc_mean.sel(trials=0, roi=pair_i, times=time_sel).mean().values
                    plt.plot(dfc_mean.sel(trials=0, roi=pair_i, times=time_sel).values)
                    plt.show()
                

        #### generate matrix for results
        mat_dfc = np.zeros(( 3, len(chan_list[:-4]), len(chan_list[:-4]) ))
        #pair_i = dfc_suj['roi'].values[0]
        for time_sel_i, time_sel in enumerate([times_pre, times_inst, times_post]):
            for pair_i in dfc_mean['roi'].data:
                pair_A, pair_B = pair_i.split('-')
                pair_A_i, pair_B_i = chan_list.index(pair_A), chan_list.index(pair_B)
                mat_dfc[time_sel_i, pair_A_i, pair_B_i] = dfc_mean.sel(trials=0, roi=pair_i, times=time_sel).max().values
                mat_dfc[time_sel_i, pair_B_i, pair_A_i] = dfc_mean.sel(trials=0, roi=pair_i, times=time_sel).max().values


        

        #### prepare sort
        df_sorted = df_loca.sort_values(['lobes', 'ROI'])
        index_sorted = df_sorted.index.values
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


        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))


        #### plot
        fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        for c in range(3):
            ax = axs[c]
            ax.set_title(time_list[c])
            ax.matshow(sort_mat(mat_dfc[c, :, :], index_sorted), vmin=np.min(mat_dfc), vmax=np.max(mat_dfc))
            if c == 0:
                ax.set_yticks(np.arange(len(chan_list[:-4])))
                ax.set_yticklabels(chan_name_sorted_mat)
        # plt.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_raw_mat.png')
        plt.close()

        nrows, ncols = 1, 3
        fig = plt.figure(facecolor='black')
        for chunk_i, chunk in enumerate(time_list):
            mne.viz.plot_connectivity_circle(sort_mat(mat_dfc[chunk_i, :, :], index_sorted), node_names=chan_name_sorted_mat, n_lines=None, 
                                            title=chunk, show=False, padding=7, fig=fig, subplot=(nrows, ncols, chunk_i+1),
                                            vmin=np.min(mat_dfc), vmax=np.max(mat_dfc))
        plt.suptitle('GC_DFC', color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_raw_circle.png')
        plt.close()

        if debug:
            plt.hist(mat_dfc[chunk_i, :, :].reshape(-1), 50, density=True)
            plt.show()

        #### thresh on previous plot
        percentile_thresh = 99
        thresh = np.percentile(mat_dfc.reshape(-1), percentile_thresh)

        mat_dfc_clean = mat_dfc.copy()

        for chunk_i in range(3):
            for x in range(mat_dfc_clean.shape[1]):
                for y in range(mat_dfc_clean.shape[1]):
                    if mat_dfc_clean[chunk_i, x, y] < thresh:
                        mat_dfc_clean[chunk_i, x, y] = 0
                    if mat_dfc_clean[chunk_i, y, x] < thresh:
                        mat_dfc_clean[chunk_i, y, x] = 0


        #### plot with thresh
        fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        for c in range(3):
            ax = axs[c]
            ax.set_title(time_list[c])
            ax.matshow(sort_mat(mat_dfc_clean[c, :, :], index_sorted), vmin=np.min(mat_dfc_clean[c, :, :]), vmax=np.max(mat_dfc_clean[c, :, :]))
            if c == 0:
                ax.set_yticks(np.arange(len(chan_list[:-4])))
                ax.set_yticklabels(chan_name_sorted_mat)
        # plt.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_raw_tresh_mat.png')
        plt.close()

        nrows, ncols = 1, 3
        fig = plt.figure(facecolor='black')
        for chunk_i, chunk in enumerate(time_list):
            mne.viz.plot_connectivity_circle(sort_mat(mat_dfc_clean[chunk_i, :, :], index_sorted), node_names=chan_name_sorted_mat, n_lines=None, 
                                            title=chunk, show=False, padding=7, fig=fig, subplot=(nrows, ncols, chunk_i+1),
                                            vmin=np.min(mat_dfc_clean), vmax=np.max(mat_dfc_clean))
        plt.suptitle('GC_DFC', color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_raw_thresh_circle.png')
        plt.close()


                

        #### mean with reduced ROI 
        #### reduce mat
        mat_dfc_mean = np.zeros(( 3, len(roi_in_data), len(roi_in_data) ))

        for chunk_i in range(3):

            mat_df_sorted = sort_mat(mat_dfc[chunk_i, :, :], index_sorted)

            mat_dfc_mean[chunk_i, :, :] = reduce_functionnal_mat(mat_df_sorted, df_sorted)
                


        #### verif
        if debug:
            plt.matshow(mat_dfc_mean[2])
            plt.show()

            plt.matshow(mat_dfc_mean[2] - mat_dfc_mean[1])
            plt.show()

        #### plot with reduced ROI
        fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        for c in range(3):
            ax = axs[c]
            ax.set_title(time_list[c])
            ax.matshow(mat_dfc_mean[c, :, :], vmin=np.min(mat_dfc_mean), vmax=np.max(mat_dfc_mean))
            if c == 0:
                ax.set_yticks(np.arange(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
        # plt.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_reduced_mat.png')
        plt.close()

        nrows, ncols = 1, 3
        fig = plt.figure(facecolor='black')
        for chunk_i, chunk in enumerate(time_list):
            mne.viz.plot_connectivity_circle(mat_dfc_mean[chunk_i, :, :], node_names=roi_in_data, n_lines=None, 
                                            title=chunk, show=False, padding=7, fig=fig, subplot=(nrows, ncols, chunk_i+1),
                                            vmin=np.min(mat_dfc_mean), vmax=np.max(mat_dfc_mean))
        plt.suptitle('GC_DFC', color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_reduced_circle.png')
        plt.close()


        #### thresh on previous plot
        percentile_thresh = 99
        thresh = np.percentile(mat_dfc_mean.reshape(-1), percentile_thresh)

        mat_dfc_clean = mat_dfc_mean.copy()

        for chunk_i in range(3):
            for x in range(mat_dfc_clean.shape[1]):
                for y in range(mat_dfc_clean.shape[1]):
                    if mat_dfc_clean[chunk_i, x, y] < thresh:
                        mat_dfc_clean[chunk_i, x, y] = 0
                    if mat_dfc_clean[chunk_i, y, x] < thresh:
                        mat_dfc_clean[chunk_i, y, x] = 0


        #### plot with thresh
        fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        for c in range(3):
            ax = axs[c]
            ax.set_title(time_list[c])
            ax.matshow(mat_dfc_clean[c, :, :], vmin=np.min(mat_dfc_clean), vmax=np.max(mat_dfc_clean))
            if c == 0:
                ax.set_yticks(np.arange(len(roi_in_data)))
                ax.set_yticklabels(roi_in_data)
        # plt.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_reduced_thresh_mat.png')
        plt.close()

        nrows, ncols = 1, 3
        fig = plt.figure(facecolor='black')
        for chunk_i, chunk in enumerate(time_list):
            mne.viz.plot_connectivity_circle(mat_dfc_clean[chunk_i, :, :], node_names=roi_in_data, n_lines=None, 
                                            title=chunk, show=False, padding=7, fig=fig, subplot=(nrows, ncols, chunk_i+1),
                                            vmin=np.min(mat_dfc_clean), vmax=np.max(mat_dfc_clean))
        plt.suptitle('GC_DFC', color='w')
        fig.set_figheight(10)
        fig.set_figwidth(12)
        # fig.show()
        fig.savefig(f'{sujet}_{cond}_GCMI_reduced_thresh_circle.png')
        plt.close()




    #### ERP CONNECTIVITY ####
    #### generate mat results

    mat_dfc_time = np.zeros(( len(pairs_possible), dfc_mean.shape[-1] ))

    #### fill mat
    name_modified = np.array([])
    count_pairs = np.zeros(( len(pairs_possible) ))
    for pair_i in dfc_mean['roi'].data:
        pair_A, pair_B = pair_i.split('-')
        pair_A_name, pair_B_name = df_loca['ROI'][df_loca['name'] == pair_A].values[0], df_loca['ROI'][df_loca['name'] == pair_B].values[0]
        pair_name_i = f'{pair_A_name}-{pair_B_name}'
        name_modified = np.append(name_modified, pair_name_i)
    
    for pair_name_i, pair_name in enumerate(pairs_possible):
        pair_name_inv = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
        mask = (name_modified == pair_name) | (name_modified == pair_name_inv)
        count_pairs[pair_name_i] = int(np.sum(mask))
        mat_dfc_time[pair_name_i,:] = np.mean(dfc_mean.data[0,mask,:], axis=0)




    #### plot and save fig
    os.chdir(os.path.join(path_results, sujet, 'FC', 'DFC', cond))

    if cond == 'AC':
        times = np.linspace(t_start_AC, t_stop_AC, mat_dfc_time.shape[1])
    if cond == 'SNIFF':
        times = np.linspace(t_start_SNIFF, t_stop_SNIFF, mat_dfc_time.shape[1])


    for pair_i, pair_name in enumerate(pairs_possible):

        fig = plt.figure()
        plt.plot(times, mat_dfc_time[pair_i,:])
        plt.ylim(mat_dfc_time[:,:].min(), mat_dfc_time[:,:].max())
        plt.vlines(0, ymin=mat_dfc_time[:,:].min() ,ymax=mat_dfc_time[:,:].max(), color='r')
        plt.title(f'{pair_name} count : {count_pairs[pair_i]}')
        # plt.show()
        
        fig.savefig((f'{sujet}_{cond}_GCMI_{pair_name}.png'))

        plt.close()
    
    #### all plot
    if debug:

        for pair_i, pair_name in enumerate(pairs_possible):

            plt.plot(times, mat_dfc_time[pair_i,:], label=pair_name)

        plt.vlines(0, ymin=mat_dfc_time[:,:].min() ,ymax=mat_dfc_time[:,:].max(), color='r')
        plt.legend()
        plt.show()










########################
######## TEST ########
########################



    if debug:


        dfc_search = dfc_mean.data.reshape(dfc_mean.shape[1], dfc_mean.shape[2])
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
            dfc_mean[0,pair_i,:].plot()
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



            
            


            #### MUTUAL INFORMATION
            # define an electrophysiological dataset
            xr_sniff_MI = xr.open_dataarray(load_list[0])
            xr_sniff_MI = xr_sniff_MI.transpose('sniffs', 'chan_list', 'times')

            mne_sniff_info = mne.create_info(list(xr_sniff_MI['chan_list'].data), srate, ch_types=['seeg']*len(xr_sniff_MI['chan_list']))
            mne_sniff = mne.EpochsArray(xr_sniff_MI.data, info=mne_sniff_info)





            xr_sniff_MI['sniffs'] = np.array(['free']*len(xr_sniff['sniffs'].data), dtype='object')
            se = SubjectEphy(mne_sniff)
            ds = DatasetEphy([xr_sniff_MI], y='sniffs', times='times', roi='chan_list')
            # define a workflow of mutual information
            wf = WfMi(mi_type='cd', inference='ffx')
            # run the workflow
            mi, pv = wf.fit(ds, n_perm=200, n_jobs=10, random_state=0)





################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    process_sniff_ERP()

    #cond = 'AC'
    for cond in ['SNIFF', 'AC']:
        process_dfc_connectivity(cond)
        









