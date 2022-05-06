

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





#cond = 'AC'
def process_dfc_connectivity(cond):


    print(f'######## {cond} FC ########')

    #### load params
    df_loca_all = pd.DataFrame(columns=['name', 'ROI', 'lobes'])
    chan_list_ieeg_all = []
    len_sniff_allsujet = []
    len_AC_allsujet = []
    
    #sujet_i = sujet_list[1]
    for sujet_i in sujet_list:

        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_i)
        df_loca = get_loca_df(sujet_i)
        df_loca['ROI'] = df_loca['ROI'] + f'_{sujet_i[-4:]}' 

        chan_list_ieeg_all.extend(chan_list_ieeg)
        df_loca_all = pd.concat([df_loca_all, df_loca])

        len_sniff_allsujet.append(len(get_sniff_starts(sujet_i)))
        len_AC_allsujet.append(len(get_ac_starts(sujet_i)))

    df_loca_all.index = range(df_loca_all.index.shape[0])

    #### identify shape to concat for sniff
    shape_sniff = np.min(len_sniff_allsujet)
    shape_ac = np.min(len_AC_allsujet)

    #### load data

    concat_coords = {'chan_list' : np.array(())}
    concat_data = np.array(())

    for sujet_i in sujet_list:

        conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet_i)
        df_loca = get_loca_df(sujet_i)
        df_loca['ROI'] = df_loca['ROI'] + f'_{sujet_i[-4:]}' 

        if cond == 'SNIFF':
            sniff_starts = get_sniff_starts(sujet_i)
            times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
            str_load_file = 'hf.nc'

        if cond == 'AC':
            ac_starts = get_ac_starts(sujet_i)
            stretch_point_TF_ac = int(np.abs(t_start_AC)*srate +  t_stop_AC*srate)
            times = np.linspace(t_start_AC, t_stop_AC, stretch_point_TF_ac)
            str_load_file = 'AC_session_hf'
        
        os.chdir(os.path.join(path_prep, sujet_i, 'sections'))

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

            if xr_sniff.shape[0] != shape_sniff:
                xr_sniff = xr_sniff[:shape_sniff, :, :]

            if sujet_i == sujet_list[0]:
                concat_data = xr_sniff.data
                concat_coords['sniffs'] = xr_sniff['sniffs'].values
                concat_coords['chan_list'] = xr_sniff['chan_list'].values

            else:
                concat_data = np.append(concat_data, xr_sniff.data, axis=1)
                concat_coords['chan_list'] = np.append(concat_coords['chan_list'], xr_sniff['chan_list'].values)



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

            coords = {'chan_list' : chan_list_ieeg, 'AC' : [0]*len(ac_starts), 'times' : times}

            xr_ac = xr.DataArray(data_AC, coords=coords.values(), dims=coords.keys())
            xr_ac = xr_ac.transpose('AC', 'chan_list', 'times')
            xr_ac['chan_list'] = df_loca['ROI'].values

            if xr_ac.shape[0] != shape_ac:
                xr_ac = xr_ac[:shape_ac, :, :]

            if sujet_i == sujet_list[0]:
                concat_data = xr_ac.data
                concat_coords['AC'] = xr_ac['AC'].values
                concat_coords['chan_list'] = xr_ac['chan_list'].values

            else:
                concat_data = np.append(concat_data, xr_ac.data, axis=1)
                concat_coords['chan_list'] = np.append(concat_coords['chan_list'], xr_ac['chan_list'].values)

    concat_coords['times'] = times

    if cond == 'SNIFF':
        desired_order_list = ['sniffs', 'chan_list', 'times']
    if cond == 'AC':
        desired_order_list = ['AC', 'chan_list', 'times']


    concat_coords = {i: concat_coords[i] for i in desired_order_list}

    xr_to_compute_all = xr.DataArray(concat_data, coords=concat_coords.values(), dims=concat_coords.keys())

    #### params for dfc
    slwin_len = .5    # in sec
    slwin_step = .5*.1  # in sec
    win_sample = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]

    #### compute dfc
    dfc = conn_dfc(xr_to_compute_all.data, win_sample, times=times, roi=df_loca_all['ROI'], n_jobs=n_core_slurms, verbose=False)

    #### rearange results and mean
    if cond == 'SNIFF':
        dfc['trials'] = xr_to_compute_all['sniffs'].data
    if cond == 'AC':
        dfc['trials'] = xr_to_compute_all['AC'].data

    del xr_to_compute_all

    dfc_mean = xr.concat(dfc, 'trials').groupby('trials').mean('trials')

    if debug:
        plt.matshow(dfc_mean[0,:100,:])
        plt.show()




    #### ARANGE ANAT DATA ####
    df_sorted = df_loca_all.sort_values(['lobes', 'ROI'])
    index_sorted = df_sorted.index.values
    chan_name_sorted = df_sorted['ROI'].values.tolist()

    #### identify roi in data
    roi_in_data = []
    rep_count = 0
    for i, name_i in enumerate(chan_name_sorted):
        name_i = name_i[:-5]
        if i == 0:
            roi_in_data.append(name_i)
            continue
        else:
            if name_i == chan_name_sorted[i-(rep_count+1)][:-5]:
                rep_count += 1
                continue
            if name_i != chan_name_sorted[i-(rep_count+1)][:-5]:
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
        mat_dfc = np.zeros(( 3, len(chan_list_ieeg_all), len(chan_list_ieeg_all) ))
        #pair_i = dfc_suj['roi'].values[0]
        for time_sel_i, time_sel in enumerate([times_pre, times_inst, times_post]):
            for pair_i in dfc_mean['roi'].data:
                pair_A, pair_B = pair_i.split('-')
                pair_A_i, pair_B_i = chan_list_ieeg_all.index(pair_A), chan_list_ieeg_all.index(pair_B)
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
        os.chdir(os.path.join(path_results, 'allplot', 'FC', 'DFC', cond))


        #### plot
        fig, axs = plt.subplots(ncols=3, figsize=(15,15))
        for c in range(3):
            ax = axs[c]
            ax.set_title(time_list[c])
            ax.matshow(sort_mat(mat_dfc[c, :, :], index_sorted), vmin=np.min(mat_dfc), vmax=np.max(mat_dfc))
            if c == 0:
                ax.set_yticks(np.arange(len(chan_list_ieeg_all)))
                ax.set_yticklabels(chan_name_sorted_mat)
        # plt.show()
        fig.savefig(f'allplot_{cond}_GCMI_raw_mat.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_raw_circle.png')
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
                ax.set_yticks(np.arange(len(chan_list_ieeg_all)))
                ax.set_yticklabels(chan_name_sorted_mat)
        # plt.show()
        fig.savefig(f'allplot_{cond}_GCMI_raw_tresh_mat.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_raw_thresh_circle.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_reduced_mat.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_reduced_circle.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_reduced_thresh_mat.png')
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
        fig.savefig(f'allplot_{cond}_GCMI_reduced_thresh_circle.png')
        plt.close()




    #### ERP CONNECTIVITY ####
    #### generate mat results

    mat_dfc_time = np.zeros(( len(pairs_possible), dfc_mean.shape[-1] ))

    #### fill mat
    name_modified = np.array([])
    count_pairs = np.zeros(( len(pairs_possible) ))
    for pair_i in dfc_mean['roi'].data:
        pair_A, pair_B = pair_i.split('-')
        pair_A_name, pair_B_name = pair_A[:-5], pair_B[:-5]
        pair_name_i = f'{pair_A_name}-{pair_B_name}'
        name_modified = np.append(name_modified, pair_name_i)
    
    #pair_name_i, pair_name = 1, pairs_possible[1]
    for pair_name_i, pair_name in enumerate(pairs_possible):
        pair_name_inv = f"{pair_name.split('-')[-1]}-{pair_name.split('-')[0]}"
        mask = (name_modified == pair_name) | (name_modified == pair_name_inv)
        count_pairs[pair_name_i] = int(np.sum(mask))
        mat_dfc_time[pair_name_i,:] = np.mean(dfc_mean.data[0,mask,:], axis=0)




    #### plot and save fig
    os.chdir(os.path.join(path_results, 'allplot', 'FC', 'DFC', cond))

    if cond == 'AC':
        times = np.linspace(t_start_AC, t_stop_AC, mat_dfc_time.shape[1])
    if cond == 'SNIFF':
        times = np.linspace(t_start_SNIFF, t_stop_SNIFF, mat_dfc_time.shape[1])

    #pair_i, pair_name = 0, pairs_possible[0]
    for pair_i, pair_name in enumerate(pairs_possible):

        fig = plt.figure()
        plt.plot(times, mat_dfc_time[pair_i,:])
        plt.ylim(mat_dfc_time[:,:].min(), mat_dfc_time[:,:].max())
        plt.vlines(0, ymin=mat_dfc_time[:,:].min() ,ymax=mat_dfc_time[:,:].max(), color='r')
        plt.title(f'{pair_name} count : {count_pairs[pair_i]}')
        # plt.show()
        
        fig.savefig((f'allplot_{cond}_GCMI_{pair_name}.png'))

        plt.close()
    
    #### all plot
    if debug:

        for pair_i, pair_name in enumerate(pairs_possible):

            plt.plot(times, mat_dfc_time[pair_i,:], label=pair_name)

        plt.vlines(0, ymin=mat_dfc_time[:,:].min() ,ymax=mat_dfc_time[:,:].max(), color='r')
        plt.legend()
        plt.show()







################################
######## DFC PLI ISPC ########
################################

def get_all_pairs(sujet_i):

    prms = get_params(sujet_i)
    df_loca = get_loca_df(sujet_i)

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

    name_modified = np.array([])
    for pair_i in pairs_to_compute:
        pair_A, pair_B = pair_i.split('-')
        pair_A_name, pair_B_name = df_loca['ROI'][df_loca['name'] == pair_A].values[0], df_loca['ROI'][df_loca['name'] == pair_B].values[0]
        pair_name_i = f'{pair_A_name}-{pair_B_name}'
        name_modified = np.append(name_modified, pair_name_i)

    #### identify number of each pair
    count_pair = {}
    #pair_possible_i = pairs_possible[0]
    for pair_possible_i in pairs_possible:
        name_modified_inv = f"{pair_possible_i.split('-')[1]}-{pair_possible_i.split('-')[0]}"
        count_i = np.sum((name_modified == pair_possible_i) | (name_modified == name_modified_inv))
        count_pair[pair_possible_i] = count_i

    return count_pair




#mat_type_i, band_i = 0, 0
def reduce_dfc_pair(data_allsujet, mat_type_i, band_i, allplot_pairs, allplot_pairs_sujet, times):

    #### generate res mat            
    pairs_uniques, uniq_idx, counts = np.unique(allplot_pairs,return_index=True,return_counts=True)
    allplot_dfc = np.zeros(( pairs_uniques.shape[0], len(times) ))

    #### get all cond
    count_pair_sujet = {}
    for sujet_i in sujet_list:
        count_pair_sujet[sujet_i] = get_all_pairs(sujet_i)

    #pair_i, pair_name = np.where(pairs_uniques == 'insula post-parahippocampique')[0][0], 'insula post-parahippocampique' 
    for pair_i, pair_name in enumerate(pairs_uniques):
        counts_pair = counts[pair_i]
        if counts_pair > 1:
            #### identify which pair need to be mean
            pair_all_i = np.where(allplot_pairs == pair_name)[0]

            sujet_to_mean = [allplot_pairs_sujet[i] for i in pair_all_i]
            count_pair = [count_pair_sujet[i][pair_name] for i in sujet_to_mean]

            coord_in_data = []
            for sujet_to_mean_i in sujet_to_mean:
                coord_in_data.append( np.where((allplot_pairs_sujet == sujet_to_mean_i) & (allplot_pairs == pair_name))[0][0] )

            #### ponderate each mean for each subject
            final_division = 0
            pair_reduced = np.zeros(( times.shape[0] ))
            for i, pair_all_i_i in enumerate(coord_in_data):
                final_division += count_pair[i]
                pair_reduced += data_allsujet[band_i, mat_type_i, pair_all_i_i, :] * count_pair[i]

            pair_reduced /= final_division

            allplot_dfc[pair_i, :] = pair_reduced
            
        else:
            #### identify which pair need to be mean
            pair_all_i = np.where(allplot_pairs == pair_name)[0]
            allplot_dfc[pair_i, :] = data_allsujet[band_i, mat_type_i, pair_all_i, :]

    return allplot_dfc




def get_all_count(pair_list_to_plot):

    pair_list_to_plot_count = {}

    count_pair_sujet = {}
    for sujet_i in sujet_list:
        count_pair_sujet[sujet_i] = get_all_pairs(sujet_i)

    #pair_i, pair_name = 0, pair_list_to_plot[0]
    for pair_i, pair_name in enumerate(pair_list_to_plot):

        sujet_count = 0

        for sujet_i in sujet_list:
            if pair_name in list(count_pair_sujet[sujet_i].keys()):
                sujet_count += count_pair_sujet[sujet_i][pair_name]

        pair_list_to_plot_count[pair_name] = sujet_count

    return pair_list_to_plot_count
    





#cond = 'SNIFF'
def allplot_dfc_pli_ispc_SNIFF_AC(cond):

    #### params
    band_prep = 'hf'
    band_names = list(freq_band_dict_FC[band_prep].keys())

    #### containers
    allplot_pairs = np.array(())
    allplot_pairs_sujet = np.array(())
    
    #sujet_i = sujet_list[2]
    for sujet_i in sujet_list:

        count_pair = get_all_pairs(sujet_i)
        allplot_pairs = np.append(allplot_pairs, list(count_pair.keys()))
        allplot_pairs_sujet = np.append(allplot_pairs_sujet, [sujet_i]*len(list(count_pair.keys())))

        #### load data
        os.chdir(os.path.join(path_precompute, sujet_i, 'FC'))

        #band_i, band = 0, 'l_gamma'
        for band_i, band in enumerate(freq_band_dict_FC[band_prep].keys()):

            #### open data
            load_i = []
            for i, session_name in enumerate(os.listdir()):
                if (session_name.find(f'pli_ispc_{band}_{cond}') != -1):
                    load_i.append(i)
                else:
                    continue

            load_list = [os.listdir()[i] for i in load_i]

            xr_load = xr.open_dataarray(load_list[0])

            times = xr_load['times'].data 

            #### generate mat
            if sujet_i == sujet_list[0] and band == band_names[0]:
                data_allsujet = np.zeros(( len(band_names), xr_load.shape[0], xr_load.shape[1], xr_load.shape[2] ))
                data_allsujet[band_i, :, :, :] = xr_load.data
            elif sujet_i == sujet_list[0] and band != band_names[0]:
                data_allsujet[band_i, :, :, :] = xr_load.data
            elif sujet_i != sujet_list[0] and band == band_names[0]:
                data_to_concat = np.zeros(( len(band_names), xr_load.shape[0], xr_load.shape[1], xr_load.shape[2] ))
                data_to_concat[band_i, :, :, :] = xr_load.data
            elif sujet_i != sujet_list[0] and band != band_names[0]:
                data_to_concat[band_i, :, :, :] = xr_load.data

        if sujet_i != sujet_list[0]:
            data_allsujet = np.concatenate([data_allsujet, data_to_concat], axis=2)

    
    #### reduce mat
    pair_list_to_plot = np.unique(allplot_pairs)
    data_to_export = np.zeros(( data_allsujet.shape[0], data_allsujet.shape[1], len(pair_list_to_plot), data_allsujet.shape[3] ))
    for band_i, band in enumerate(band_names):
        for mat_type_i, mat_type in enumerate(['pli', 'ispc']):
            data_to_export[mat_type_i, band_i, :, :] = reduce_dfc_pair(data_allsujet, mat_type_i, band_i, allplot_pairs, allplot_pairs_sujet, times)

    #### plot res
    pair_list_to_plot_count = get_all_count(pair_list_to_plot)

    os.chdir(os.path.join(path_results, 'allplot', 'FC', 'DFC', cond))

    for mat_type_i, mat_type in enumerate(['pli', 'ispc']):

        #pair_i, pair_name = 0, pair_list_to_plot[0]
        for pair_i, pair_name in enumerate(pair_list_to_plot):

            fig, axs = plt.subplots(nrows=len(band_names))

            for band_i, band in enumerate(band_names):

                ax = axs[band_i]

                ax.plot(times, data_to_export[mat_type_i, band_i, pair_i, :])
                ax.vlines(0, ymin=data_to_export[mat_type_i, band_i, pair_i, :].min(), ymax=data_to_export[mat_type_i, band_i, pair_i, :].max(), color='r')

                if band_i == 0:
                    ax.set_title(f'{pair_name} {mat_type} count : {pair_list_to_plot_count[pair_name]}')
                
                ax.set_ylabel(band)

            #plt.show()

            fig.set_figheight(15)
            fig.set_figwidth(15)
            fig.savefig(f'allplot_{cond}_{mat_type}_{pair_name}.png')

            plt.close()



        






################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #cond = 'AC'
    for cond in ['SNIFF', 'AC']:
        #process_dfc_connectivity(cond)
        execute_function_in_slurm_bash('n15_res_allplot_fc', 'process_dfc_connectivity', [cond])
        
        #allplot_dfc_pli_ispc_SNIFF_AC(cond)
        execute_function_in_slurm_bash('n15_res_allplot_fc', 'allplot_dfc_pli_ispc_SNIFF_AC', [cond])






