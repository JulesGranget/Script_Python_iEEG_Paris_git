

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib
import physio
from mne.time_frequency import psd_array_multitaper

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False




################################
######## PRECOMPUTE TF ########
################################


def compute_sujet_HRV(sujet):

    #### verify if already computed
    os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'HRV'))

    if electrode_recording_type == 'monopolaire':
        if os.path.exists(f'{sujet}_tf_{cond}.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}.npy'):
            print('ALREADY COMPUTED', flush=True)
            return
    if electrode_recording_type == 'bipolaire':
        if os.path.exists(f'{sujet}_tf_{cond}_bi.npy') and os.path.exists(f'{sujet}_tf_raw_{cond}_bi.npy'):
            print('ALREADY COMPUTED', flush=True)
            return

    #### params
    respfeatures_allcond = load_respfeatures(sujet)

    #### select data without aux chan
    #cond = 'AL'
    for cond in ['AL', 'AC']:

        if cond != 'AL':

            #sujet = sujet_list[3]
            for sujet in sujet_list:

                if sujet == 'pat_03146_1608':
                    continue

                os.chdir(os.path.join(path_results, 'allplot', 'allcond', 'HRV'))

                chan_list, chan_list_ieeg = get_chanlist(sujet, electrode_recording_type)

                #### RB RRI
                ecg_rest = load_data(sujet, 'FR_CV', electrode_recording_type)[chan_list.index('ECG'),:]
                respi = load_data(sujet, 'FR_CV', electrode_recording_type)[chan_list.index('ventral'),:]*-1
                respi, resp_cycles = physio.compute_respiration(respi, srate, parameter_preset='human_airflow')
                respi_freq = resp_cycles['cycle_freq'].mean()
                ecg_rest, ecg_rest_peaks = physio.compute_ecg(ecg_rest, srate, parameter_preset='human_ecg')
                rri_rest = np.diff(ecg_rest_peaks['peak_time'])*1e3

                time_vec_rest = ecg_rest_peaks['peak_time'][1:].values
                time_vec_rest_up = np.linspace(time_vec_rest[0], time_vec_rest[-1], int(time_vec_rest[-1]*srate))

                f = scipy.interpolate.interp1d(time_vec_rest, rri_rest, kind='quadratic')
                rri_rest_up = f(time_vec_rest_up)

                ####AL RRI
                data = load_data(sujet, cond, electrode_recording_type)

                data_AL_rri = []
                time_vec_AL = []
                min_max_rri = np.array([])
                min_max_ecg = np.array([])

                for AL_i in range(len(data)+1):

                    if AL_i == len(data):

                        min_max_rri = np.append(min_max_rri, rri_rest.min())
                        min_max_rri = np.append(min_max_rri, rri_rest.max())
                        min_max_ecg = np.append(min_max_ecg, ecg_rest.min())
                        min_max_ecg = np.append(min_max_ecg, ecg_rest.max())

                    else:

                        ecg = data[AL_i][chan_list.index('ECG'),:]
                        ecg, ecg_peaks = physio.compute_ecg(ecg, srate, parameter_preset='human_ecg')
                        rri = np.diff(ecg_peaks['peak_time'])*1e3
                        time_vec = ecg_peaks['peak_time'][1:].values
                        time_vec_up = np.linspace(time_vec[0], time_vec[-1], int(time_vec[-1]*srate))

                        f = scipy.interpolate.interp1d(time_vec, rri, kind='quadratic')
                        rri_up = f(time_vec_up)

                        if debug:

                            plt.plot(ecg)
                            plt.show()

                            plt.plot(time_vec, rri, label='raw')
                            plt.plot(time_vec_up, rri_up, label='up')
                            plt.legend()
                            plt.show()

                        data_AL_rri.append(rri)
                        time_vec_AL.append(time_vec)
                        min_max_rri = np.append(min_max_rri, rri.min())
                        min_max_rri = np.append(min_max_rri, rri.max())
                        min_max_ecg = np.append(min_max_ecg, ecg.min())
                        min_max_ecg = np.append(min_max_ecg, ecg.max())

                min, max = min_max_rri.min(), min_max_rri.max()
                min_ecg, max_ecg = min_max_ecg.min(), min_max_ecg.max()

                if debug:

                    plt.plot(np.arange(respi.shape[0])/srate, respi)
                    plt.vlines(resp_cycles['inspi_time'].values, ymin=respi.min(), ymax=respi.max(), color='r')
                    plt.show()

                    plt.plot(time_vec_rest, rri_rest, label='raw')
                    plt.plot(time_vec_rest_up, rri_rest_up, label='up')
                    plt.legend()
                    plt.show()

                #### AL
                max_length = np.array([_data[chan_list.index('ECG'),:].size for _, _data in enumerate(data)]).max()

                fig, axs = plt.subplots(nrows=len(data)+1, sharex=True, sharey=True)

                plt.suptitle(sujet)

                for AL_i in range(len(data)+1):

                    if AL_i == len(data):

                        time_vec_mask = time_vec_rest < max_length/srate
                        ax = axs[AL_i]
                        ax.plot(time_vec_rest[time_vec_mask], rri_rest[time_vec_mask])
                        ax.set_title(f"RB")
                        ax.set_ylim(min,max)

                    else:

                        ax = axs[AL_i]
                        ax.plot(time_vec_AL[AL_i], data_AL_rri[AL_i])
                        ax.set_title(f"AL_{AL_i+1}")
                        ax.set_ylim(min,max)

                plt.tight_layout()
                # plt.show()

                plt.savefig(f"{sujet}_all_AL.jpeg")

                fig, axs = plt.subplots(nrows=len(data)+1, sharex=True)

                plt.suptitle(sujet)

                for AL_i in range(len(data)+1):

                    if AL_i == len(data):

                        time_vec_mask = np.arange(ecg_rest.size)/srate < max_length/srate
                        ax = axs[AL_i]
                        ax.plot(ecg_rest[time_vec_mask])
                        ax.set_title(f"RB")
                        ax.set_ylim(min_ecg,max_ecg)

                    else:

                        ecg = data[AL_i][chan_list.index('ECG'),:]
                        ecg, ecg_peaks = physio.compute_ecg(ecg, srate, parameter_preset='human_ecg')

                        ax = axs[AL_i]
                        ax.plot(ecg)
                        ax.set_title(f"AL_{AL_i+1}")
                        ax.set_ylim(min_ecg,max_ecg)

                plt.tight_layout()
                # plt.show()

                plt.savefig(f"{sujet}_all_ecg.jpeg")

                

                # Compute multitaper power spectral density
                Pxx, HzPxx = psd_array_multitaper(rri_up, srate, fmin=0.01, fmax=20)
                Pxx_rest, HzPxx_rest = psd_array_multitaper(rri_rest_up, srate, fmin=0.01, fmax=20)
                Pxx_respi, HzPxx_respi = psd_array_multitaper(respi, srate, fmin=0.01, fmax=20)

                nwind = int(10*srate)
                nfft = nwind*10
                noverlap = np.round(nwind/2)
                hannw = scipy.signal.windows.hann(nwind)

                HzPxx_respi, Pxx_respi = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
                HzPxx_respi[np.argmax(Pxx_respi[HzPxx_respi < 20])]
                
                # Plot the results
                plt.figure()
                plt.plot(HzPxx, np.log(Pxx), label='AL')
                plt.plot(HzPxx_rest, np.log(Pxx_rest), label='RB')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power/Frequency (dB/Hz)')
                plt.title('Multitaper Spectral Estimation')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(HzPxx_respi, np.log(Pxx_respi), label='respi')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power/Frequency (dB/Hz)')
                plt.title('Multitaper Spectral Estimation')
                plt.legend()
                plt.show()

                if debug:

                    plt.plot(ecg)
                    plt.show()

                    plt.plot(rri)
                    plt.show()








            else:
                data_AL = load_data(sujet, cond, electrode_recording_type)
                AL_len_list = np.array([data_AL[session_i].shape[-1] for session_i in range(len(data_AL))])
                data = np.zeros(( len(chan_list_ieeg), AL_len_list.sum() ))
                AL_pre, AL_post = 0, AL_len_list[0]
                #AL_i = 1
                for AL_i in range(AL_n):

                    if AL_i != 0:
                        AL_pre, AL_post = AL_pre + AL_len_list[AL_i-1], AL_post + AL_len_list[AL_i]

                    data[:,AL_pre:AL_post] = data_AL[AL_i][:len(chan_list_ieeg),:]

            print('COMPUTE', flush=True)

            #### select wavelet parameters
            wavelets = get_wavelets()

            #### compute
            os.chdir(path_memmap)
            tf_allchan = np.memmap(f'{sujet}_tf_{cond}_precompute_convolutions_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), nfrex, data.shape[1]))

            def compute_tf_convolution_nchan(n_chan):

                print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                x = data[n_chan,:]

                tf = np.zeros((nfrex, x.shape[0]))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

            del data

            #### stretch or chunk
            if cond == 'FR_CV':

                n_cycle_stretch = stretch_data_tf(respfeatures_allcond[cond][0], stretch_point_TF, tf_allchan[0,:,:], srate)[0].shape[0]
                tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), n_cycle_stretch, nfrex, stretch_point_TF))
            
                print('STRETCH_VS', flush=True)
                tf_allband_stretched[:] = compute_stretch_tf(sujet, tf_allchan, cond, respfeatures_allcond, stretch_point_TF, srate, electrode_recording_type)

            if cond == 'AC':

                ac_starts = get_ac_starts(sujet)
                tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), len(ac_starts), nfrex, stretch_point_TF_ac_resample))
                
                print('CHUNK_AC', flush=True)
                tf_allband_stretched[:] = compute_stretch_tf_AC(sujet, tf_allchan, ac_starts, srate, electrode_recording_type)

            if cond == 'SNIFF':
                
                sniff_starts = get_sniff_starts(sujet)
                tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), len(sniff_starts), nfrex, stretch_point_TF_sniff_resampled))

                print('CHUNK_SNIFF', flush=True)
                tf_allband_stretched[:] = compute_stretch_tf_SNIFF(sujet, tf_allchan, sniff_starts, srate, electrode_recording_type)

            if cond == 'AL':

                tf_allband_stretched = np.memmap(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_ieeg), AL_n, nfrex, resampled_points_AL))

                print('CHUNK_AL', flush=True)
                tf_allband_stretched[:] = compute_stretch_tf_AL(sujet, cond, tf_allchan, AL_len_list, srate, electrode_recording_type)

            if debug:

                plt.pcolormesh(np.median(tf_allband_stretched[0,:,:,:], axis=0))
                plt.show()
            
            #### save
            print('SAVE', flush=True)
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if electrode_recording_type == 'monopolaire':
                np.save(f'{sujet}_tf_{cond}.npy', tf_allband_stretched)
            if electrode_recording_type == 'bipolaire':
                np.save(f'{sujet}_tf_{cond}_bi.npy', tf_allband_stretched)
            
            os.chdir(path_memmap)
            try:
                os.remove(f'{sujet}_tf_{cond}_precompute_convolutions_{electrode_recording_type}.dat')
            except:
                pass

            try:
                os.remove(f'{sujet}_{cond}_tf_allband_stretched_{electrode_recording_type}.dat')
            except:
                pass

            try:
                os.remove(f'{sujet}_tf_{cond}_resample_{electrode_recording_type}.dat')
            except:
                pass

            print('done', flush=True)
















################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        



