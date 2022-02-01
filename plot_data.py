
import os
import neo
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

import mne
import scipy.fftpack
import scipy.signal


debug = False




#path_data, trc_folder = 'D:\LPPR_CMO_PROJECT\Lyon\Data\iEEG', 'LYONNEURO_2019_CAPp'
def return_raw(path_data, trc_folder):

    os.chdir(os.path.join(path_data,trc_folder))

    # identify number of trc file
    trc_file_names = glob.glob('*.TRC')

    # extract file one by one
    data_whole = []
    chan_list_whole = []
    srate_whole = []
    events_name_whole = []
    events_time_whole = []
    #file_i, file_name = 0, trc_file_names[0]
    for file_i, file_name in enumerate(trc_file_names):

        # current file
        print(file_name)

        # extract segment with neo
        reader = neo.MicromedIO(filename=file_name)
        seg = reader.read_segment()
        print('len seg : ' + str(len(seg.analogsignals)))
        
        # extract data
        data_whole_file = []
        chan_list_whole_file = []
        srate_whole_file = []
        events_name_file = []
        events_time_file = []
        #anasig = seg.analogsignals[2]
        for seg_i, anasig in enumerate(seg.analogsignals):
            
            chan_list_whole_file.append(anasig.array_annotations['channel_names'].tolist()) # extract chan
            data_whole_file.append(anasig[:, :].magnitude.transpose()) # extract data
            srate_whole_file.append(int(anasig.sampling_rate.rescale('Hz').magnitude.tolist())) # extract srate

        if srate_whole_file != [srate_whole_file[0] for i in range(len(srate_whole_file))] :
            print('srate different in segments')
            exit()
        else :
            srate_file = srate_whole_file[0]

        # concatenate data
        for seg_i in range(len(data_whole_file)):
            if seg_i == 0 :
                data_file = data_whole_file[seg_i]
                chan_list_file = chan_list_whole_file[seg_i]
            else :
                data_file = np.concatenate((data_file,data_whole_file[seg_i]), axis=0)
                [chan_list_file.append(chan_list_whole_file[seg_i][i]) for i in range(np.size(chan_list_whole_file[seg_i]))]


        # event
        if len(seg.events[0].magnitude) == 0 : # when its VS recorded
            events_name_file = ['999', '999']
            events_time_file = [0, len(data_file[0,:])]
        else : # for other sessions
            #event_i = 0
            for event_i in range(len(seg.events[0])):
                events_name_file.append(seg.events[0].labels[event_i])
                events_time_file.append(int(seg.events[0].times[event_i].magnitude * srate_file))

        # fill containers
        data_whole.append(data_file)
        chan_list_whole.append(chan_list_file)
        srate_whole.append(srate_file)
        events_name_whole.append(events_name_file)
        events_time_whole.append(events_time_file)

    # verif
    #file_i = 1
    #data = data_whole[file_i]
    #chan_list = chan_list_whole[file_i]
    #events_time = events_time_whole[file_i]
    #srate = srate_whole[file_i]

    #chan_name = 'p19+'
    #chan_i = chan_list.index(chan_name)
    #file_stop = (np.size(data,1)/srate)/60
    #start = 0 *60*srate 
    #stop = int( file_stop *60*srate )
    #plt.plot(data[chan_i,start:stop])
    #plt.vlines( np.array(events_time)[(np.array(events_time) > start) & (np.array(events_time) < stop)], ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]))
    #plt.show()

    # concatenate 
    data = data_whole[0]
    chan_list = chan_list_whole[0]
    events_name = events_name_whole[0]
    events_time = events_time_whole[0]
    srate = srate_whole[0]

    if len(trc_file_names) > 1 :
        #trc_i = 0
        for trc_i in range(len(trc_file_names)): 

            if trc_i == 0 :
                len_trc = np.size(data_whole[trc_i],1)
                continue
            else:
                    
                data = np.concatenate((data,data_whole[trc_i]), axis=1)

                [events_name.append(events_name_whole[trc_i][i]) for i in range(len(events_name_whole[trc_i]))]
                [events_time.append(events_time_whole[trc_i][i] + len_trc) for i in range(len(events_time_whole[trc_i]))]

                if chan_list != chan_list_whole[trc_i]:
                    print('not the same chan list')
                    exit()

                if srate != srate_whole[trc_i]:
                    print('not the same srate')
                    exit()

                len_trc += np.size(data_whole[trc_i],1)

    rmv = []
    rmv_name = []
    for i, chan_name in enumerate(chan_list):
        if chan_name.startswith("aa") :
            rmv.append(i)
            rmv_name.append(chan_name)
        else:
            continue


    chan_list_rmv = chan_list
    chan_suppr = 0
    for rmv_i in rmv:
        data = np.delete(data, rmv_i-chan_suppr, 0)
        chan_list_rmv = np.delete(chan_list_rmv, rmv_i-chan_suppr, 0)
        chan_suppr += 1


    # events
    events = np.zeros((np.size(events_time),3))
    i = 0
    for en, et in zip(events_name, events_time):
        events[i,0] = et
        events[i,1] = 0
        events[i,2] = en
        i += 1

    info = mne.create_info(chan_list_rmv.tolist(), sfreq=srate)
    raw = mne.io.RawArray(data, info)



    return raw, events





if __name__== '__main__':
# Extract data

    path_data = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Data\\iEEG\\raw_data'

    trc_folder = 'CAPp'
    #trc_folder = 'CHEe'
    #trc_folder = 'MAZm'
    #trc_folder = 'MUGa'
    #trc_folder = 'GOBc'

    raw, events = return_raw(path_data, trc_folder)

    start = 0
    duration = 300
    n_channels = 10

    raw.plot(events=events, duration=duration, start=start, n_channels=n_channels)









