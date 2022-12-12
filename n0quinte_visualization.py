
import numpy as np
import matplotlib.pyplot as plt

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False



################################
######## FUNCTION ########
################################


#nchan, condition, band_prep = [0, 5, 10], 'FR_CV', 'lf'
def vizu(sujet, nchan, condition, band_prep):

    prms = get_params(sujet, electrode_recording_type)

    data = load_data(sujet, condition, electrode_recording_type, band_prep=band_prep)

    time = np.arange(data.shape[1])/prms['srate']

    respi = zscore(data[-3, :])

    for i, nchan_i in enumerate(nchan):
        
        data_plot = zscore(data[nchan_i, :])

        plt.plot(time, data_plot + 3 * i, label=prms['chan_list'][nchan_i])
    
    plt.plot(time, respi + len(nchan) * 3, label='respi')

    plt.title(condition)
    plt.legend()
    plt.show()




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    prms = get_params(sujet, electrode_recording_type)

    vizu(sujet=sujet, nchan=[0, 1, 2, 3], condition='SNIFF', band_prep='lf')

    n_plot = 5
    nblocs = int((len(prms['chan_list'])-3)/n_plot)
    for bloc_i in range(nblocs):
        vizu(sujet=sujet, nchan=np.arange(n_plot)+bloc_i*n_plot, condition='SNIFF', band_prep='lf')



