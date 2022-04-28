

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from soupsieve import select

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False


########################
######## DRAFT ########
########################





def zscore(x):

    _x = ( x - np.mean(x) ) / np.std(x)

    return _x

data_ieeg, chan_list_ieeg, data_aux, chan_list_aux, srate

times = np.arange(0,data_ieeg.shape[1])/srate

select = [1]

plt.plot(times, zscore(data_aux[0,:])+1)
for select_i in select:
    plt.plot(times, zscore(data_ieeg[select_i,:]))
plt.show()












