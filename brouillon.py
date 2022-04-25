

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

from n0_config import *
from n0bis_config_analysis_functions import *

debug = False


########################
######## DRAFT ########
########################



band_prep = 'hf'

data_ac = load_data('AC', band_prep=band_prep)
data_fr_cv = load_data('FR_CV', band_prep=band_prep)
data_sniff = load_data('SNIFF', band_prep=band_prep)



nchan = 20
plt.plot(data_fr_cv[nchan,:], label='FR_CV')
plt.plot(data_ac[nchan,:], label='AC')
plt.plot(data_sniff[nchan,:], label='SNIFF')
plt.legend()
plt.show()



df = pd.read_excel('/home/jules/smb4k/CRNLDATA/crnldata/cmo/Etudiants/NBuonviso202201_trigeminal_sna_Anis/Analyses/Test_hrv_time/df_allchunk.xlsx')
df = df.drop(columns=df.columns[0])
df = df[df['chunk'] == 300]
df = df.drop(columns=df.columns[0])
df = df.groupby(['cond', 'sujet']).mean()

df = df.set_index(['sujet', 'cond', 'trial'])

X = df.values
 
ICA = FastICA(n_components=len(df.columns))
ICA_data = ICA.fit_transform(X)

pca = PCA(n_components=len(df.columns))
reduced_data = pca.fit_transform(X)




plt.scatter(ICA_data[:, 0], ICA_data[:, 1], s=2, marker="o", zorder=10, color="steelblue", alpha=0.5)
plt.show()

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=2, marker="o", zorder=10, color="steelblue", alpha=0.5)
plt.show()
 




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












