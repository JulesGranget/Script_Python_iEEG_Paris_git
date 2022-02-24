

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config import *
from n0bis_analysis_functions import *

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

