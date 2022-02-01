
#### to run all analysis

import os
from n0_config import *

os.chdir(path_main_workdir)
import n4_precompute_surrogates

os.chdir(path_main_workdir)
import n5_precompute_TF

os.chdir(path_main_workdir)
import n6_power_analysis

os.chdir(path_main_workdir)
import n7_fc_analysis






