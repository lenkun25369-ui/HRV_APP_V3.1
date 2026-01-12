#!/usr/bin/env python
# coding: utf-8

# In[4]:
import os
import sys
import types


os.environ["MPLBACKEND"] = "Agg"

# ---- Completely disable biosppy plotting modules ----
fake_plotting = types.ModuleType("biosppy.inter_plotting")
sys.modules["biosppy.inter_plotting"] = fake_plotting
sys.modules["biosppy.inter_plotting.ecg"] = fake_plotting
sys.modules["biosppy.inter_plotting.acc"] = fake_plotting

# ---- Also block tkinter just in case ----
sys.modules["tkinter"] = None
sys.modules["_tkinter"] = None

import pickle

import numpy as np

import scipy.signal as sg
import wfdb
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt
import pandas as pd
import neurokit2 as nk
import csv
from collections import Counter
import warnings
#from IPython.display import display
import shutil
import posixpath
import pyhrv
import json
#from vectorizedsampleentropy import vectsampen as vse
import math
import heartpy as hp
from statsmodels.tsa.ar_model import AR
import pyhrv.tools as tools
import glob
from sklearn.metrics import r2_score
import collections
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.signal
from biosppy.signals import ecg
import gc
gc.enable()
pd.set_option('display.max_colwidth', None)
fs = 125
base_dir = "dataset"
sampling_rate = 360
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']  # non-beat labels
before = 90
after = 110
tol = 0.05
cpuCount = os.cpu_count() 
import psutil
#import#imp
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
import sys
# import rpy2.rinterface
from function_all import *


# In[16]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ECGgrid(data, timelag1=600, timelag2=1200, nrows=40, ncols=40):
    """
    Python version of R ECGgrid function
    data: pandas DataFrame with column 'ECG'
    """

    # --- ensure dataframe ---
    df = pd.DataFrame(data)

    ecg = df['ECG'].values

    # --- standardize ECG (optional, same as R scale) ---
    ecg = StandardScaler().fit_transform(ecg.reshape(-1, 1)).flatten()

    # --- Takens embedding function ---
    def takens_embedding(x, tau):
        n = len(x) - tau
        return np.column_stack((x[:n], x[tau:tau+n]))

    # --- grid counting function ---
    def grid_counting(embedded):
        x = embedded[:, 0]
        y = embedded[:, 1]

        # 2D histogram (grid)
        H, xedges, yedges = np.histogram2d(
            x, y,
            bins=[nrows, ncols]
        )

        # count non-empty cells
        nonzero_cells = np.count_nonzero(H)

        return (nonzero_cells / (nrows * ncols)) * 100

    # --- first grid count ---
    emb1 = takens_embedding(ecg, timelag1)
    grid_count = grid_counting(emb1)

    # --- second grid count ---
    emb2 = takens_embedding(ecg, timelag2)
    grid_count1 = grid_counting(emb2)

    # --- SgridTAU ---
    sgridTAU = grid_count1 - grid_count

    # --- output ---
    return pd.DataFrame({
        "Grid Counting": [grid_count],
        "SgridTAU": [sgridTAU]
    })

ECG_grid_func_r = ECGgrid


# In[6]:


# Define HRV input parameters
# Time Domain Settings
settings_time = {
    'threshold': 50,            # Computation of NNXX/pNNXX with 50 ms threshold -> NN50 & pNN50
    'plot': True,               # If True, plots NNI histogram
    'binsize': 7.8125           # Binsize of the NNI histogram
}

# Frequency Domain Settings
settings_welch = {
    'nfft': 2 ** 12,            # Number of points computed for the FFT result
    'detrend': True,            # If True, detrend NNI series by subtracting the mean NNI
    'window': 'hann'         # Window function used for PSD estimation
}

settings_lomb = {
    'nfft': 2**8,               # Number of points computed for the Lomb PSD
    'ma_size': 5                # Moving average window size
}

settings_ar = {
    'nfft': 2**12,              # Number of points computed for the AR PSD
    'order': 32                 # AR order
}

# Nonlinear Parameter Settings
settings_nonlinear = {
    'short': [4, 16],           # Interval limits of the short term fluctuations
    'long': [17, 64],           # Interval limits of the long term fluctuations
    'dim': 2,                   # Sample entropy embedding dimension
    'tolerance': None           # Tolerance distance for which the vectors to be considered equal (None sets default values)
}


# In[7]:


model = load_model("models/model_focalloss.h5",
        custom_objects={"focal_loss": categorical_focal_loss(gamma=2),"f1": f1})
model


# In[17]:


# input raw ECG data
import pandas as pd
df = pd.read_csv(sys.argv[1])
df


# In[18]:


II_data = df['II']
del  df
hrv_update = hp.preprocessing.scale_data(II_data, lower=0, upper=200)
del II_data
hrv1=pd.DataFrame(['NA','NA','NA','NA','NA','NA','NA','NA'])
h0=hrv1.transpose()
h0.columns =['aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1', 'Mean.rate',
                       'Poincar..SD2', 'shannEn', 'LF.HF.ratio.LombScargle']                     
hrv1=hrv1.transpose()
final1 = hrv_update
v0=0
v1=hrv_update.shape[0]
step0=37500
bin0=np.append(np.arange(v0, v1, step = step0), v1)
index = [0]
new_bin0 = np.delete(bin0, index)
check=[]

for i in new_bin0: 
            
            warnings.filterwarnings("ignore")
            index = np.append(np.arange((i/step0-1)*step0+1, i, step = 1),i)
            index =index.astype(int)-1
            hrdata1 = final1[index]
            hrdata1 = hrdata1[~pd.isnull(hrdata1)]
            hrdata1 = np.array(hrdata1)  
            x1_test,x2_test = wave_to_newtest_input_data_form1(hrdata1,show=False)
            new_test=np.argmax(model.predict([x1_test,x2_test], batch_size=1024, verbose=0), axis=-1)                    
            result = np.array([Counter(new_test)[0]/len(new_test),Counter(new_test)[1]/len(new_test),Counter(new_test)[2]/len(new_test),Counter(new_test)[3]/len(new_test)])
            r1 = np.reshape(result, (1, 4))
            check.append(r1.item(0)) 
            data1 = normal_ecg_transfer(hrdata1,new_test,fs)
            #r_from_pd_df = ro.conversion.py2rpy(data1)
            hrvpart0= hrvtransform2_only_normal_ECG_filter_hr(hrdata1,new_test,fs,hrv1,settings_time,settings_welch,settings_ar,settings_lomb,settings_nonlinear)
            hrvpart = ECG_grid_func_r(data1)
            hrvpart['time_min'] = i/7500
            warnings.filterwarnings("ignore")
            # remove dataframe 2 indices
            hrvpart0.reset_index(drop=True, inplace=True)
            # remove dataframe 2 indices
            hrvpart.reset_index(drop=True, inplace=True)
            hrvpart = pd.concat([hrvpart0, hrvpart], axis=1)
            h0 = pd.concat([h0,hrvpart], ignore_index=True)

h0 = h0.iloc[1:]


# In[21]:


h0.to_csv("h0.csv", index=False)
h0.to_csv(sys.argv[2], index=False)
print(h0.to_json(orient="records"))
# In[19]:


# output 10 variables
h0


# In[20]:


# output 10 variables
h0


# In[ ]:





# In[ ]:





# In[ ]:




