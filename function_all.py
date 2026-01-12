import pickle
import sys
import numpy as np
import os
import scipy.signal as sg
import wfdb
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
#import#import matplotlib.pyplot
from scipy.signal import resample
import glob
import pywt
import pandas as pd
import neurokit2 as nk
import csv
from collections import Counter
import warnings

import shutil
import posixpath
import pyhrv
import json
from vectorizedsampleentropy import vectsampen as vse
import math
import heartpy as hp
from statsmodels.tsa.ar_model import AutoReg
import pyhrv.tools as tools
import glob
from sklearn.metrics import r2_score
import collections
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.signal
from biosppy.signals import ecg
import gc
import bz2
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
def auto_ar_predict(X, max_lag=50, criterion='aic'):
    X = np.asarray(X).flatten()
    best_score = np.inf
    best_lag = 1

    for lag in range(1, max_lag+1):

            model = AutoReg(X, lags=lag, old_names=False)
            result = model.fit()
            score = result.aic if criterion=='aic' else result.bic
            if score < best_score:
                best_score = score
                best_lag = lag

            continue

    model_final = AutoReg(X, lags=best_lag, old_names=False)
    result_final = model_final.fit()
    predictions = result_final.predict(start=best_lag, end=len(X)-1)

    return {
        'best_lag': best_lag,
        'model': model_final,
        'fitted_model': result_final,
        'predictions': predictions
    }


def normal_ecg_transfer(data,new_test,fs):
    out1 = ecg.ecg(signal=data, sampling_rate=fs, show=False)
    ECG_seg = out1["templates"]
    ECG_seg_normal=ECG_seg[new_test==0]
    clean_ecg=ECG_seg_normal.ravel()
    data1 = pd.DataFrame(clean_ecg,columns = ['ECG'])
    return data1
	


def f1(y_true, y_pred):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=4)

    tp = K.sum(y_true * y_pred, axis=0)
    tn = K.sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return K.mean(2 * precision * recall / (precision + recall + K.epsilon()))


def categorical_focal_loss(gamma=2):
    """
        Categorical form of focal loss.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
            model.compile(loss=categorical_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])
            model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)
        Notes:
           1. The alpha variable is the class_weight of keras.fit, so in implementation of the focal loss function
           we needn't define this variable.
           2. (important!!!) The output of the loss is the loss value of each training sample, not the total or average
            loss of each batch.
    """

    def focal_loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        return K.sum(-y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred), axis=-1)

    return focal_loss


def load_data(filename="mitdb.pkl.bz2"):
    import pickle

    with bz2.BZ2File(filename, "rb") as f:
        (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = pickle.load(f)

    return (x1_train, x2_train, y_train), (x1_test, x2_test, y_test)


if __name__ == "__main__":
    (x1_train, x2_train, y_train), (x1_test, x2_test, y_test) = load_data()

    x1_train = np.expand_dims(x1_train, axis=-1)
    x1_test = np.expand_dims(x1_test, axis=-1)

    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    model = load_model(os.path.join("", "model_focalloss.h5"),
                       custom_objects={"focal_loss": categorical_focal_loss(gamma=2),
                                       "f1": f1})



	


def wave_to_newtest(data,show):
    record = wfdb.rdrecord(data)
    data0= record.__dict__['p_signal']
    var_name = record.__dict__['sig_name']
    df =  pd.DataFrame(data0,columns=var_name)
    del data0
    del record
    del var_name
    ecg = df['II']
    #ex0=ecg[0:2000]
    # # Find peaks
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
# Automatically process the (raw) ECG signal
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)
# Extract clean ECG and R-peaks location
    rpeaks = info["ECG_R_Peaks"]
    cleaned_ecg = signals["ECG_Clean"]

# Visualize R-peaks in ECG signal
    #plot = nk.events_plot(rpeaks, cleaned_ecg)
    epochs = nk.ecg_segment(cleaned_ecg, rpeaks=None, sampling_rate=fs, show=show)

    
    #generate ECG part output 200 for each ECG
    x1_train_update = pd.DataFrame()

# ECG part
    for i in range(len(epochs)):
        test0 = ECG_data_extend(epochs[str(i+1)])
        test0 = np.reshape(test0, (1, 200)) 
        x1_train_update = x1_train_update.append(pd.DataFrame(data = test0), ignore_index=True)
        
    
#  RR part    
# RR to ms
    rri_0 = np.diff(rpeaks)*0.008*1000
    avg_rri = np.mean(rri_0)


#generate rr part   output 4 for each ECG
    x2_update = pd.DataFrame()
    for index in range(len(rpeaks)):

        if index == 0:
            pre_rri = rri_0[index]  
            post_rri = rri_0[index] 
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])
        
        elif index == len(rpeaks) - 1:
            pre_rri = rri_0[index - 1]
            post_rri = rri_0[(index-1)] 
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])
        
        else:
            pre_rri = rri_0[index - 1]
            post_rri = rri_0[index]
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])

        r0 = np.array([pre_rri - avg_rri, post_rri - avg_rri, ratio_rri, local_rri - avg_rri])
        r1 = np.reshape(r0, (1, 4))
        x2_update = x2_update.append(pd.DataFrame(data = r1), ignore_index=True)

    # transform
    x1_new = np.expand_dims(x1_train_update, axis=-1)
    scaler = RobustScaler()
    x2_new = scaler.fit_transform(x2_update)
    
    return x1_new,x2_new



def wave_out_mean_rr_sd(data):
    record = wfdb.rdrecord(data)
    data0= record.__dict__['p_signal']
    var_name = record.__dict__['sig_name']
    df =  pd.DataFrame(data0,columns=var_name)
    ecg = df['II']
    del record
    del data0
    del var_name
    # # Find peaks
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
# Automatically process the (raw) ECG signal
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)
# Extract clean ECG and R-peaks location
    rpeaks = info["ECG_R_Peaks"]
    
#  RR part    
# RR to ms
    rri_0 = np.diff(rpeaks)*0.008
    avg_rri = np.mean(rri_0)
    std_rri = np.std(rri_0)
    return avg_rri,std_rri
	
	

def ECG_data_extend(data):
    # input data 
    a0 = data['Signal']
    #==========function========================
    # Return y by giving the lienar equaltion and the x
    def getY(x, lineEQ):
        a = lineEQ[0]
        b = lineEQ[1]
        return a * x + b
    
    
    

# Get the linear equalition from p1 and p2
    def getLineEqu(p1, p2):
    # AX=B
        A = np.array([[p1[0], 1],[p2[0],1]])
        B = np.array([p1[1], p2[1]])
    
    # X = [a, b]^t
    # y = ax + b
        a, b = np.linalg.solve(A, B)
    #print('y = ax + b. a = ', a,  ' b = ', b)
        return a, b
    
    def find_mid(part0):
# usage
        p1 = [0, part1[0]]
        p2 = [1, part1[1]]
        a, b = getLineEqu(p1, p2)
# verification
        y = getY(0.5, [a, b])
#     print('y = ', y)
        return y
    #===loop  update data ===============================
# each one mid point 
    imd=[]
#p = 0
    for i in range(round(len(a0)-1)):
    #print(len(a0.iloc[(i):(i+2), ])) 
    #p=p+1
        part0 = a0.iloc[(i):(i+2), ]
        part1 = part0.to_numpy()

    #print(part0)
    #print(find_mid(part1))
        imd.append(find_mid(part1))
#print(a0.iloc[(i):(i+2), ])
# imd.append() 
    a1=a0.to_numpy()
# np.arange(1, 100, step = 2)
    imd_ECG=np.insert(a1,np.arange(1, len(a0), step = 1),imd,axis=None)
# imd_ECG
    if len(imd_ECG)==200:
        finalupdate =  imd_ECG
    elif len(imd_ECG) > 200:
        fina_ecg = imd_ECG[0:200]
        finalupdate =  fina_ecg
    else:
        r0 = 200-len(imd_ECG)
        imp = np.repeat(imd_ECG[len(imd_ECG)-1], r0, axis=0)
        finalupdate = np.concatenate((imd_ECG, imp))
    return finalupdate
	




# setting  HRV functuon        
def estimate_shannon_entropy(dna_sequence):
            m = len(dna_sequence)
            bases = collections.Counter([tmp_base for tmp_base in dna_sequence])
 
            shannon_entropy_value = 0
            for base in bases:
        # number of residues
                n_i = bases[base]
        # n_i (# residues type i) / M (# residues in column)
                p_i = n_i / float(m)
                entropy_i = p_i * (math.log(p_i, 2))
                shannon_entropy_value += entropy_i
 
            return shannon_entropy_value * -1
			

def d1(v1,v2):
            """                                                                                                     
            d0 is Nominal approach:                                                                                 
            multiply/add in a loop                                                                                  
            """
            out = 0
            for k in range(len(v1)):
                out += math.sqrt(v1[k] * v2[k])
            return out


def poincare_sd2(rr):
    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)

    m = np.mean(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)
    
    
    return sd2


def rr_normal_class(class_result,rr):
    filter_ind =[]
    for i in range(0, (len(class_result)-1), 1):
    #---------
#     print(a0[i])
#     print(a0[i+1])
#         print(i)
        if (class_result[i] ==0 and class_result[i+1]==0):
            l0 = 0
        else:
            l0 = 1
        filter_ind.append(l0)
        filter_ind_final = filter_ind
        
    filter_ind_final = np.array(filter_ind_final)
    ind0=np.where(filter_ind_final == 0)
    normal_rr=np.array(rr)[ind0]
    return normal_rr


#  update funciton-normal
def hrvtransform2_only_normal_ECG(hrdata1,new_test,fs,hrv1,settings_time,settings_welch,settings_ar,settings_lomb,settings_nonlinear,max_lag = 1):
        #plt.pause(1) 
        #plt.close('all') 
        hrdata1 = hrdata1[~pd.isnull(hrdata1)]
        hrdata1 = np.array(hrdata1)  
        out1 = ecg.ecg(signal=hrdata1, sampling_rate=fs, show=False) 
# Extract clean ECG and R-peaks location
        rpeaks = out1["rpeaks"]
        rri_0 = np.diff(rpeaks)*0.008*1000
        rr_normal=rr_normal_class(new_test,rri_0)

#         working_data, measures = hp.process(hrdata1, fs)
#hp.plotter(working_data, measures)
        total_time_proportion=sum(rr_normal)/sum(rri_0)
        number_proportion=len(rr_normal)/len(rri_0)
        nni = np.array(rr_normal)
 
        
        
        ##------------------------------------------
        hr = 60/(nni/1000)
#  Mean.rate
        meanrate= np.mean(hr)
    
    

# Poincar..SD2
        sd2=poincare_sd2(nni)


   # Compute the pyHRV parameters
        results = pyhrv.hrv(nni=nni,
                       kwargs_time=settings_time,
                       kwargs_welch=settings_welch,
                       kwargs_ar=settings_ar,
                       kwargs_lomb=settings_lomb,
                       kwargs_nonlinear=settings_nonlinear)





#DFA.Alpha.1
        DFA_Alpha1 = results['dfa_alpha1']





#LF.HF.ratio.LombScargle
        ratio=results['lomb_ratio']

        nni0=np.array(nni)/1000

        nni_diff=np.diff(nni0)
        nni_rmfirst=nni0[1:] 

# aFdP
# RR allan factor  distance

        aFdP = np.var(nni_diff)/(2*np.mean(nni_rmfirst) ) -1



# fFdP
# RR fano factor distance
        fano_rr = np.var(nni0)/np.mean(nni0)
        fFdP =fano_rr-1

        # ax = pd.plotting.autocorrelation_plot(nni0)
        # c0=ax.lines[5].get_data()[1]
        # arr = np.array(c0)

        df = pd.DataFrame(data=nni0)
        df.columns =['rr']
        df['time'] =np.cumsum(df['rr'])
        columns_titles = ["time","rr"]
        df=df.reindex(columns=columns_titles)
        acf_vals = acf(nni0, nlags=50)
        lag = np.argmax(acf_vals[1:]) 
        if lag == 0: 
            lag = 1
        df['stationary'] = df['rr'].diff(lag)



#create datasets
        X = df['stationary'].dropna()
# the autoregression model
        ar_result = auto_ar_predict(X,max_lag = max_lag)
        best_lag =ar_result["best_lag"]
        # best_lag = 1
        model = AutoReg(X, lags=best_lag)
        model_fitted = model.fit()
        predictions = model_fitted.predict(start=best_lag, end=len(X))

        r2 = r2_score(df['stationary'].tail(len(predictions)), predictions)
        rmse = sqrt(mean_squared_error(df['stationary'].tail(len(predictions)), predictions))

# Aerr
        ARerr = rmse 

# QSE


        L = nni
        r = 0.2*np.std(L)
        m = 1
        QSE=vse.qse(L, m, r) 

        shannEn=estimate_shannon_entropy(nni0)

        se=vse.sampen(L, m, r) 
        m=3
        KLPE = 1-se/math.log2(math.factorial(m))


# for histSI  Similarity index of the distributions  update @ 20201123
# 2 d(n)=r(n)-r(n-1)
        dn=np.diff(nni)
        n00=len(dn)
        cutN=round((len(dn)-1)/2)

        if n00 >= 259:   
            d_block1 = dn[(len(nni)-258):(len(nni)-130)]
            d_block2 = dn[(len(nni)-129):len(nni)]
            ironman_ser = pd.Series(d_block1)
            b1range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            ironman_ser = pd.Series(d_block2)
            b2range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            comb=pd.concat([b1range, b2range], axis=0)
            min_value=round(comb.min()/10, 0)
            max_value=round(comb.max()/10, 0)
            step = ((max_value-min_value)*10)/30  # 30
            bin0=np.append(np.arange((min_value*10), (max_value*10), step = round(step)), (max_value*10))



##----------------------------------------------------------
#　指定區間   間格 0.004
##----------------------------------------------------------
            bins = bin0 #[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
            score_cut_block1 = pd.cut(d_block1, bins)
            score_cut_block2 = pd.cut(d_block2, bins)
            prob_block1 = (pd.value_counts(score_cut_block1)/len(score_cut_block1))
            prob_block2 = (pd.value_counts(score_cut_block2)/len(score_cut_block2))


            prob_block1_d=prob_block1.to_frame()
            prob_block2_d=prob_block2.to_frame()
            comb=pd.merge(prob_block1_d, prob_block2_d, left_index=True, right_index=True)



#histSI
            histSI = d1(prob_block1,prob_block2)*100
    
        else:   
            d_block1 = dn[(len(dn)-2*cutN):(len(dn)-(cutN))]
            d_block2 = dn[(len(dn)-(cutN+1)):len(dn)]
            ironman_ser = pd.Series(d_block1)
            b1range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            ironman_ser = pd.Series(d_block2)
            b2range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            comb=pd.concat([b1range, b2range], axis=0)
            min_value=round(comb.min()/10, 0)
            max_value=round(comb.max()/10, 0)
            step = ((max_value-min_value)*10)/30  # 30
            bin0=np.append(np.arange((min_value*10), (max_value*10), step = round(step)), (max_value*10))



##----------------------------------------------------------
#　指定區間   間格 0.004
##----------------------------------------------------------
            bins = bin0 #[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
            score_cut_block1 = pd.cut(d_block1, bins)
            score_cut_block2 = pd.cut(d_block2, bins)
            prob_block1 = (pd.value_counts(score_cut_block1)/len(score_cut_block1))
            prob_block2 = (pd.value_counts(score_cut_block2)/len(score_cut_block2))


            prob_block1_d=prob_block1.to_frame()
            prob_block2_d=prob_block2.to_frame()
            comb=pd.merge(prob_block1_d, prob_block2_d, left_index=True, right_index=True)



#histSI
            histSI = d1(prob_block1,prob_block2)*100

        
#======================================================================================================================
    
        ratio=float(ratio)
        DFA_Alpha1=float(DFA_Alpha1)
        sd2=sd2/1000
   


        hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,QSE,meanrate,sd2,KLPE,shannEn,ratio,histSI,total_time_proportion,
        number_proportion])

        #hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,QSE,meanrate,sd2,KLPE,shannEn,ratio,histSI])

        hrv0 = pd.DataFrame(hrvvar)
#hrv0
        hrv0=hrv0.transpose()

        #hrv1=hrv1.append(hrv0)
        hrv1=pd.concat([hrv1,hrv0], ignore_index=True)
        hrv1.columns=['aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1', 'QSE', 'Mean.rate',
       'Poincar..SD2', 'shannEn', 'LF.HF.ratio.LombScargle', 'KLPE', 'histSI','total_time_proportion',
        'number_proportion']
        #print(hrv1)    
        #plt.close()
        
        plt.close('all')
        
        df1 = pd.DataFrame(hrv1)
        a0 = df1.reset_index(drop=True)
        a0 = pd.DataFrame(a0)
        finalhrv = a0.drop([0])
        for i in finalhrv.columns:
                 finalhrv[i] = pd.to_numeric(finalhrv[i])
        return finalhrv


def hrvtransform1(hrdata1,fs,hrv1,settings_time,settings_welch,settings_ar,settings_lomb,settings_nonlinear,max_lag = 1):

        hrdata1 = hrdata1[~pd.isnull(hrdata1)]
        hrdata1 = np.array(hrdata1)  
        working_data, measures = hp.process(hrdata1, fs)
#hp.plotter(working_data, measures)
        total_time_proportion=sum(working_data['RR_list_cor'])/sum(working_data['RR_list'])
        number_proportion=len(working_data['RR_list_cor'])/len(working_data['RR_list'])


#  Mean.rate
        meanrate=measures['bpm']

# Poincar..SD2
        sd2=measures['sd2']

        nni=working_data['RR_list_cor']
   # Compute the pyHRV parameters
        results = pyhrv.hrv(nni=nni,
                       kwargs_time=settings_time,
                       kwargs_welch=settings_welch,
                       kwargs_ar=settings_ar,
                       kwargs_lomb=settings_lomb,
                       kwargs_nonlinear=settings_nonlinear)





#DFA.Alpha.1
        DFA_Alpha1 = results['dfa_alpha1']





#LF.HF.ratio.LombScargle
        ratio=results['lomb_ratio']

        nni0=np.array(nni)/1000

        nni_diff=np.diff(nni0)
        nni_rmfirst=nni0[1:] 

# aFdP
# RR allan factor  distance

        aFdP = np.var(nni_diff)/(2*np.mean(nni_rmfirst) ) -1



# fFdP
# RR fano factor distance
        fano_rr = np.var(nni0)/np.mean(nni0)
        fFdP =fano_rr-1

        # ax = pd.plotting.autocorrelation_plot(nni0)
        # c0=ax.lines[5].get_data()[1]
        # arr = np.array(c0)

        df = pd.DataFrame(data=nni0)
        df.columns =['rr']
        df['time'] =np.cumsum(df['rr'])
        columns_titles = ["time","rr"]
        df=df.reindex(columns=columns_titles)
        acf_vals = acf(nni0, nlags=50)
        lag = np.argmax(acf_vals[1:]) 
        if lag == 0: 
            lag = 1
        df['stationary'] = df['rr'].diff(lag)



#create datasets
        X = df['stationary'].dropna()
# the autoregression model
        ar_result = auto_ar_predict(X,max_lag = max_lag)
        best_lag =ar_result["best_lag"]
        # best_lag = 1
        model = AutoReg(X, lags=best_lag)
        model_fitted = model.fit()
        predictions = model_fitted.predict(start=best_lag, end=len(X))

        r2 = r2_score(df['stationary'].tail(len(predictions)), predictions)
        rmse = sqrt(mean_squared_error(df['stationary'].tail(len(predictions)), predictions))

# Aerr
        ARerr = rmse 

# QSE


        L = nni
        r = 0.2*np.std(L)
        m = 1
        QSE=vse.qse(L, m, r) 

        shannEn=estimate_shannon_entropy(nni0)

        se=vse.sampen(L, m, r) 
        m=3
        KLPE = 1-se/math.log2(math.factorial(m))


# for histSI  Similarity index of the distributions  update @ 20201123
# 2 d(n)=r(n)-r(n-1)
        dn=np.diff(nni)
        n00=len(dn)
        cutN=round((len(dn)-1)/2)

        if n00 >= 259:   
            d_block1 = dn[(len(nni)-258):(len(nni)-130)]
            d_block2 = dn[(len(nni)-129):len(nni)]
            ironman_ser = pd.Series(d_block1)
            b1range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            ironman_ser = pd.Series(d_block2)
            b2range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            comb=pd.concat([b1range, b2range], axis=0)
            min_value=round(comb.min()/10, 0)
            max_value=round(comb.max()/10, 0)
            step = ((max_value-min_value)*10)/30  # 30
            bin0=np.append(np.arange((min_value*10), (max_value*10), step = round(step)), (max_value*10))



##----------------------------------------------------------
#　指定區間   間格 0.004
##----------------------------------------------------------
            bins = bin0 #[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
            score_cut_block1 = pd.cut(d_block1, bins)
            score_cut_block2 = pd.cut(d_block2, bins)
            prob_block1 = (pd.value_counts(score_cut_block1)/len(score_cut_block1))
            prob_block2 = (pd.value_counts(score_cut_block2)/len(score_cut_block2))


            prob_block1_d=prob_block1.to_frame()
            prob_block2_d=prob_block2.to_frame()
            comb=pd.merge(prob_block1_d, prob_block2_d, left_index=True, right_index=True)



#histSI
            histSI = d1(prob_block1,prob_block2)*100
    
        else:   
            d_block1 = dn[(len(dn)-2*cutN):(len(dn)-(cutN))]
            d_block2 = dn[(len(dn)-(cutN+1)):len(dn)]
            ironman_ser = pd.Series(d_block1)
            b1range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            ironman_ser = pd.Series(d_block2)
            b2range=pd.Series([ironman_ser.min(),ironman_ser.max()])
            comb=pd.concat([b1range, b2range], axis=0)
            min_value=round(comb.min()/10, 0)
            max_value=round(comb.max()/10, 0)
            step = ((max_value-min_value)*10)/30  # 30
            bin0=np.append(np.arange((min_value*10), (max_value*10), step = round(step)), (max_value*10))



##----------------------------------------------------------
#　指定區間   間格 0.004
##----------------------------------------------------------
            bins = bin0 #[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
            score_cut_block1 = pd.cut(d_block1, bins)
            score_cut_block2 = pd.cut(d_block2, bins)
            prob_block1 = (pd.value_counts(score_cut_block1)/len(score_cut_block1))
            prob_block2 = (pd.value_counts(score_cut_block2)/len(score_cut_block2))


            prob_block1_d=prob_block1.to_frame()
            prob_block2_d=prob_block2.to_frame()
            comb=pd.merge(prob_block1_d, prob_block2_d, left_index=True, right_index=True)



#histSI
            histSI = d1(prob_block1,prob_block2)*100

        
#======================================================================================================================
    
        ratio=float(ratio)
        DFA_Alpha1=float(DFA_Alpha1)
        sd2=sd2/1000
   


        hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,QSE,meanrate,sd2,KLPE,shannEn,ratio,histSI,total_time_proportion,
        number_proportion])

        #hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,QSE,meanrate,sd2,KLPE,shannEn,ratio,histSI])

        hrv0 = pd.DataFrame(hrvvar)
#hrv0
        hrv0=hrv0.transpose()

        #hrv1=hrv1.append(hrv0)
        hrv1=pd.concat([hrv1,hrv0], ignore_index=True)
        hrv1.columns=['aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1', 'QSE', 'Mean.rate',
       'Poincar..SD2', 'shannEn', 'LF.HF.ratio.LombScargle', 'KLPE', 'histSI','total_time_proportion',
        'number_proportion']
        #print(hrv1)    
        #plt.close()
        
        plt.close('all')
        
        df1 = pd.DataFrame(hrv1)
        a0 = df1.reset_index(drop=True)
        a0 = pd.DataFrame(a0)
        finalhrv = a0.drop([0])
        for i in finalhrv.columns:
                 finalhrv[i] = pd.to_numeric(finalhrv[i])
        return finalhrv
		

def ECG_data_extend1(data):
    # input data 
    a0 = data
    #==========function========================
    # Return y by giving the lienar equaltion and the x
    def getY(x, lineEQ):
        a = lineEQ[0]
        b = lineEQ[1]
        return a * x + b
    
    
    

# Get the linear equalition from p1 and p2
    def getLineEqu(p1, p2):
    # AX=B
        A = np.array([[p1[0], 1],[p2[0],1]])
        B = np.array([p1[1], p2[1]])
    
    # X = [a, b]^t
    # y = ax + b
        a, b = np.linalg.solve(A, B)
    #print('y = ax + b. a = ', a,  ' b = ', b)
        return a, b
    
    def find_mid(part0):
# usage
        p1 = [0, part1[0]]
        p2 = [1, part1[1]]
        a, b = getLineEqu(p1, p2)
# verification
        y = getY(0.5, [a, b])
#     print('y = ', y)
        return y
    #===loop  update data ===============================
# each one mid point 
    imd=[]
#p = 0
    for i in range(round(len(a0)-1)):
    #print(len(a0.iloc[(i):(i+2), ])) 
    #p=p+1
        part0 = a0[(i):(i+2), ]
        part1 = part0

    #print(part0)
    #print(find_mid(part1))
        imd.append(find_mid(part1))
#print(a0.iloc[(i):(i+2), ])
# imd.append() 
    a1=a0
# np.arange(1, 100, step = 2)
    imd_ECG=np.insert(a1,np.arange(1, len(a0), step = 1),imd,axis=None)
# imd_ECG
    if len(imd_ECG)==200:
        finalupdate =  imd_ECG
    elif len(imd_ECG) > 200:
        fina_ecg = imd_ECG[0:200]
        finalupdate =  fina_ecg
    else:
        r0 = 200-len(imd_ECG)
        imp = np.repeat(imd_ECG[len(imd_ECG)-1], r0, axis=0)
        finalupdate = np.concatenate((imd_ECG, imp))
    return finalupdate
	
	
#  update funciton-normal
def hrvtransform2_only_normal_ECG_filter_hr(hrdata1,new_test,fs,hrv1,settings_time,settings_welch,settings_ar,settings_lomb,settings_nonlinear,max_lag = 1):

        #from biosppy.signals import ecg
        hrdata1 = hrdata1[~pd.isnull(hrdata1)]
        hrdata1 = np.array(hrdata1)  
        out1 = ecg.ecg(signal=hrdata1, sampling_rate=fs, show=False) 
# Extract clean ECG and R-peaks location
        rpeaks = out1["rpeaks"]
        rri_0 = np.diff(rpeaks)*0.008*1000
        rr_normal=rr_normal_class(new_test,rri_0)

#         working_data, measures = hp.process(hrdata1, fs)
#hp.plotter(working_data, measures)


        nni = np.array(rr_normal)
        nni_old = nni
        # nni filter in hr 50~150 
        hr = 60/(nni/1000)
        ind0=np.where((hr >= 50 ) &  (hr <= 150))
#         normal_hr_filter_in50_150=np.array(hr)[ind0]
        
        #update nni  with  50<= HR  <= 150
        nni = np.array(nni)[ind0]
        total_time_proportion=sum(nni)/sum(rri_0)
        number_proportion=len(nni)/len(rri_0)
        
        hr = 60/(nni/1000)
#  Mean.rate
        meanrate= np.mean(hr)

# Poincar..SD2
        sd2=poincare_sd2(nni)


   # Compute the pyHRV parameters
        results = pyhrv.hrv(nni=nni,
                       kwargs_time=settings_time,
                       kwargs_welch=settings_welch,
                       kwargs_ar=settings_ar,
                       kwargs_lomb=settings_lomb,
                       kwargs_nonlinear=settings_nonlinear)





#DFA.Alpha.1
        DFA_Alpha1 = results['dfa_alpha1']





#LF.HF.ratio.LombScargle
        ratio=results['lomb_ratio']

        nni0=np.array(nni)/1000

        nni_diff=np.diff(nni0)
        nni_rmfirst=nni0[1:] 

# aFdP
# RR allan factor  distance

        aFdP = np.var(nni_diff)/(2*np.mean(nni_rmfirst) ) -1



# fFdP
# RR fano factor distance
        fano_rr = np.var(nni0)/np.mean(nni0)
        fFdP =fano_rr-1

        # ax = pd.plotting.autocorrelation_plot(nni0)
        # c0=ax.lines[5].get_data()[1]
        # arr = np.array(c0)
        
        df = pd.DataFrame(data=nni0)
        df.columns =['rr']
        df['time'] =np.cumsum(df['rr'])
        columns_titles = ["time","rr"]
        df=df.reindex(columns=columns_titles)
        acf_vals = acf(nni0, nlags=50)
        lag = np.argmax(acf_vals[1:]) 
        if lag == 0: 
            lag = 1
        df['stationary'] = df['rr'].diff(lag)
        # print(f"lag = {lag}")



#create datasets
        X = df['stationary'].dropna()
        # print(X)
        # print(max(abs(X)))
# the autoregression model
        ar_result = auto_ar_predict(X,max_lag = max_lag)
        best_lag =ar_result["best_lag"]
        # print(f"best_lag1 = {best_lag}")
        # best_lag = 1
        model = AutoReg(X, lags=best_lag)
        model_fitted = model.fit()
        predictions = model_fitted.predict(start=best_lag, end=len(X))

        r2 = r2_score(df['stationary'].tail(len(predictions)), predictions)
        rmse = sqrt(mean_squared_error(df['stationary'].tail(len(predictions)), predictions))
        # print(f"rmse1 = {rmse}")

# Aerr
        ARerr = rmse 


        L = nni
        r = 0.2*np.std(L)
        m = 1
        QSE=vse.qse(L, m, r) 

        shannEn=estimate_shannon_entropy(nni0)




#======================================================================================================================
    
        ratio=float(ratio)
        DFA_Alpha1=float(DFA_Alpha1)
        sd2=sd2/1000
   


        hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,meanrate,sd2,shannEn,ratio])


        hrv0 = pd.DataFrame(hrvvar)
#hrv0
        hrv0=hrv0.transpose()

        #hrv1=hrv1.append(hrv0)
        hrv1=pd.concat([hrv1,hrv0], ignore_index=True)
        hrv1.columns=['aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1', 'Mean.rate','Poincar..SD2', 'shannEn', 'LF.HF.ratio.LombScargle']
        
        
        plt.close('all')
        
        df1 = pd.DataFrame(hrv1)
        a0 = df1.reset_index(drop=True)
        a0 = pd.DataFrame(a0)
        finalhrv = a0.drop([0])
        for i in finalhrv.columns:
                 finalhrv[i] = pd.to_numeric(finalhrv[i])
        return finalhrv
		
		


def wave_to_newtest_input_data_form1(data,show):
    from biosppy.signals import ecg

    out1 = ecg.ecg(signal=data, sampling_rate=fs, show=show)
    # # Find peaks
# Extract clean ECG and R-peaks location
    rpeaks = out1["rpeaks"]
#     bin0=np.append(np.arange(0, 597, step = 3), 597)
#     bin_final = bin0+1
    ECG_seg = out1["templates"]
#     len(bin_final)

# Visualize R-peaks in ECG signal
    #plot = nk.events_plot(rpeaks, cleaned_ecg)

    
    #generate ECG part output 200 for each ECG
    x1_train_update = pd.DataFrame()

# ECG part
    for i in range(len(ECG_seg)):
        test0 = ECG_data_extend1(ECG_seg[(i),])
        test0 = np.reshape(test0, (1, 200)) 
        #x1_train_update = x1_train_update.append(pd.DataFrame(data = test0), ignore_index=True)
        x1_train_update = pd.concat([x1_train_update, pd.DataFrame(data = test0)], ignore_index=True)
        
    
#  RR part    
# RR to ms
    rri_0 = np.diff(rpeaks)*0.008*1000
    avg_rri = np.mean(rri_0)


#generate rr part   output 4 for each ECG
    x2_update = pd.DataFrame()
    for index in range(len(rpeaks)):

        if index == 0:
            pre_rri = rri_0[index]  
            post_rri = rri_0[index] 
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])
        
        elif index == len(rpeaks) - 1:
            pre_rri = rri_0[index - 1]
            post_rri = rri_0[(index-1)] 
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])
        
        else:
            pre_rri = rri_0[index - 1]
            post_rri = rri_0[index]
            ratio_rri = pre_rri / post_rri
            local_rri = np.mean(rri_0[np.maximum(index - 10, 0):index])

        r0 = np.array([pre_rri - avg_rri, post_rri - avg_rri, ratio_rri, local_rri - avg_rri])
        r1 = np.reshape(r0, (1, 4))
        #x2_update = x2_update.append(pd.DataFrame(data = r1), ignore_index=True)
        x2_update = pd.concat([x2_update, pd.DataFrame(data = r1)], ignore_index=True)


    # transform
    x1_new = np.expand_dims(x1_train_update, axis=-1)
    scaler = RobustScaler()
    x2_new = scaler.fit_transform(x2_update)
    
    return x1_new,x2_new


def update_new_bin1(old_bin):
    #old_bin = new_bin0
    min_ind0 = old_bin/7500
    min_ind0[len(min_ind0)-1] = math.floor(min_ind0[len(min_ind0)-1])
    last = min_ind0[len(min_ind0)-1] 
    begin = min_ind0[0]
    ind_min = np.append(np.arange(begin, last, step = 1), last)
    update_bin = ind_min*7500
    return(update_bin)



