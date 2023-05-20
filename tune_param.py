# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:17:08 2020

@author: Acer
"""
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse_restrt import quick_hyperopt
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
AMD=loadmat('FmatctB3020.mat')
X=AMD["Fmatct_lng"][:,:4]
Y=AMD["Fmatct_lng"][:,4]

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]
#%%
TrVTs=[60,20,20]
Xhyper=XR[:int(size*0.8)]
Yhyper=YR[:int(size*0.8)]


# lgbm_params = quick_hyperopt(XR, YR, 'lgbm', 1000)
# xgb_params = quick_hyperopt(XR, YR, 'xgb', 1000)
# [xgb_params_lng,ttrials] = quick_hyperopt(XR, YR, 'lgbm', 200,diagnostic=True)
# [cb_params_lng,ttrials] = quick_hyperopt(XR, YR,ttrials, 'cbm', 70,diagnostic=True)
# pickle.dump([cb_params_lng,ttrials],open("cb_nest_500.pkl","wb"))
[xgb_params_lng,ttrials]=pickle.load(open("xgb3_params_anal2000.pkl","rb"))
[cb_params_lng2,ttrials2] = quick_hyperopt(XR, YR,ttrials, 'cb', 60,diagnostic=True)
pickle.dump([cb_params_lng2,ttrials2],open("cb2_nest_500.pkl","wb"))

# X=AMD["Fmatct"][:,:4]
# Y=AMD["Fmatct"][:,4]
# size = len(X)
# R = np.random.RandomState(3)
# idx = list(range(size))
# #shuffle the data
# R.shuffle(idx)
# #shuffle the data
# XR,YR=[X[idx], Y[idx]]
# # lgbm_params = quick_hyperopt(XR, YR, 'lgbm', 1000)
# # xgb_params = quick_hyperopt(XR, YR, 'xgb', 1000)



# # f = open('xgb_tuned.pckl', 'wb')
# # pickle.dump(xgb_params, f)
# # f.close()
