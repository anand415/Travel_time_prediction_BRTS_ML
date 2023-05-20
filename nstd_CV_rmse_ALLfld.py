# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 03:00:36 2020

@author: anand
"""
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse import quick_hyperopt
import sys
from xgboost import XGBRegressor
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 
from lightgbm import LGBMRegressor
from tqdm import tqdm
import pandas as pd
import timeit

start = timeit.default_timer()
  
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
cv_outer = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

# y_prederr=[]
# trainind=[]
# testind=[]
# y_actts=[]
# y_acttr=[]
# lgbp_fld_param=[]
# kk=0
# for train_index, test_index in cv_outer.split(XR, YR):
#     start = timeit.default_timer()
#     X_train, X_test = XR[train_index], XR[test_index]
#     y_train, y_test = YR[train_index], YR[test_index]
#     lgb_params_lng,ttrials=pickle.load(,open("lgb_nest_6000_fine{}.pkl".format(kk),"rb"))
#       # print(train_index[20:30])
#     lgbp_fld_param.append(lgb_params_lng)
#     # print(lgbp.keys())
#     del lgb_params_lng['metric']
#     del lgb_params_lng['objective']
#     lgb = LGBMRegressor(verbose=-1,random_state=24, n_jobs=-1,**lgb_params_lng)
#     mdl = lgb.fit(X_train, y_train)

#     y_pred_tst = mdl.predict(X_test)
#     y_pred_trn = mdl.predict(X_train)
#     y_actts.append(y_test)
#     y_acttr.append(y_test)
#     trainind.append(train_index)
#     testind.append(test_index)
#     stop = timeit.default_timer()
#     print(stop-start)

# pickle.dump([y_pred_tst,y_pred_trn,y_actts,y_acttr,trainind,testind],open("lgb_6000_allfolds.pkl","wb"))
# # 




y_prederr=[]
trainind=[]
testind=[]
y_actts=[]
y_acttr=[]
y_predts=[]
y_predtr=[]

xgbp_fld_param=[]
kk=0
for train_index, test_index in cv_outer.split(XR, YR):
    start = timeit.default_timer()
    X_train, X_test = XR[train_index], XR[test_index]
    y_train, y_test = YR[train_index], YR[test_index]
    kk=kk+1
    xgb_params_lng,ttrials=pickle.load(open("xgb_nest_2k_fine{}.pkl".format(kk),"rb"))
      # print(train_index[20:30])
    xgbp_fld_param.append(xgb_params_lng)
    # print(xgbp.keys())
    del xgb_params_lng['eval_metric']
    del xgb_params_lng['objective']
    xgb = XGBRegressor(random_state=24, n_jobs=-1,**xgb_params_lng)
    
    xgb.set_params(n_estimators=2000,colsample_bytree=1,colsample_bynode=1,
                     colsample_bylevel= 1,
                     subsample=1)
    
    mdl = xgb.fit(X_train, y_train)

    y_pred_tst = mdl.predict(X_test)
    y_pred_trn = mdl.predict(X_train)
    y_actts.append(y_test)
    y_acttr.append(y_train)
    y_predts.append(y_pred_tst)
    y_predtr.append(y_pred_trn)
    trainind.append(train_index)
    testind.append(test_index)
    stop = timeit.default_timer()
    print(stop-start)
    
# pickle.dump([y_predts,y_predtr,y_actts,y_acttr,trainind,testind],open("xgb_2000_allfolds.pkl","wb"))
