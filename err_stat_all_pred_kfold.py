# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 20:47:47 2021

@author: Satish
"""
#%% clear the variables
from IPython import get_ipython;   
get_ipython().magic('reset -f')

#%% necessary packages 
import pickle as pk
# from scipy.io import loadmat
import numpy as np
from scipy.io import savemat
# from quick_hyperoptt_rmse_nbr import quick_hyperopt
import sys
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 
import timeit
if not sys.warnoptions:
    warnings.simplefilter("ignore")

[sxd,allmod_param]=pk.load(open("L_U_24x7_param_NBR1000.pkl","rb"))
data_L_U_24x7=pk.load(open("data_L_U_24x7.pkl","rb"))
train_data=data_L_U_24x7[0]
test_data=data_L_U_24x7[1]
test_data=test_data[test_data[:,-1]>20,:]
# temprl_train=pickle.load(open("temprl_train5sec_10min_5conv.pkl","rb"))
evals=[30]
# nbr=[1000]
x_col_sel=[slice(0,-1)]
# cse1=[slice(0,2),[0,2],slice(0,3),slice(0,11,2),slice(0,11)]
cv_outer = KFold(n_splits=5, random_state=42,shuffle=True)
xgb_params_set=[]
allerrstat_all=[]
#%%
y_pred_train=[]
y_pred_test=[]
y_act_train=[]
y_act_test=[]
# for kk,ww in enumerate(x_col_sel):   
x=train_data[:,:-1]
y=train_data[:,-1]
r = np.random.RandomState(3)
idx = list(range(len(x)))
#shuffle the data
train_idx=r.shuffle(idx)
#shuffle the data
xr,yr=[x[idx], y[idx]]
params=allmod_param[0][0]
del params['eval_metric']
del params['objective']
#%% training xgb model with k-fold nested validation
mdl= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=1000, **params)
# mdl = mdl.fit(xr, yr)
# print(params)
trn_idx_kfld=[]
tst_idx_kfld=[]
y_act_tst_kfld=[]
y_act_trn_kfld=[]
y_pred_tst_kfld=[]
y_pred_trn_kfld=[]
allfolderrstat=[]
for zz,[train_index, test_index] in enumerate(cv_outer.split(xr, yr)):
    start = timeit.default_timer()
    x_trn_kfld, x_tst_kfld = xr[train_index], xr[test_index]
    y_trn_kfld, y_tst_kfld = yr[train_index], yr[test_index]
    # print([kk,zz])
    mdl = mdl.fit(x_trn_kfld, y_trn_kfld)
    y_pred_tst_kfld.append(mdl.predict(x_tst_kfld))
    y_pred_trn_kfld.append(mdl.predict(x_trn_kfld))
    y_act_tst_kfld.append(y_tst_kfld)
    y_act_trn_kfld.append(y_trn_kfld)
    trn_idx_kfld.append(train_index)
    tst_idx_kfld.append(test_index)
    stop = timeit.default_timer()
    # print(stop-start)
    errstat=[y_pred_trn_kfld,y_pred_tst_kfld,y_act_trn_kfld,y_act_tst_kfld,tst_idx_kfld,trn_idx_kfld]
    allfolderrstat.append(errstat)
    #%% test and train errors calculation with the trained xgb model
    #train data 24 days
x_trn_full=xr
y_trn_full=yr
#test data preparation
x=test_data[:,:-1]
y=test_data[:,-1]
r = np.random.RandomState(3)
idx = list(range(len(x)))
#shuffle the data
test_idx=r.shuffle(idx)
#shuffle the data
xr,yr=[x[idx], y[idx]]
x_tst_full=xr
y_tst_full=yr
y_pred_train.append(mdl.predict(x_trn_full))
y_pred_test.append(mdl.predict(x_tst_full))
y_act_train.append(y_trn_full)
y_act_test.append(y_tst_full)
errstat1=[y_pred_train,y_pred_test,y_act_train,y_act_test,train_idx,test_idx]
# allfolderrstat.append(errstat)
allerrstat_all.append(errstat) 
pk.dump(allfolderrstat,open('allerrstat_all_kfold.pkl',"wb"))
pk.dump(allerrstat_all,open('allerrstat_all.pkl',"wb"))
# savemat("allerrstat_all.mat", np.array(allerrstat_all))
#%% Plot predicted vs actual only for the test data
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_act_test, y_pred_test, label= "stars", color= "green", 
            marker= "*", s=30)
plt.figure()
plt.scatter(y_act_train, y_pred_train, label= "stars", color= "red", 
            marker= "*", s=30)

# fig, ax = plt.subplots(2,1, figsize=(20,12))
# ax[0].plot(y_act_test, y_pred_test, label= "stars", color= "green", 
#             marker= "*", s=30)
# ax[0].set_title("Actual vs Predicted for test data")
# ax[0].set_xlabel("actual_test")
# ax[0].set_ylabel("Predicted_test")
# ax[1].plot(y_act_train, y_pred_train, label= "stars", color= "red", 
#             marker= "*", s=30)
# ax[1].set_title("Actual vs Predicted for train data")
# ax[1].set_xlabel("Actual_train")
# ax[1].set_ylabel("Predicted_train")
