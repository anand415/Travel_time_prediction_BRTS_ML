# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 00:09:39 2020

@author: anand
"""
from quick_hyperoptt import quick_hyperopt
from skopt.plots import plot_evaluations
import pickle
import matplotlib.pyplot as plt
import numpy as np

# [param,trials]=pickle.load(open("xgb_params_anal2000fn_40.pkl","rb"))

[param,trials]=pickle.load(open("xgb_nest_2000_fine4.pkl","rb"))
# [param,trials]=pickle.load(open("lgb_nest_6000.pkl","rb"))


# lgb_nest_6000

#%%

Aparam=ttrials.vals
keys=list(Aparam)

# plot_evaluations(trials)
AR=ttrials.losses
lsses=AR()
lsses_ct=[i for i in lsses if i <80] 
aparamct=Aparam 
print(np.min(lsses))
# print(param)


# # lsses[np.array(lsses)<120]
# for ss in keys:
#     aparamct[ss]=[Aparam[ss][ii] for ii,jj in enumerate(lsses) if jj <80]
    
    
    
# for ss in keys:
#     plt.figure()
#     plt.title(ss)
#     plt.plot(aparamct[ss],lsses_ct,'.')

for ss in keys:
    plt.figure()
    plt.title(ss)
    plt.plot(Aparam[ss],lsses,'.')
        
    