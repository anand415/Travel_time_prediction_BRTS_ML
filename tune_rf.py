# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 00:43:13 2020

@author: anand
"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from  sklearn.preprocessing  import StandardScaler
import numpy as np
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
import pickle
# import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate,cross_val_score
import timeit
import pickle 
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp, tpe, Trials, STATUS_OK,rand
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error

AMD=loadmat('FmatctB3020.mat')
X=AMD["Fmatct_lng"][:,:4]
Y=AMD["Fmatct_lng"][:,4]
size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
XR=X[idx]
YR=Y[idx]
test_size = int(0.2 * len(YR))
np.random.seed(13)
indices = np.random.permutation(len(XR))
X_train = XR[indices[:-test_size]]
y_train = YR[indices[:-test_size]]
X_test = XR[indices[-test_size:]]
y_test = YR[indices[-test_size:]]

seed=4
def objective(params):
    # scores=[]
    # est=int(params['n_estimators'])
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])
    mf=params['max_features']
    model=RandomForestRegressor(n_estimators=500,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,max_features=mf)
    score=np.sqrt(-np.mean((cross_val_score(model, XR, YR, cv=5, n_jobs=-1, scoring=('neg_mean_squared_error')))))
    # asx=[]
    # for ll in scores:
    #     asx.append(np.mean(np.array([vv for ff,vv in ll.items()]),1))
    
    # fnl=np.array(asx)  
    # score=(np.abs(fnl[:,[2]]))
    return score
    # model.fit(X_train,y_train)
    # pred=model.predict(X_test)
    # score=mean_squared_error(y_test,pred)
    # print(score.dtype)
    # return score

def optimize(trial):
    max_featuresslist=['auto', 'sqrt', 'log2']
    params={
           'max_depth':hp.uniform('max_depth',2,30),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6),
           'max_features':hp.choice('max_features',max_featuresslist)
           }
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=50,rstate=np.random.RandomState(seed))
    return best

trial=Trials()
best=optimize(trial)