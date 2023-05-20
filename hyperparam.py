# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:45:47 2020

@author: anand
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import numpy as np
from sklearn.model_selection import RepeatedKFold

import mat4py as m4p
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# from skgarden import RandomForestQuantileRegressor
import joblib
import timeit
import scipy.io


# ----------------------------------------

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

start = timeit.default_timer()

AMD=loadmat('FmatctN.mat')
X=AMD["Fmatct_lng"][:,:4]
y=AMD["Fmatct_lng"][:,4]

ttmtt=[257016,259756,261493,264774,266384]

size = len(X)
#trainsize = round(size*0.7)
idx = list(range(size))
#shuffle the data
np.random.shuffle(idx)

XR=X[idx]
yR=y[idx]


# -----------------------------------------
tree= DecisionTreeRegressor(random_state=24) # using the random state for reproducibility
bg= BaggingRegressor(random_state=24, n_jobs=-1)
GBr = GradientBoostingRegressor(random_state=24)
rfmse= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10)
rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')
hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)
xgb= XGBRegressor(random_state=24,n_jobs=-1,max_depth=6,n_estimators=100, objective='reg:squarederror')
lgb = LGBMRegressor(random_state=24, n_jobs=-1)
cbr= CatBoostRegressor(verbose=0,random_state=24)    

                
params = {
    # "colsample_bytree": uniform(0.7, 0.3),
    # "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.01, 0.09), # default 0.1 
    # "max_depth": randint(6,25), # default 3
    # "max_iter": randint(100, 450), # default 100
    # "subsample": uniform(0.6, 0.4)
}

paramslbg = {
    # "colsample_bytree": uniform(0.7, 0.3),
    # "gamma": uniform(0, 0.5),
    "n_estimators": range(3000,8000,500), # default 0.1 
    "learning_rate":  np.linspace(0.1,0.9,5), # default 0.1 
    # "subsample": uniform(0.6, 0.4)
}


# search = RandomizedSearchCV(lgb, param_distributions=paramslbg, scoring='neg_mean_squared_error', random_state=42, n_iter=35, cv=5, verbose=10, n_jobs=-1)

search =GridSearchCV(lgb, param_grid=paramslbg, scoring='neg_mean_squared_error', cv=5, verbose=10, n_jobs=1)

search.fit(X, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

report_best_scores(search.cv_results_, 10)
