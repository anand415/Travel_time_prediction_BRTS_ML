# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:55:45 2020

@author: anand
"""
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import RepeatedKFold
import mat4py as m4p
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
# from skgarden import RandomForestQuantileRegressor
import joblib
import timeit
import scipy.io

start = timeit.default_timer()

AMD=loadmat('FmatctN.mat')
X=AMD["Fmatct_lng"][:,:4]
y=AMD["Fmatct_lng"][:,4]

ttmtt=[257016,259756,261493,264774,266384]

size = len(X)
#trainsize = round(size*0.7)
idx = list(range(size))
#shuffle the data
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)

XR=X[idx]
yR=y[idx]
# X1=np.delete(X,ttmtt,axis=0)
# y1=np.delete(y,ttmtt,axis=0)


# Xspcl=X[ttmtt]
# yspcl=y[ttmtt[1]].reshape(-1,1)

# idx2 = list(range(size-1))
# np.random.shuffle(idx2)

# separating data into training and test
#X= train.drop('Cover_Type', axis=1) # cover_type is our target feature, which has 7 classes
#y= train['Cover_Type']
#X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.25)


# # first, initialize the classificators
tree= DecisionTreeRegressor(random_state=24) # using the random state for reproducibility
bg= BaggingRegressor(random_state=24, n_jobs=-1)
##knn= KNeighborsRegressor()
#svm= SVC(random_state=24)
GBr = GradientBoostingRegressor(random_state=24)


rf= RandomForestRegressor(random_state=24,n_estimators=250)
rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')


hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)


xgb= XGBRegressor(random_state=24,n_jobs=-1,max_depth=6,n_estimators=100, objective='reg:squarederror')
lgb = LGBMRegressor(random_state=24, n_jobs=-1,n_estimators=100,max_depth=6)
cbr= CatBoostRegressor(verbose=0,random_state=24)
#svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
neigh = KNeighborsRegressor(n_neighbors=1)
# rfqr = RandomForestQuantileRegressor(n_estimators=100,
    # random_state=0, n_jobs=-1)

# # now, create a list with the objects 
models= [rf]
#models= [neigh,tree, xgb,bg, lgb,cbr,rf]
scores=[]
y_pred=[]
trainind=[]
testind=[]
y_act=[]
y_trainpred=[]
y_trainprederr=[]
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)



filenameL=[]
for model in models:
      for ii,[train_index, test_index] in enumerate(cv.split(XR)):
    # then predict on the test set
            X_train, X_test = XR[train_index], XR[test_index]
            y_train, y_test = yR[train_index], yR[test_index]
            # X_testn=np.concatenate((X_test,Xspcl))
            model.fit(X_train, y_train) # fit the model
            A=(type(model).__name__)
            res = [char for char in type(model).__name__ if char.isupper()] 
            mn=''.join(res)
#            filename=mn+str(ii)
#            joblib.dump(model, filename)  
#            filenameL.append(filename)
            y_trainprederr.append(y_train-model.predict(X_train))
            y_trainpred.append(model.predict(X_train))
            y_pred.append(model.predict(X_test))
            y_act.append(y_test)
            trainind.append(train_index)
            testind.append(test_index)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            
# scipy.io.savemat('testTR_err.mat', dict(predtrain=y_trainpred,prd=y_pred,act=y_act,tsstind=testind,trrind=trainind))

for model in models:
    # for train_index, test_index in cv.split(XR):
            # then predict on the test set
          scores.append(cross_validate(model, XR, yR, cv=cv, n_jobs=-1, scoring=('r2', 'neg_mean_squared_error','neg_mean_absolute_error'), return_train_score=True))
          
          stop = timeit.default_timer()

          print('Time: ', stop - start) 
          
print(np.mean(abs(scores[0]['test_neg_mean_absolute_error'])),np.sqrt(np.mean(abs(scores[0]['test_neg_mean_squared_error']))))

train_test
          