# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:34:52 2020

@author: Acer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 06:20:50 2020

@author: anand
"""

# the libraries we need
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
import scipy.io
import timeit
import pickle 

start = timeit.default_timer()

AMD=loadmat('FmatctB3020.mat')
X=AMD["Fmatct_lng"][:,:4]
y=AMD["Fmatct_lng"][:,4]
size = len(X)
#trainsize = round(size*0.7)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
XR=X[idx]
yR=y[idx]
# separating data into training and test
#X= train.drop('Cover_Type', axis=1) # cover_type is our target feature, which has 7 classes
#y= train['Cover_Type']
#X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.25)
# xgbp = pickle.load(open("xgbp_shrt_tuned.pkl","rb"))
# # lgbp = pickle.load(open("lgbP.pkl","rb"))
# lgbp2 = pickle.load(open("lgbp_mae.pkl","rb"))
# # [lgbp,xgbp] = pickle.load(open("tuned_params_lng_nov22.pkl","rb"))

[xgbp,ttrials]=pickle.load(open("xgb_params_anal500.pkl","rb"))

xgb_U_lng = XGBRegressor(random_state=24, n_jobs=-1,**xgbp)
xgb_U_lng.set_params(n_estimators=500)

# first, initialize the classificators
rf= RandomForestRegressor(random_state=24,n_jobs=-1,n_estimators=100)

# xgb = XGBRegressor(random_state=24,verbose=-1,n_jobs=-1,**xgbp)
# xgb.set_params(n_estimators=200,learning_rate=0.1)

# lgb = LGBMRegressor(random_state=24, n_jobs=-1,**lgbp2)
# lgb.set_params(n_estimators=200,learning_rate=0.1)


cbr= CatBoostRegressor(verbose=0,random_state=24)

# now, create a list with the objects 
models= [xgb_U_lng]
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
y_pred=[]
trainind=[]
testind=[]
y_act=[]
y_trainpred=[]
y_trainprederr=[]
scores=[]


for model in models:
    scores.append(cross_validate(model, XR, yR, cv=cv, n_jobs=-1, scoring=('r2', 'neg_mean_squared_error','neg_mean_absolute_error'), return_train_score=True))
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

asx=[]
for ll in scores:
    asx.append(np.mean(np.array([vv for ff,vv in ll.items()]),1))

fnl=np.array(asx)   
fnl[:,[4,5]]=np.sqrt(np.abs(fnl[:,[4,5]]))
fnl[:,[6,7]]=(np.abs(fnl[:,[6,7]]))
           
# data = {'scores_B3020' : scores}
# m4p.savemat('scores_B3020.mat', data)   

print(fnl)
# for model in models:
#     for ii,[train_index, test_index] in enumerate(cv.split(XR)):
#     # then predict on the test set
#           X_train, X_test = XR[train_index], XR[test_index]
#           y_train, y_test = yR[train_index], yR[test_index]
#           # X_testn=np.concatenate((X_test,Xspcl))
#           model.fit(X_train,y_train)
#           y_trainprederr.append(y_train-(model.predict(X_train)))
#           y_pred.append((model.predict(X_test)))
#           y_act.append(y_test)
#           trainind.append(train_index)
#           testind.append(test_index)
#           stop = timeit.default_timer()
#           print('Time: ', stop - start)
          
# scipy.io.savemat('errmat_BG.mat', dict(predtrainerr=y_trainprederr,predtrain=y_trainpred,prd=y_pred,act=y_act,tsstind=testind,trrind=trainind))  
  

