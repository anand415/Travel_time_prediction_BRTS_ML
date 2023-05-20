from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import joblib
from xgboost import XGBRegressor
import pickle 
#from memory_profiler import profile
import bz2
import _pickle as cPickle
import timeit
from  scipy.io import savemat,loadmat
import joblib

start = timeit.default_timer()
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
n_est=1000
LR=0.3
# [y_actXA,y_trainprederrXA,trainindXA,testindXA,y_prederrXA,xgb_fld_paramA]=pickle.load(open("xgb_2000_allfolds.pkl","rb"))
xgb_bst_flds=joblib.load(open("XGBR0","rb"))

# xgb_U_lng = XGBRegressor(random_state=24, n_jobs=-1,**xgb_bst_flds)
# xgb_U_lng.set_params(n_estimators=100)
# lltmp=xgb_U_lng.get_params() 
# n_est=lltmp['n_estimators']
# LR=lltmp['learning_rate']  
# # xgb_U_lng=XGBRegressor(**xxgbp)
# # xgb.get_params()
# xgb_U_lng.fit(X[idx], Y[idx])
# print(np.mean(np.abs(YR-xgb_U_lng.predict(XR))))


#%%
stop = timeit.default_timer()
print('Time: ', stop - start) 
# print(xgb_U_lng)
dvdU=loadmat('dvdU.mat')    
dvdU=dvdU['dvdU']
tstU=[]
dvdU[11]=[521,523]
td=300
for dd in dvdU:
    for ss in range(1,8):
        for tt in range(2,18*3600,td):
                tstU.append([dd[0],dd[1],tt+21600,ss])
                
tstmat=np.array(tstU) 
yyt25Uintr=xgb_bst_flds.predict(tstmat)
ss=[tstmat,np.array(yyt25Uintr).reshape(-1,1)]
zs=np.hstack(ss)
bn=int((18*3600*7)/td)

AAs=zs[11*bn:12*bn,:]
AAs1=AAs[AAs[:,3]==1,:]
AAs2=AAs[AAs[:,3]==2,:]
AAs3=AAs[AAs[:,3]==3,:]
AAs4=AAs[AAs[:,3]==4,:]
AAs5=AAs[AAs[:,3]==5,:]
AAs6=AAs[AAs[:,3]==6,:]
AAs7=AAs[AAs[:,3]==7,:]
plt.figure()

# fig, ax = plt.subplots()
plt.title(str([n_est,LR]))
# plt.plot((AAs[:,2]/3600),AAs[:,4],'.', color='C1')


plt.plot((AAs1[:,2]/3600),AAs1[:,4],'.', color='C1')
plt.plot((AAs2[:,2]/3600),AAs2[:,4],'.', color='C2')
plt.plot((AAs3[:,2]/3600),AAs3[:,4],'.', color='C3')
plt.plot((AAs4[:,2]/3600),AAs4[:,4],'.', color='C4')
plt.plot((AAs5[:,2]/3600),AAs5[:,4],'.', color='C5')
plt.plot((AAs6[:,2]/3600),AAs6[:,4],'.', color='C6')
plt.plot((AAs7[:,2]/3600),AAs7[:,4],'.', color='C7')
plt.gca().set_ylim([0, 80])
# savemat('ictxgb.mat',dict(AAs=AAs))
#%%
AAs=zs[13*bn:14*bn,:]
AAs1=AAs[AAs[:,3]==1,:]
AAs2=AAs[AAs[:,3]==2,:]
AAs3=AAs[AAs[:,3]==3,:]
AAs4=AAs[AAs[:,3]==4,:]
AAs5=AAs[AAs[:,3]==5,:]
AAs6=AAs[AAs[:,3]==6,:]
AAs7=AAs[AAs[:,3]==7,:]
plt.figure()

# fig, ax = plt.subplots()
plt.title(str([n_est,LR]))
# plt.plot((AAs[:,2]/3600),AAs[:,4],'.', color='C1')


plt.plot((AAs1[:,2]/3600),AAs1[:,4],'.', color='C1')
plt.plot((AAs2[:,2]/3600),AAs2[:,4],'.', color='C2')
plt.plot((AAs3[:,2]/3600),AAs3[:,4],'.', color='C3')
plt.plot((AAs4[:,2]/3600),AAs4[:,4],'.', color='C4')
plt.plot((AAs5[:,2]/3600),AAs5[:,4],'.', color='C5')
plt.plot((AAs6[:,2]/3600),AAs6[:,4],'.', color='C6')
plt.plot((AAs7[:,2]/3600),AAs7[:,4],'.', color='C7')
plt.gca().set_ylim([0, 80])
savemat('ictxgb14.mat',dict(AAs=AAs))

# #%%
# tstU=[]
# yy=[]
# shiv=range(430,460)
# for dd in shiv:
#     tstU=[]
#     for ss in range(1,8):
#         for tt in range(1,18*3600,td):
#                 tstU.append([dd,dd+1,tt+21600,ss])
#     tstmat=np.array(tstU) 
#     yy.append(np.array(xgb_bst_flds.predict(tstmat)))
# savemat("shiv.mat",dict(yy=yy))

#%%    
# for ss in yy:
          
                
# tstmat=np.array(tstU) 
# yyt25Uintr=xgb_bst_flds.predict(tstmat)
# ss=[tstmat,np.array(yyt25Uintr).reshape(-1,1)]
# zs=np.hstack(ss)
# bn=int((18*3600*7)/td)

# AAs=zs[11*bn:12*bn,:]
