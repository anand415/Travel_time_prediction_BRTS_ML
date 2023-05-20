# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 06:44:09 2020

@author: anand
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:37:58 2020

@author: anand
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 02:08:16 2020

@author: anand
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat

import joblib
#from memory_profiler import profile
import timeit

start = timeit.default_timer()

# boston = load_boston()
#X = boston["data"]
#Y = boston["target"]
#@profile
#def something_to_profile():
AMD=loadmat('FmatctB3020.mat')
X=AMD["Fmatct_lng"][:,:4]
Y=AMD["Fmatct_lng"][:,4]
size = len(X)
trainsize = round(size*0.7)
idx = list(range(size))
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
rf_U_lng = RandomForestRegressor(n_jobs=-1,random_state=24,n_estimators=150)
rf_U_lng.fit(X[idx], Y[idx])
# joblib.dump(rf_U_lng,'rf_U_lng', compress=True)
stop = timeit.default_timer()
print('Time: ', stop - start) 
print(rf_U_lng)
#%%
dvdU=loadmat('dvdU.mat')    
dvdU=dvdU['dvdU']
tstU=[]
td=300
for dd in dvdU:
    for ss in range(2,7):
        for tt in range(1,18*3600,td):
                tstU.append([dd[0],dd[1],tt+21600,ss])
                
tstmat=np.array(tstU) 
yyt25Uintr=rf_U_lng.predict(tstmat)
ss=[tstmat,np.array(yyt25Uintr).reshape(-1,1)]
zs=np.hstack(ss)
bn=int((18*3600*5)/td)

AAs=zs[11*bn:12*bn,:]
AAs1=AAs[AAs[:,3]==1,:]
AAs2=AAs[AAs[:,3]==2,:]
AAs3=AAs[AAs[:,3]==3,:]
AAs4=AAs[AAs[:,3]==4,:]
AAs5=AAs[AAs[:,3]==5,:]
AAs6=AAs[AAs[:,3]==6,:]
AAs7=AAs[AAs[:,3]==7,:]

# fig, ax = plt.subplots()
# plt.title(str([n_est,LR]))
plt.plot((AAs1[:,2]/3600),AAs1[:,4],'.', color='C1')
plt.plot((AAs2[:,2]/3600),AAs2[:,4],'.', color='C2')
plt.plot((AAs3[:,2]/3600),AAs3[:,4],'.', color='C3')
plt.plot((AAs4[:,2]/3600),AAs4[:,4],'.', color='C4')
plt.plot((AAs5[:,2]/3600),AAs5[:,4],'.', color='C5')
plt.plot((AAs6[:,2]/3600),AAs6[:,4],'.', color='C6')
plt.plot((AAs7[:,2]/3600),AAs7[:,4],'.', color='C7')
plt.gca().set_ylim([0, 80])
# ax.set(ylim =(0, 80),  
#         autoscale_on = False)
#%%
td=60
tstU=[]
yy=[]
shiv=range(300,660)
for dd in shiv:
    tstU=[]
    for ss in range(2,7):
        for tt in range(12*3600,15*3600,td):
                tstU.append([dd,dd+1,tt+21600,ss])
    tstmat=np.array(tstU) 
    yy.append(np.array(rf_U_lng.predict(tstmat)))
savemat("shiv.mat",dict(yy=yy))
