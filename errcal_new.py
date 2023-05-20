# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:19:01 2020

@author: anand
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.io import savemat,loadmat


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

conv=loadmat('conventional_TTP_stats')

conv_trn2=list(np.array([conv['convtrn_RMSE'][0],conv['convtrn_RR2'][0],conv['convtrn_MAE'][0],conv['convtrn_bias'][0],conv['convtrn_std'][0],np.array([47])]))
conv_tst2=np.array([conv['convtst_RMSE'][0],conv['convtst_RR2'][0],conv['convtst_MAE'][0],conv['convtst_bias'][0],conv['convtst_std'][0],np.array([2])])

conv_trn=[]
conv_tst=[]
for ww,dd in zip(conv_trn2,conv_tst2):
    conv_trn.append(ww[0])
    conv_tst.append(dd[0])
    
allerrstat=[]    
allerrstat.append(pickle.load(open("rf_allfolds.pkl","rb")))
allerrstat.append(pickle.load(open("lgb_2000_allfolds.pkl","rb")))
allerrstat.append(pickle.load(open("xgb_2000_allfolds.pkl","rb")))
    
# [y_pred_tst_xgb,y_pred_trn_xgb,y_act_tst_xgb,y_act_trn_xgb,trnind,tstind]=pickle.load(open("xgb_2000_allfolds.pkl","rb"))
# [y_pred_tst_lgb,y_pred_trn_lgb,y_act_tst_lgb,y_act_trn_lgb,trnind,tstind]=pickle.load(open("lgb_2000_allfolds.pkl","rb"))
# [y_pred_tst_rf,y_pred_trn_rf,y_act_tst_rf,y_act_trn_rf,trnind,tstind]=pickle.load(open("rf_allfolds.pkl","rb"))


# act_trn_all=[y_act_trn_rf,y_act_trn_lgb,y_act_trn_xgb]

# pred_trn_all=[y_pred_trn_rf,y_pred_trn_lgb,y_pred_trn_xgb]

# act_tst_all=[y_act_tst_rf,y_act_tst_lgb,y_act_tst_xgb]

# pred_tst_all=[y_pred_tst_rf,y_pred_tst_lgb,y_pred_tst_xgb]

MAE_trn=[]
RMSE_trn=[]
R2_trn=[]
R_trn=[]
bias_trn=[]
std_trn=[]
mape_trn=[]

MAE_tst=[]
RMSE_tst=[]
R_tst=[]
R2_tst=[]
bias_tst=[]
std_tst=[]
act_tst_all=[]
pred_tst_all=[]
mape_tst=[]

# indx_trn_all=[trnind]

# indx_tst_all=[tstind]

for erst in allerrstat:
    [y_pred_tst,y_pred_trn,y_act_tst,y_act_trn,trnind,tstind]=erst
    indx_trn_all=[trnind]
    indx_tst_all=[tstind]
    
    act=np.hstack(np.array(y_act_trn))
    pred=np.hstack(np.array(y_pred_trn))
    RMSE_trn.append(np.sqrt(mean_squared_error(act, pred)))
    MAE_trn.append((mean_absolute_error(act, pred)))
    R_trn.append(np.cov(act,pred) / (np.std(act) * np.std(pred)))
    R2_trn.append(100*(r2_score(act, pred)))    
    bias_trn.append(np.abs((np.mean(act-pred))))
    std_trn.append(np.sqrt(np.var(act-pred)))
    mape_trn.append(mean_absolute_percentage_error(act, pred))

    
    
    
    act=np.hstack(np.array(y_act_tst))
    pred=np.hstack(np.array(y_pred_tst))
    act_tst_all.append(act)
    pred_tst_all.append(pred)
  
    RMSE_tst.append(np.sqrt(mean_squared_error(act, pred)))
    MAE_tst.append((mean_absolute_error(act, pred)))
    R2_tst.append(100*(r2_score(act, pred)))
    R_tst.append(np.cov(act,pred) / (np.std(act) * np.std(pred)))
    bias_tst.append(np.abs((np.mean(act-pred))))
    std_tst.append(np.sqrt(np.var(act-pred)))
    mape_tst.append(mean_absolute_percentage_error(act, pred))


# RMSE_trn=[]
# for yy,ss in zip(act_trn_all,pred_trn_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     RMSE_trn.append(np.sqrt(mean_squared_error(act, pred)))
    
    
# MAE_trn=[]
# for yy,ss in zip(act_trn_all,pred_trn_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     MAE_trn.append((mean_absolute_error(act, pred)))
    
    
# R2_trn=[]
# for yy,ss in zip(act_trn_all,pred_trn_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     R2_trn.append(100*(r2_score(act, pred)))    

# bias_trn=[]
# for yy,ss in zip(act_trn_all,pred_trn_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     bias_trn.append(np.abs((np.mean(act-pred))))

# std_trn=[]
# for yy,ss in zip(act_trn_all,pred_trn_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     std_trn.append(np.sqrt(np.var(act-pred)))
 
    

# RMSE_tst=[]
# for yy,ss in zip(act_tst_all,pred_tst_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     RMSE_tst.append(np.sqrt(mean_squared_error(act, pred)))
    
    
# MAE_tst=[]
# for yy,ss in zip(act_tst_all,pred_tst_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     MAE_tst.append((mean_absolute_error(act, pred)))
    
    
# R2_tst=[]
# for yy,ss in zip(act_tst_all,pred_tst_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     R2_tst.append(100*(r2_score(act, pred)))    

# bias_tst=[]
# for yy,ss in zip(act_tst_all,pred_tst_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     bias_tst.append(np.abs((np.mean(act-pred))))

# std_tst=[]
# for yy,ss in zip(act_tst_all,pred_tst_all):
#     act=np.hstack(np.array(yy))
#     pred=np.hstack(np.array(ss))
#     std_tst.append(np.sqrt(np.var(act-pred)))
    
sim_trn=[850,650, 846]
sim_tst=[38,118, 23]
# sim_tstMS=[]
# for ww in sim_tstS:
#     sim_tstMS.append(ww)

stats_trn2=np.array([RMSE_trn,R2_trn,MAE_trn,bias_trn,std_trn,mape_trn])   
stats_tst2=np.array([RMSE_tst,R2_tst,MAE_tst,bias_tst,std_tst,mape_tst])  

stats_trn=np.concatenate((stats_trn2, np.array(sim_trn).reshape(1,-1)),axis=0)
stats_tst=np.concatenate((stats_tst2, np.array(sim_tst).reshape(1,-1)),axis=0)

# savemat('pred_act.mat',dict(pred=pred_tst_all,act=act_tst_all))
 
#%%

algnme=['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']

fstcol=['\multirow{2}{5em}{RMSE (sec)}',
'\multirow{2}{5em}{R2 (\%)}',
'\multirow{2}{5em}{MAE (sec)}',
'\multirow{2}{5em}{Bias (sec)}',
'\multirow{2}{5em}{$\sigma$(AE) (sec)}','\multirow{2}{5em}{Computation time (sec)}']
# xgb,lgb,cbr,rf
with open("myfile.txt",'w',encoding='utf-8') as file1:
    for ii in range(0,6):
          file1.write("{fstC} & {tr} & {C:3.2f} &{R:3.2f} & {L:3.2f} & {X:3.2f} \\\\".format(fstC=fstcol[ii],tr='Train',
                       C=conv_trn[ii],R=stats_trn[ii,0],L=stats_trn[ii,1],X=stats_trn[ii,2])) 
          file1.write('\n')
          file1.write(" & {tr} & {C:3.2f} &{R:3.2f} & {L:3.2f} & {X:3.2f} \\\\".format(tr='Test',
                       C=conv_tst[ii],R=stats_tst[ii,0],L=stats_tst[ii,1],X=stats_tst[ii,2])) 
        
        # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));

      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
          file1.write('\n')

    