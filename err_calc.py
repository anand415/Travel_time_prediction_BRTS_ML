# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:08:33 2020

@author: anand
"""
import pickle
import numpy as np
import pandas as pd

[y_actXA,y_trainprederrXA,trainindXA,testindXA,y_prederrXA,xgb_fld_paramXA]=pickle.load(open("xgb_maehyper_tuned_paramall.pkl","rb"))

# [y_actXS,y_trainprederrXS,trainindXS,testindXS,y_prederrXS,xgb_fld_paramXS]=pickle.load(open("xgb_msehyper_tuned_paramall.pkl","rb"))
# [y_actLS,y_trainprederrLS,trainindLS,testindLS,y_prederrLS,lgb_fld_paramS]=pickle.load(open("lgb_mssehyper_tuned_paramall.pkl","rb"))
# [y_actLA,y_trainprederrLA,trainindLA,testindLA,y_prederrLA,lgb_fld_paramA]=pickle.load(open("lgb_maehyper_tuned_paramall.pkl","rb"))



TR_XS_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_trainprederrXS)))))
TR_XA_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_trainprederrXA)))))
TR_LA_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_trainprederrLA)))))
TR_LS_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_trainprederrLS)))))




TR_XS_A=(np.mean(np.abs(np.hstack(np.array(y_trainprederrXS)))))
TR_XA_A=(np.mean(np.abs(np.hstack(np.array(y_trainprederrXA)))))
TR_LA_A=(np.mean(np.abs(np.hstack(np.array(y_trainprederrLA)))))
TR_LS_A=(np.mean(np.abs(np.hstack(np.array(y_trainprederrLS)))))

TR_XS_bs=(np.mean((np.hstack(np.array(y_trainprederrXS)))))
TR_XA_bs=(np.mean((np.hstack(np.array(y_trainprederrXA)))))
TR_LA_bs=(np.mean((np.hstack(np.array(y_trainprederrLA)))))
TR_LS_bs=(np.mean((np.hstack(np.array(y_trainprederrLS)))))


TR_XS_st=(np.std(np.abs(np.hstack(np.array(y_trainprederrXS)))))
TR_XA_st=(np.std(np.abs(np.hstack(np.array(y_trainprederrXA)))))
TR_LA_st=(np.std(np.abs(np.hstack(np.array(y_trainprederrLA)))))
TR_LS_st=(np.std(np.abs(np.hstack(np.array(y_trainprederrLS)))))


yact=np.hstack(np.array(y_actXA))
TSS=np.sum(np.square(yact-np.mean(yact)))
TR_XS_R2=1-((np.sum(np.square(np.hstack(np.array(y_trainprederrXS)))))/TSS)
TR_XA_R2=1-((np.sum(np.square(np.hstack(np.array(y_trainprederrXA)))))/TSS)
TR_LA_R2=1-((np.sum(np.square(np.hstack(np.array(y_trainprederrLA)))))/TSS)
TR_LS_R2=1-((np.sum(np.square(np.hstack(np.array(y_trainprederrLS)))))/TSS)


# ----------------------------

TS_XS_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_prederrXS)))))
TS_XA_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_prederrXA)))))
TS_LA_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_prederrLA)))))
TS_LS_S=np.sqrt(np.mean(np.square(np.hstack(np.array(y_prederrLS)))))




TS_XS_A=(np.mean(np.abs(np.hstack(np.array(y_prederrXS)))))
TS_XA_A=(np.mean(np.abs(np.hstack(np.array(y_prederrXA)))))
TS_LA_A=(np.mean(np.abs(np.hstack(np.array(y_prederrLA)))))
TS_LS_A=(np.mean(np.abs(np.hstack(np.array(y_prederrLS)))))


TS_XS_bs=(np.mean((np.hstack(np.array(y_prederrXS)))))
TS_XA_bs=(np.mean((np.hstack(np.array(y_prederrXA)))))
TS_LA_bs=(np.mean((np.hstack(np.array(y_prederrLA)))))
TS_LS_bs=(np.mean((np.hstack(np.array(y_prederrLS)))))


TS_XS_st=(np.std(np.abs(np.hstack(np.array(y_prederrXS)))))
TS_XA_st=(np.std(np.abs(np.hstack(np.array(y_prederrXA)))))
TS_LA_st=(np.std(np.abs(np.hstack(np.array(y_prederrLA)))))
TS_LS_st=(np.std(np.abs(np.hstack(np.array(y_prederrLS)))))


yact=np.hstack(np.array(y_actXA))
TSS=np.sum(np.square(yact-np.mean(yact)))
TS_XS_R2=1-((np.sum(np.square(np.hstack(np.array(y_prederrXS)))))/TSS)
TS_XA_R2=1-((np.sum(np.square(np.hstack(np.array(y_prederrXA)))))/TSS)
TS_LA_R2=1-((np.sum(np.square(np.hstack(np.array(y_prederrLA)))))/TSS)
TS_LS_R2=1-((np.sum(np.square(np.hstack(np.array(y_prederrLS)))))/TSS)
# ----------------------------

indx={'mse','bmse','cmae','dmae','eR2','fR2'}
Compare_result = pd.DataFrame({'xgb_TR': [TR_XS_S,TR_XA_S,TR_XS_A,TR_XA_A,TR_XS_R2,TR_XA_R2,TR_XS_st,TR_XA_st],'xgb_TS': [TS_XS_S,TS_XA_S,TS_XS_A,TS_XA_A,TS_XS_R2,TS_XA_R2,TS_XS_st,TS_XA_st], 
                               'lgb_TR': [TR_LS_S,TR_LA_S,TR_LS_A,TR_LA_A,TR_LS_R2,TR_LA_R2,TR_LS_st,TR_LA_st],'lgb_TS': [TS_LS_S,TS_LA_S,TS_LS_A,TS_LA_A,TS_LS_R2,TS_LA_R2,TS_LS_st,TS_LA_st]}
                              # index=indx
                              )

# Compare_result_old=Compare_result

# Compare_result = Compare_result.reindex(indx)

# Compare_result[1,:]=[XS_S,XA_S]
#                LS_S LA_S
               


