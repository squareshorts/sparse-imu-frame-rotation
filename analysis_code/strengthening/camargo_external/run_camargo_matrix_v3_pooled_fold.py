from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import make_augmented_training_set,perturb_by_subject
from strengthening.run_rotation_matrix_v2 import WeightedRidge
ANGLES=np.asarray([0.,2.5,5.,7.5,10.,12.5,15.]);MC=50;COPIES=5;SEED=20260907
TASKS={'knee_two':{'sensors':(1,2),'targets':(0,1),'names':('knee_stance_flex_deg','knee_swing_flex_deg')},'ankle_two':{'sensors':(2,3),'targets':(2,3),'names':('ankle_dorsiflex_deg','ankle_plantarflex_deg')}}
def alpha_for(cap):return 10.0 if cap<=2.5 else 100.0
def main(fold,data_path,outdir):
 t=time.time();z=np.load(data_path,allow_pickle=True);X,T,G,D,S,F=z['X'],z['T'],z['subjects'],z['durations'],z['speeds'],z['outer_fold'];names={int(i):str(n) for i,n in zip(z['subjects'],z['subject_names'])};outdir.mkdir(parents=True,exist_ok=True);tr=np.flatnonzero(F!=fold);te=np.flatnonzero(F==fold);acc={};fit=[]
 for task_name,task in TASKS.items():
  models={}
  for cap in ANGLES:
   Xtr,Ytr,Gtr,Dtr,Str=make_augmented_training_set(X[tr],T[tr][:,task['targets']],G[tr],D[tr],S[tr],task['sensors'],float(cap),COPIES,seed=SEED+fold*100000+len(task_name)*1000+int(round(cap*10)))
   w=np.full(len(Xtr),1/(COPIES+1),float) if cap>0 else np.ones(len(Xtr));a=alpha_for(cap);Ftr=engineered_features(Xtr,task['sensors'],Dtr,Str);models[float(cap)]=WeightedRidge(a).fit(Ftr,Ytr,w);fit.append({'outer_fold':fold,'task':task_name,'train_augmentation_max_deg':cap,'alpha':a,'n_fit_rows':len(Xtr),'total_training_weight':float(w.sum())})
  y=T[te][:,task['targets']];g=G[te]
  for ang in ANGLES:
   reps=1 if ang==0 else MC
   for rep in range(reps):
    if ang==0:Xte=X[te]
    else:
     pseed=SEED+fold*1000000+len(task_name)*10000+int(round(ang*10))*100+rep;Xte,_=perturb_by_subject(X[te],g,task['sensors'],float(ang),pseed,exact_angle=True)
    Fte=engineered_features(Xte,task['sensors'],D[te],S[te])
    for cap,m in models.items():
     pred=m.predict(Fte);e2=(pred-y)**2
     for sid in np.unique(g):
      ii=g==sid
      for j,target in enumerate(task['names']):
       k=(task_name,target,cap,float(ang),int(sid));ss,n=acc.get(k,(0.,0));acc[k]=(ss+float(e2[ii,j].sum()),n+int(ii.sum()))
   print('matrixv3',fold,task_name,ang,flush=True)
 rows=[]
 for (task,target,cap,ang,sid),(ss,n) in acc.items():rows.append({'outer_fold':fold,'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,'subject':sid,'subject_name':names[sid],'rmse_deg':float(np.sqrt(ss/n)),'pooled_error_count':n})
 pd.DataFrame(rows).to_csv(outdir/f'participant_matrix_fold{fold}.csv',index=False);pd.DataFrame(fit).to_csv(outdir/f'fit_log_fold{fold}.csv',index=False);print('elapsed',time.time()-t)
if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--data',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();main(a.fold,Path(a.data),Path(a.outdir))
