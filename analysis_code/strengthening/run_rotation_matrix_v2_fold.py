from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.confirmatory.data import load_study_data
from src.confirmatory.folds import load_outer_folds
from src.confirmatory.constants import PRIMARY_TASKS
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import make_augmented_training_set,perturb_by_subject
from strengthening.run_rotation_matrix_v2 import WeightedRidge,ANGLES,MC,COPIES,SEED,TASKS,OUT

def alpha_for(cap): return 10.0 if cap<=2.5 else 100.0

def main(ofold):
 t0=time.time();d=load_study_data(ROOT/'data/processed/strict_parameters.npz');outer=load_outer_folds(ROOT/'protocol/outer_folds.csv',d.subjects)
 tr=np.flatnonzero(outer!=ofold);te=np.flatnonzero(outer==ofold);accum={};fitlog=[]
 for ti,task_name in enumerate(TASKS):
  task=PRIMARY_TASKS[task_name];models={}
  for cap in ANGLES:
   seed=SEED+ofold*2_000_000+int(round(cap*10))*10_000+ti
   Xtr,Ytr,Gtr,Dtr,Str=make_augmented_training_set(d.X[tr],d.T[tr][:,task['targets']],d.subjects[tr],d.durations[tr],d.speeds[tr],task['sensors'],float(cap),COPIES,seed=seed)
   w=np.full(len(Xtr),1.0/(COPIES+1),float) if cap>0 else np.ones(len(Xtr),float)
   Ftr=engineered_features(Xtr,task['sensors'],Dtr,Str);a=alpha_for(cap);models[float(cap)]=WeightedRidge(a).fit(Ftr,Ytr,w)
   fitlog.append({'outer_fold':ofold,'task':task_name,'train_augmentation_max_deg':float(cap),'alpha':a,'n_training_rows':len(Xtr),'total_training_weight':float(w.sum()),'feature_dim':Ftr.shape[1]})
  y=d.T[te][:,task['targets']];g=d.subjects[te];unique=np.unique(g)
  for ang in ANGLES:
   reps=1 if ang==0 else MC
   for rep in range(reps):
    if ang==0:Xte=d.X[te]
    else:
     pseed=SEED+ofold*3_000_000+ti*100_000+int(round(ang*10))*100+rep
     Xte,_=perturb_by_subject(d.X[te],g,task['sensors'],float(ang),pseed,exact_angle=True)
    Fte=engineered_features(Xte,task['sensors'],d.durations[te],d.speeds[te])
    for cap,m in models.items():
     e=(m.predict(Fte)-y)**2
     for sid in unique:
      ii=(g==sid);sums=e[ii].sum(axis=0);cnt=int(ii.sum())
      for j,target in enumerate(task['target_names']):
       k=(task_name,target,cap,float(ang),int(sid));ss,n=accum.get(k,(0.0,0));accum[k]=(ss+float(sums[j]),n+cnt)
   print('fold',ofold,'task',task_name,'angle',ang,flush=True)
 rows=[]
 for (task,target,cap,ang,sid),(s,n) in accum.items():rows.append({'outer_fold':ofold,'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,'subject':sid,'rmse_deg':float(np.sqrt(s/n)),'pooled_error_count':n})
 pd.DataFrame(rows).to_csv(OUT/f'participant_matrix_fold{ofold}.csv',index=False);pd.DataFrame(fitlog).to_csv(OUT/f'fit_log_fold{ofold}.csv',index=False)
 print('elapsed',time.time()-t0,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();main(a.fold)
