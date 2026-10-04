from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.confirmatory.data import load_study_data
from src.confirmatory.folds import load_outer_folds
from src.confirmatory.constants import PRIMARY_TASKS
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import perturb_by_subject
from strengthening.run_orientation_baselines_fold import build_training_templates,template_register,functional_register
from strengthening.run_rotation_matrix_v2 import WeightedRidge,ANGLES,MC,SEED,TASKS
OUT=ROOT/'strengthening'/'orientation_baselines_v2';OUT.mkdir(parents=True,exist_ok=True);METHODS=('template_registration','gait_functional_calibration')
def tf(method,X,g,sensors,templates):return template_register(X,g,sensors,templates) if method=='template_registration' else functional_register(X,g,sensors)
def main(f):
 t=time.time();d=load_study_data(ROOT/'data/processed/strict_parameters.npz');outer=load_outer_folds(ROOT/'protocol/outer_folds.csv',d.subjects);tr=np.flatnonzero(outer!=f);te=np.flatnonzero(outer==f);acc={};inv=[]
 for ti,task_name in enumerate(TASKS):
  task=PRIMARY_TASKS[task_name];templates=build_training_templates(d.X[tr],d.subjects[tr],task['sensors']);models={};clean={}
  for method in METHODS:
   Xtr=tf(method,d.X[tr],d.subjects[tr],task['sensors'],templates);clean[method]=tf(method,d.X[te],d.subjects[te],task['sensors'],templates);Ftr=engineered_features(Xtr,task['sensors'],d.durations[tr],d.speeds[tr]);models[method]=WeightedRidge(100.0).fit(Ftr,d.T[tr][:,task['targets']])
  y=d.T[te][:,task['targets']];g=d.subjects[te];subs=np.unique(g)
  for ang in ANGLES:
   reps=1 if ang==0 else MC
   for rep in range(reps):
    if ang==0:Xp=d.X[te]
    else:
     seed=SEED+f*3_000_000+ti*100_000+int(round(ang*10))*100+rep;Xp,_=perturb_by_subject(d.X[te],g,task['sensors'],float(ang),seed,exact_angle=True)
    for method in METHODS:
     Xr=tf(method,Xp,g,task['sensors'],templates);maxabs=max(float(np.max(np.abs(Xr[:,s]-clean[method][:,s]))) for s in task['sensors']);inv.append({'outer_fold':f,'task':task_name,'method':method,'test_rotation_deg':ang,'realization':rep,'max_abs_registered_signal_difference_vs_clean':maxabs});F=engineered_features(Xr,task['sensors'],d.durations[te],d.speeds[te]);e=(models[method].predict(F)-y)**2
     for sid in subs:
      ii=g==sid;sums=e[ii].sum(axis=0);cnt=int(ii.sum())
      for j,target in enumerate(task['target_names']):
       k=(method,task_name,target,float(ang),int(sid));ss,n=acc.get(k,(0.,0));acc[k]=(ss+float(sums[j]),n+cnt)
   print(f,task_name,ang,flush=True)
 rows=[]
 for (method,task,target,ang,sid),(s,n) in acc.items():rows.append({'outer_fold':f,'method':method,'task':task,'target':target,'test_rotation_deg':ang,'subject':sid,'rmse_deg':float(np.sqrt(s/n))})
 pd.DataFrame(rows).to_csv(OUT/f'participant_metrics_fold{f}.csv',index=False);pd.DataFrame(inv).to_csv(OUT/f'invariance_fold{f}.csv',index=False);print('elapsed',time.time()-t)
if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();main(a.fold)
