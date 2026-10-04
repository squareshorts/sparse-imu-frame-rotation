from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'strengthening'))
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import perturb_by_subject
from strengthening.run_rotation_matrix_v2 import WeightedRidge
from run_orientation_baselines_fold import build_training_templates,template_register,functional_register
ANGLES=np.asarray([0.,2.5,5.,7.5,10.,12.5,15.]);MC=50;SEED=20260907;ALPHA=100.0
TASKS={'knee_two':{'sensors':(1,2),'targets':(0,1),'names':('knee_stance_flex_deg','knee_swing_flex_deg')},'ankle_two':{'sensors':(2,3),'targets':(2,3),'names':('ankle_dorsiflex_deg','ankle_plantarflex_deg')}}
def main(fold,data_path,outdir):
 t=time.time();z=np.load(data_path,allow_pickle=True);X,T,G,D,S,F=z['X'],z['T'],z['subjects'],z['durations'],z['speeds'],z['outer_fold'];names={int(i):str(n) for i,n in zip(z['subjects'],z['subject_names'])};outdir.mkdir(parents=True,exist_ok=True);tr=np.flatnonzero(F!=fold);te=np.flatnonzero(F==fold);acc={};inv=[]
 for task_name,task in TASKS.items():
  sensors=task['sensors'];templates=build_training_templates(X[tr],G[tr],sensors);train={'template_registration':template_register(X[tr],G[tr],sensors,templates),'gait_functional_calibration':functional_register(X[tr],G[tr],sensors)};clean={'template_registration':template_register(X[te],G[te],sensors,templates),'gait_functional_calibration':functional_register(X[te],G[te],sensors)};models={}
  for method,Xtr in train.items():models[method]=WeightedRidge(ALPHA).fit(engineered_features(Xtr,sensors,D[tr],S[tr]),T[tr][:,task['targets']])
  y=T[te][:,task['targets']];g=G[te]
  for ang in ANGLES:
   reps=1 if ang==0 else MC
   for rep in range(reps):
    if ang==0:Xp=X[te]
    else:
     pseed=SEED+fold*1000000+len(task_name)*10000+int(round(ang*10))*100+rep;Xp,_=perturb_by_subject(X[te],g,sensors,float(ang),pseed,exact_angle=True)
    for method in models:
     Xr=template_register(Xp,g,sensors,templates) if method=='template_registration' else functional_register(Xp,g,sensors);inv.append({'outer_fold':fold,'task':task_name,'method':method,'test_rotation_deg':ang,'realization':rep,'max_abs_registered_signal_difference_vs_clean':max(float(np.max(np.abs(Xr[:,s]-clean[method][:,s]))) for s in sensors)});pred=models[method].predict(engineered_features(Xr,sensors,D[te],S[te]));e2=(pred-y)**2
     for sid in np.unique(g):
      ii=g==sid
      for j,target in enumerate(task['names']):
       k=(method,task_name,target,float(ang),int(sid));ss,n=acc.get(k,(0.,0));acc[k]=(ss+float(e2[ii,j].sum()),n+int(ii.sum()))
   print('basev3',fold,task_name,ang,flush=True)
 rows=[]
 for (method,task,target,ang,sid),(ss,n) in acc.items():rows.append({'outer_fold':fold,'method':method,'task':task,'target':target,'test_rotation_deg':ang,'subject':sid,'subject_name':names[sid],'rmse_deg':float(np.sqrt(ss/n)),'pooled_error_count':n})
 pd.DataFrame(rows).to_csv(outdir/f'participant_metrics_fold{fold}.csv',index=False);pd.DataFrame(inv).to_csv(outdir/f'invariance_fold{fold}.csv',index=False);print('elapsed',time.time()-t)
if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--data',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();main(a.fold,Path(a.data),Path(a.outdir))
