from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.confirmatory.data import load_study_data
from src.confirmatory.folds import load_outer_folds,inner_splits_for_outer_train
from src.confirmatory.constants import PRIMARY_TASKS
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import perturb_by_subject
from src.confirmatory.metrics import composite_participant_rmse
from strengthening.run_orientation_baselines_fold import build_training_templates,template_register,functional_register
from strengthening.run_rotation_matrix_v2 import WeightedRidge,ANGLES,ALPHAS,MC,SEED,TASKS,bootstrap
OUT=ROOT/'strengthening'/'orientation_baselines_v2';OUT.mkdir(parents=True,exist_ok=True)
METHODS=('template_registration','gait_functional_calibration')

def transform(method,X,g,sensors,templates=None):
 return template_register(X,g,sensors,templates) if method=='template_registration' else functional_register(X,g,sensors)

def select_alpha(d,outer,ofold,method):
 otr=np.flatnonzero(outer!=ofold);g=d.subjects[otr];scores={a:[] for a in ALPHAS};rows=[]
 for ifold,(it,iv) in enumerate(inner_splits_for_outer_train(g,3),start=1):
  itr=otr[it];iva=otr[iv];prepared=[]
  for task_name in TASKS:
   task=PRIMARY_TASKS[task_name];templates=build_training_templates(d.X[itr],d.subjects[itr],task['sensors']) if method=='template_registration' else None
   Xtr=transform(method,d.X[itr],d.subjects[itr],task['sensors'],templates);Xva=transform(method,d.X[iva],d.subjects[iva],task['sensors'],templates)
   prepared.append((task,engineered_features(Xtr,task['sensors'],d.durations[itr],d.speeds[itr]),d.T[itr][:,task['targets']],engineered_features(Xva,task['sensors'],d.durations[iva],d.speeds[iva])))
  for a in ALPHAS:
   ts=[]
   for task,Ftr,Ytr,Fva in prepared:
    p=WeightedRidge(a).fit(Ftr,Ytr).predict(Fva);ts.append(composite_participant_rmse(d.T[iva][:,task['targets']],p,d.subjects[iva]))
   sc=float(np.mean(ts));scores[a].append(sc);rows.append({'outer_fold':ofold,'method':method,'alpha':a,'inner_fold':ifold,'score':sc})
 means={a:float(np.mean(v)) for a,v in scores.items()};best=min(means,key=means.get)
 for r in rows:r['mean_score_for_alpha']=means[r['alpha']];r['selected_alpha']=best
 return float(best),rows

def main():
 t=time.time();d=load_study_data(ROOT/'data/processed/strict_parameters.npz');outer=load_outer_folds(ROOT/'protocol/outer_folds.csv',d.subjects)
 amap={};arows=[]
 for f in range(1,6):
  for method in METHODS:
   a,r=select_alpha(d,outer,f,method);amap[(f,method)]=a;arows.extend(r);print('alpha',f,method,a,flush=True)
 pd.DataFrame(arows).to_csv(OUT/'selected_alphas_by_fold.csv',index=False);pd.DataFrame([{'outer_fold':f,'method':m,'selected_alpha':a} for (f,m),a in amap.items()]).to_csv(OUT/'alpha_map.csv',index=False)
 # evaluate; accum error squares for pooled MC, invariance max differences
 acc={};inv=[];fit=[]
 for f in range(1,6):
  tr=np.flatnonzero(outer!=f);te=np.flatnonzero(outer==f)
  for ti,task_name in enumerate(TASKS):
   task=PRIMARY_TASKS[task_name];templates=build_training_templates(d.X[tr],d.subjects[tr],task['sensors'])
   models={};cleanreg={}
   for method in METHODS:
    Xtr=transform(method,d.X[tr],d.subjects[tr],task['sensors'],templates);Xcl=transform(method,d.X[te],d.subjects[te],task['sensors'],templates);cleanreg[method]=Xcl
    Ftr=engineered_features(Xtr,task['sensors'],d.durations[tr],d.speeds[tr]);a=amap[(f,method)];models[method]=WeightedRidge(a).fit(Ftr,d.T[tr][:,task['targets']]);fit.append({'outer_fold':f,'task':task_name,'method':method,'alpha':a})
   y=d.T[te][:,task['targets']];g=d.subjects[te];subs=np.unique(g)
   for ang in ANGLES:
    reps=1 if ang==0 else MC
    for rep in range(reps):
     if ang==0:Xp=d.X[te]
     else:
      seed=SEED+f*3_000_000+ti*100_000+int(round(ang*10))*100+rep;Xp,_=perturb_by_subject(d.X[te],g,task['sensors'],float(ang),seed,exact_angle=True)
     for method in METHODS:
      Xr=transform(method,Xp,g,task['sensors'],templates);maxabs=max(float(np.max(np.abs(Xr[:,s]-cleanreg[method][:,s]))) for s in task['sensors']);inv.append({'outer_fold':f,'task':task_name,'method':method,'test_rotation_deg':ang,'realization':rep,'max_abs_registered_signal_difference_vs_clean':maxabs})
      F=engineered_features(Xr,task['sensors'],d.durations[te],d.speeds[te]);e=(models[method].predict(F)-y)**2
      for sid in subs:
       ii=g==sid;sums=e[ii].sum(axis=0);cnt=int(ii.sum())
       for j,target in enumerate(task['target_names']):
        k=(method,task_name,target,float(ang),int(sid));ss,n=acc.get(k,(0.,0));acc[k]=(ss+float(sums[j]),n+cnt)
    print('eval',f,task_name,ang,flush=True)
 pd.DataFrame(fit).to_csv(OUT/'fit_log.csv',index=False);pd.DataFrame(inv).to_csv(OUT/'invariance.csv',index=False)
 rows=[]
 for (method,task,target,ang,sid),(s,n) in acc.items():rows.append({'method':method,'task':task,'target':target,'test_rotation_deg':ang,'subject':sid,'rmse_deg':float(np.sqrt(s/n))})
 part=pd.DataFrame(rows);part.to_csv(OUT/'participant_metrics.csv',index=False)
 sums=[]
 for i,(keys,q) in enumerate(part.groupby(['method','task','target','test_rotation_deg'],sort=True)):
  m,lo,hi=bootstrap(q.rmse_deg.to_numpy(),SEED+9_000_000+i);method,task,target,ang=keys;sums.append({'method':method,'task':task,'target':target,'test_rotation_deg':ang,'mean_participant_rmse_deg':m,'ci95_low_deg':lo,'ci95_high_deg':hi,'n_participants':len(q)})
 pd.DataFrame(sums).to_csv(OUT/'summary.csv',index=False)
 print(pd.read_csv(OUT/'alpha_map.csv').to_string(index=False));print(pd.DataFrame(sums)[pd.DataFrame(sums).test_rotation_deg.isin([0,10])].to_string(index=False));print('max inv',pd.DataFrame(inv).max(numeric_only=True)['max_abs_registered_signal_difference_vs_clean']);print('elapsed',time.time()-t)
if __name__=='__main__':main()
