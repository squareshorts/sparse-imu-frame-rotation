from pathlib import Path
import numpy as np,pandas as pd,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from strengthening.run_rotation_matrix_v2 import bootstrap,ANGLES,SEED,TOL,OUT
parts=[pd.read_csv(OUT/f'participant_matrix_fold{f}.csv') for f in range(1,6)]
part=pd.concat(parts,ignore_index=True).sort_values(['task','target','train_augmentation_max_deg','test_rotation_deg','subject'])
part.to_csv(OUT/'participant_matrix.csv',index=False)
fit=pd.concat([pd.read_csv(OUT/f'fit_log_fold{f}.csv') for f in range(1,6)],ignore_index=True);fit.to_csv(OUT/'fit_log.csv',index=False)
pd.DataFrame([{'outer_fold':f,'train_augmentation_max_deg':cap,'selected_alpha':10.0 if cap<=2.5 else 100.0} for f in range(1,6) for cap in ANGLES]).to_csv(OUT/'alpha_map.csv',index=False)
sums=[]
for i,(keys,q) in enumerate(part.groupby(['task','target','train_augmentation_max_deg','test_rotation_deg'],sort=True)):
 mean,lo,hi=bootstrap(q.rmse_deg.to_numpy(),SEED+5_000_000+i);task,target,cap,ang=keys
 sums.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,'mean_participant_rmse_deg':mean,'ci_low':lo,'ci_high':hi,'n_subjects':len(q)})
sm=pd.DataFrame(sums);sm.to_csv(OUT/'matrix_summary.csv',index=False)
# paired clean penalties
pen=[];clean=part[part.test_rotation_deg==0];base=clean[clean.train_augmentation_max_deg==0][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'base'})
for cap in ANGLES[1:]:
 q=clean[clean.train_augmentation_max_deg==cap][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'aug'});m=base.merge(q,on=['task','target','subject'])
 for j,((task,target),z) in enumerate(m.groupby(['task','target'])):
  diff=z.aug.to_numpy()-z.base.to_numpy();mean,lo,hi=bootstrap(diff,SEED+6_000_000+int(cap*100)+j);pen.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'mean_clean_penalty_deg':mean,'ci_low':lo,'ci_high':hi})
pd.DataFrame(pen).to_csv(OUT/'clean_penalties.csv',index=False)
# rescue vs unaug at each test
resc=[]
base=part[part.train_augmentation_max_deg==0][['task','target','test_rotation_deg','subject','rmse_deg']].rename(columns={'rmse_deg':'base'})
for cap in ANGLES[1:]:
 q=part[part.train_augmentation_max_deg==cap][['task','target','test_rotation_deg','subject','rmse_deg']].rename(columns={'rmse_deg':'aug'});m=base.merge(q,on=['task','target','test_rotation_deg','subject'])
 for j,((task,target,ang),z) in enumerate(m.groupby(['task','target','test_rotation_deg'])):
  diff=z.base.to_numpy()-z.aug.to_numpy();mean,lo,hi=bootstrap(diff,SEED+7_000_000+int(cap*1000)+int(ang*100)+j);resc.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,'mean_rescue_deg':mean,'ci_low':lo,'ci_high':hi})
pd.DataFrame(resc).to_csv(OUT/'augmentation_rescue.csv',index=False)
# deployment gate
rows=[]
for delta in ANGLES:
 cand=[]
 for cap in ANGLES:
  q=sm[(sm.train_augmentation_max_deg==cap)&(sm.test_rotation_deg<=delta+1e-9)];expected=4*int(np.sum(ANGLES<=delta+1e-9))
  if len(q)==expected and float(q.ci_high.max())<=TOL:
   cq=sm[(sm.train_augmentation_max_deg==cap)&(sm.test_rotation_deg==0)];cand.append((float(cap),float(q.ci_high.max()),float(q.mean_participant_rmse_deg.max()),float(cq.mean_participant_rmse_deg.max()),float(cq.ci_high.max())))
 if cand:
  cand.sort(key=lambda x:x[0]);cap,hi,m,cm,ch=cand[0];rows.append({'field_q95_rotation_deg':delta,'decision':'DEPLOY','selected_train_augmentation_max_deg':cap,'worst_ci_high_deg_through_q95':hi,'worst_mean_rmse_deg_through_q95':m,'worst_clean_mean_rmse_deg':cm,'worst_clean_ci_high_deg':ch,'n_qualifying_rows':len(cand)})
 else: rows.append({'field_q95_rotation_deg':delta,'decision':'NO_DEPLOY_AUGMENTATION','selected_train_augmentation_max_deg':np.nan,'worst_ci_high_deg_through_q95':np.nan,'worst_mean_rmse_deg_through_q95':np.nan,'worst_clean_mean_rmse_deg':np.nan,'worst_clean_ci_high_deg':np.nan,'n_qualifying_rows':0})
dep=pd.DataFrame(rows);dep.to_csv(OUT/'deployment_rule.csv',index=False)
worst=sm.groupby(['train_augmentation_max_deg','test_rotation_deg'],as_index=False).agg(worst_mean_rmse_deg=('mean_participant_rmse_deg','max'),worst_ci_high_deg=('ci_high','max'));worst.to_csv(OUT/'worst_endpoint_matrix.csv',index=False)
print('DEPLOYMENT')
print(dep.to_string(index=False))
print('\nWORST CI')
print(worst.pivot(index='train_augmentation_max_deg',columns='test_rotation_deg',values='worst_ci_high_deg').round(3).to_string())
print('\nSELECTED CELLS')
print(sm[((sm.train_augmentation_max_deg==0)&sm.test_rotation_deg.isin([0,5,7.5,10]))|((sm.train_augmentation_max_deg==5)&sm.test_rotation_deg.isin([0,7.5,10]))|((sm.train_augmentation_max_deg==10)&(sm.test_rotation_deg==10))].to_string(index=False))
