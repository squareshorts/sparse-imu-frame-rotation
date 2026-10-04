from pathlib import Path
import numpy as np,pandas as pd,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from strengthening.run_rotation_matrix_v2 import bootstrap,SEED
OUT=ROOT/'strengthening'/'orientation_baselines_v2'
part=pd.concat([pd.read_csv(OUT/f'participant_metrics_fold{f}.csv') for f in range(1,6)],ignore_index=True);part.to_csv(OUT/'participant_metrics.csv',index=False)
inv=pd.concat([pd.read_csv(OUT/f'invariance_fold{f}.csv') for f in range(1,6)],ignore_index=True);inv.to_csv(OUT/'invariance.csv',index=False)
s=[]
for i,(keys,q) in enumerate(part.groupby(['method','task','target','test_rotation_deg'],sort=True)):
 m,lo,hi=bootstrap(q.rmse_deg.to_numpy(),SEED+10_000_000+i);method,task,target,ang=keys;s.append({'method':method,'task':task,'target':target,'test_rotation_deg':ang,'mean_participant_rmse_deg':m,'ci95_low_deg':lo,'ci95_high_deg':hi,'n_participants':len(q)})
sm=pd.DataFrame(s);sm.to_csv(OUT/'summary.csv',index=False)
print(sm[sm.test_rotation_deg.isin([0,10])].to_string(index=False));print('\nmax inv',inv.max(numeric_only=True)['max_abs_registered_signal_difference_vs_clean'])
