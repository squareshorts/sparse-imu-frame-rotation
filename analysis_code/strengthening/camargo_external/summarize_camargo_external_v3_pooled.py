from __future__ import annotations

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, help='Prepared Camargo external-analysis directory')
args = ap.parse_args()
BASE = Path(args.base).resolve()
MATRIX = BASE/'matrix_v3_pooled'
BASELINES = BASE/'baselines_v3_pooled'
BOOT=5000
SEED=20260907
ANGLES=np.array([0.,2.5,5.,7.5,10.,12.5,15.])
CAPS=ANGLES.copy()


def boot_mean(x, seed):
    x=np.asarray(x,float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),(BOOT,len(x)))
    b=x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(b,.025)), float(np.quantile(b,.975))


def summarize(df, keys):
    rows=[]
    for i,(k,g) in enumerate(df.groupby(keys,sort=True)):
        if not isinstance(k,tuple): k=(k,)
        m,lo,hi=boot_mean(g.rmse_deg.to_numpy(),SEED+10000+i*37)
        row=dict(zip(keys,k)); row.update(mean_participant_rmse_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,n_participants=int(g.subject.nunique()))
        rows.append(row)
    return pd.DataFrame(rows)

part=pd.concat([pd.read_csv(MATRIX/f'participant_matrix_fold{i}.csv') for i in range(1,6)],ignore_index=True)
bpart=pd.concat([pd.read_csv(BASELINES/f'participant_metrics_fold{i}.csv') for i in range(1,6)],ignore_index=True)
inv=pd.concat([pd.read_csv(BASELINES/f'invariance_fold{i}.csv') for i in range(1,6)],ignore_index=True)
assert part.subject.nunique()==19
assert bpart.subject.nunique()==19
part.to_csv(MATRIX/'participant_matrix.csv',index=False)
bpart.to_csv(BASELINES/'participant_metrics.csv',index=False)
inv.to_csv(BASELINES/'invariance.csv',index=False)
ms=summarize(part,['task','target','train_augmentation_max_deg','test_rotation_deg'])
bs=summarize(bpart,['method','task','target','test_rotation_deg'])
ms.to_csv(MATRIX/'matrix_summary.csv',index=False)
bs.to_csv(BASELINES/'summary.csv',index=False)

# unaug degradation vs clean
rows=[]
u=part[part.train_augmentation_max_deg==0]
u0=u[u.test_rotation_deg==0][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'clean'})
for angle in ANGLES[1:]:
    ua=u[u.test_rotation_deg==angle][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'rot'})
    mm=u0.merge(ua,on=['task','target','subject'])
    for j,((task,target),g) in enumerate(mm.groupby(['task','target'],sort=True)):
        d=g.rot.to_numpy()-g.clean.to_numpy(); m,lo,hi=boot_mean(d,SEED+20000+int(angle*100)+j)
        rel=100*d/np.maximum(g.clean.to_numpy(),1e-12)
        rows.append(dict(task=task,target=target,test_rotation_deg=angle,mean_paired_degradation_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,mean_relative_degradation_pct=float(rel.mean()),median_relative_degradation_pct=float(np.median(rel)),n_participants=len(g)))
deg=pd.DataFrame(rows);deg.to_csv(MATRIX/'unaugmented_degradation.csv',index=False)

# clean penalties for each cap vs unaug clean
rows=[]
base=part[(part.train_augmentation_max_deg==0)&(part.test_rotation_deg==0)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'base'})
for cap in CAPS[1:]:
    aa=part[(part.train_augmentation_max_deg==cap)&(part.test_rotation_deg==0)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'aug'})
    mm=base.merge(aa,on=['task','target','subject'])
    for j,((task,target),g) in enumerate(mm.groupby(['task','target'],sort=True)):
        d=g.aug.to_numpy()-g.base.to_numpy();m,lo,hi=boot_mean(d,SEED+30000+int(cap*100)+j)
        rows.append(dict(task=task,target=target,train_augmentation_max_deg=cap,mean_clean_penalty_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,n_participants=len(g)))
cp=pd.DataFrame(rows);cp.to_csv(MATRIX/'clean_penalties.csv',index=False)

# augmentation vs unaug at each matching test angle; positive means aug better
rows=[]
for angle in ANGLES[1:]:
    ua=part[(part.train_augmentation_max_deg==0)&(part.test_rotation_deg==angle)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'unaug'})
    for cap in CAPS[1:]:
        aa=part[(part.train_augmentation_max_deg==cap)&(part.test_rotation_deg==angle)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'aug'})
        mm=ua.merge(aa,on=['task','target','subject'])
        for j,((task,target),g) in enumerate(mm.groupby(['task','target'],sort=True)):
            d=g.unaug.to_numpy()-g.aug.to_numpy();m,lo,hi=boot_mean(d,SEED+40000+int(angle*1000)+int(cap*10)+j)
            rows.append(dict(task=task,target=target,test_rotation_deg=angle,train_augmentation_max_deg=cap,mean_improvement_unaug_minus_aug_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,n_participants=len(g)))
res=pd.DataFrame(rows);res.to_csv(MATRIX/'augmentation_vs_unaugmented.csv',index=False)

# baseline invariance
isum=inv.groupby(['method','test_rotation_deg'],as_index=False).agg(max_abs_registered_signal_difference_vs_clean=('max_abs_registered_signal_difference_vs_clean','max'),median_abs_registered_signal_difference_vs_clean=('max_abs_registered_signal_difference_vs_clean','median'))
isum.to_csv(BASELINES/'invariance_summary.csv',index=False)

# Baselines paired vs unaug and matching aug10 at 10, plus aug7.5 at7.5.
rows=[]
for angle,cap in [(7.5,7.5),(10.,10.)]:
    for reference,refdf in [
        ('unaugmented',part[(part.train_augmentation_max_deg==0)&(part.test_rotation_deg==angle)]),
        (f'aug{cap:g}',part[(part.train_augmentation_max_deg==cap)&(part.test_rotation_deg==angle)])
    ]:
        ref=refdf[['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'reference_rmse'})
        for method in sorted(bpart.method.unique()):
            bb=bpart[(bpart.method==method)&(bpart.test_rotation_deg==angle)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'baseline_rmse'})
            mm=ref.merge(bb,on=['task','target','subject'])
            for j,((task,target),g) in enumerate(mm.groupby(['task','target'],sort=True)):
                # positive = baseline better (reference RMSE - baseline RMSE)
                d=g.reference_rmse.to_numpy()-g.baseline_rmse.to_numpy();m,lo,hi=boot_mean(d,SEED+50000+int(angle*1000)+len(rows)*11+j)
                rows.append(dict(method=method,reference=reference,test_rotation_deg=angle,task=task,target=target,mean_improvement_reference_minus_baseline_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,n_participants=len(g)))
bcomp=pd.DataFrame(rows);bcomp.to_csv(BASELINES/'paired_comparisons.csv',index=False)

# Clean baseline vs clean raw
rows=[]
raw0=part[(part.train_augmentation_max_deg==0)&(part.test_rotation_deg==0)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'raw'})
for method in sorted(bpart.method.unique()):
    bb=bpart[(bpart.method==method)&(bpart.test_rotation_deg==0)][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'baseline'})
    mm=raw0.merge(bb,on=['task','target','subject'])
    for j,((task,target),g) in enumerate(mm.groupby(['task','target'],sort=True)):
        # baseline - raw, positive means baseline clean cost
        d=g.baseline.to_numpy()-g.raw.to_numpy();m,lo,hi=boot_mean(d,SEED+60000+len(rows)*13+j)
        rows.append(dict(method=method,task=task,target=target,mean_clean_delta_baseline_minus_raw_deg=m,ci95_low_deg=lo,ci95_high_deg=hi,n_participants=len(g)))
cleanbase=pd.DataFrame(rows);cleanbase.to_csv(BASELINES/'clean_cost_vs_raw.csv',index=False)

# Compact JSON
report={
 'n_participants':int(part.subject.nunique()),
 'n_windows':int(np.load(BASE/'camargo_external.npz')['X'].shape[0]),
 'clean_unaugmented':ms[(ms.train_augmentation_max_deg==0)&(ms.test_rotation_deg==0)].to_dict('records'),
 'unaugmented_degradation_7p5':deg[deg.test_rotation_deg==7.5].to_dict('records'),
 'unaugmented_degradation_10':deg[deg.test_rotation_deg==10].to_dict('records'),
 'aug7p5_vs_unaug_at7p5':res[(res.test_rotation_deg==7.5)&(res.train_augmentation_max_deg==7.5)].to_dict('records'),
 'aug10_vs_unaug_at10':res[(res.test_rotation_deg==10)&(res.train_augmentation_max_deg==10)].to_dict('records'),
 'clean_penalty_aug10':cp[cp.train_augmentation_max_deg==10].to_dict('records'),
 'baseline_invariance':isum.to_dict('records'),
 'baseline_comparisons':bcomp.to_dict('records'),
 'baseline_clean_costs':cleanbase.to_dict('records')
}
(BASE/'external_replication_key_results_v3_pooled.json').write_text(json.dumps(report,indent=2))

for name,df in [
 ('Clean unaug',ms[(ms.train_augmentation_max_deg==0)&(ms.test_rotation_deg==0)]),
 ('Degradation 7.5',deg[deg.test_rotation_deg==7.5]),
 ('Degradation 10',deg[deg.test_rotation_deg==10]),
 ('Aug 7.5 vs unaug @7.5',res[(res.test_rotation_deg==7.5)&(res.train_augmentation_max_deg==7.5)]),
 ('Aug10 vs unaug @10',res[(res.test_rotation_deg==10)&(res.train_augmentation_max_deg==10)]),
 ('Aug10 clean penalty',cp[cp.train_augmentation_max_deg==10]),
 ('Baseline summary @10',bs[bs.test_rotation_deg==10]),
 ('Baseline comparisons',bcomp),
 ('Baseline clean cost',cleanbase),
 ('Invariance',isum)]:
    print('\n###',name,'\n',df.to_string(index=False))
