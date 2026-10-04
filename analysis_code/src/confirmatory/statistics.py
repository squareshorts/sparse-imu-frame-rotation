from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import bootstrap_mean_ci
from .utils import load_json, project_root

PRIMARY_TWO_TASKS=("knee_two","ankle_two")
BEST_SINGLE_TASK={
    "knee_stance_flex_deg":"knee_thigh",
    "knee_swing_flex_deg":"knee_shank",
    "ankle_dorsiflex_deg":"ankle_foot",
    "ankle_plantarflex_deg":"ankle_foot",
}
TWO_TASK={
    "knee_stance_flex_deg":"knee_two",
    "knee_swing_flex_deg":"knee_two",
    "ankle_dorsiflex_deg":"ankle_two",
    "ankle_plantarflex_deg":"ankle_two",
}


def bh_fdr(pvals):
    p=np.asarray(pvals,float);n=len(p);order=np.argsort(p);ranked=p[order]
    q=ranked*n/np.arange(1,n+1);q=np.minimum.accumulate(q[::-1])[::-1];q=np.clip(q,0,1)
    out=np.empty(n);out[order]=q;return out


def collapse_mc(df: pd.DataFrame) -> pd.DataFrame:
    keys=["subject","target","task","family","training_regime","test_angle_deg"]
    return df.groupby(keys,as_index=False)["rmse"].mean()


def rescue_contrast(df, aug_regime, angle, cfg, seed, primary_only=True):
    q=df[df.task.isin(PRIMARY_TWO_TASKS)] if primary_only else df
    a=q[(q.training_regime=="clean")&(q.test_angle_deg==angle)][["subject","target","task","rmse"]].rename(columns={"rmse":"rmse_clean_train"})
    b=q[(q.training_regime==aug_regime)&(q.test_angle_deg==angle)][["subject","target","task","rmse"]].rename(columns={"rmse":"rmse_aug_train"})
    m=a.merge(b,on=["subject","target","task"],how="inner");rows=[]
    for (task,target),g in m.groupby(["task","target"]):
        diff=g.rmse_clean_train.to_numpy()-g.rmse_aug_train.to_numpy()
        mean,lo,hi=bootstrap_mean_ci(diff,int(cfg["bootstrap_reps"]),seed+len(rows))
        try:p=float(wilcoxon(diff,zero_method="wilcox").pvalue)
        except ValueError:p=1.0
        sd=np.std(diff,ddof=1);dz=float(mean/sd) if sd>0 else np.nan
        rows.append({"task":task,"target":target,"angle_deg":angle,"aug_regime":aug_regime,"n_subjects":len(g),"mean_rescue_rmse_deg":mean,"ci_low":lo,"ci_high":hi,"p_raw":p,"paired_dz":dz})
    out=pd.DataFrame(rows);out["p_fdr_bh"]=bh_fdr(out.p_raw.to_numpy());return out


def two_vs_single(df, angle, regime, cfg, seed):
    rows=[]
    for target,single_task in BEST_SINGLE_TASK.items():
        two_task=TWO_TASK[target]
        a=df[(df.task==single_task)&(df.target==target)&(df.training_regime==regime)&(df.test_angle_deg==angle)][["subject","rmse"]].rename(columns={"rmse":"single_rmse"})
        b=df[(df.task==two_task)&(df.target==target)&(df.training_regime==regime)&(df.test_angle_deg==angle)][["subject","rmse"]].rename(columns={"rmse":"two_rmse"})
        m=a.merge(b,on="subject");diff=m.single_rmse.to_numpy()-m.two_rmse.to_numpy()
        mean,lo,hi=bootstrap_mean_ci(diff,int(cfg["bootstrap_reps"]),seed+len(rows))
        try:p=float(wilcoxon(diff,zero_method="wilcox").pvalue)
        except ValueError:p=1.0
        rows.append({"target":target,"angle_deg":angle,"training_regime":regime,"two_task":two_task,"single_task":single_task,"n_subjects":len(m),"mean_two_sensor_advantage_deg":mean,"ci_low":lo,"ci_high":hi,"p_raw":p})
    out=pd.DataFrame(rows);out["p_fdr_bh"]=bh_fdr(out.p_raw.to_numpy());return out


def clean_penalty(df, cfg, seed):
    q=df[(df.task.isin(PRIMARY_TWO_TASKS))&(df.test_angle_deg==0)]
    base=q[q.training_regime=="clean"][["subject","task","target","rmse"]].rename(columns={"rmse":"clean_rmse"})
    rows=[]
    for regime in ("aug_5deg","aug_10deg"):
        a=q[q.training_regime==regime][["subject","task","target","rmse"]].rename(columns={"rmse":"aug_rmse"})
        m=base.merge(a,on=["subject","task","target"])
        for (task,target),g in m.groupby(["task","target"]):
            diff=g.aug_rmse.to_numpy()-g.clean_rmse.to_numpy()
            rel=100*diff/np.maximum(g.clean_rmse.to_numpy(),1e-9)
            mean,lo,hi=bootstrap_mean_ci(diff,int(cfg["bootstrap_reps"]),seed+len(rows))
            rows.append({"training_regime":regime,"task":task,"target":target,"n_subjects":len(g),"mean_clean_penalty_deg":mean,"ci_low":lo,"ci_high":hi,"mean_relative_penalty_pct":float(np.mean(rel)),"material_penalty":bool((mean>0.5) or (np.mean(rel)>10.0))})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/confirmatory.json");ap.add_argument("--quick",action="store_true");args=ap.parse_args()
    root=project_root();cfg=load_json(root/args.config)
    if args.quick:cfg["bootstrap_reps"]=int(cfg["quick"]["bootstrap_reps"])
    df=pd.read_csv(root/"results/confirmatory/orientation_augmentation/participant_metrics.csv");c=collapse_mc(df)
    outdir=root/"results/confirmatory/statistics";outdir.mkdir(parents=True,exist_ok=True)

    primary=pd.concat([rescue_contrast(c,"aug_5deg",5.0,cfg,cfg["seed"]+1,True),rescue_contrast(c,"aug_10deg",10.0,cfg,cfg["seed"]+2,True)],ignore_index=True)
    primary.to_csv(outdir/"augmentation_rescue_primary.csv",index=False)
    secondary=pd.concat([rescue_contrast(c,"aug_5deg",5.0,cfg,cfg["seed"]+3,False),rescue_contrast(c,"aug_10deg",10.0,cfg,cfg["seed"]+4,False)],ignore_index=True)
    secondary.to_csv(outdir/"augmentation_rescue_all_configs_secondary.csv",index=False)
    tvs=pd.concat([two_vs_single(c,5.0,"aug_5deg",cfg,cfg["seed"]+5),two_vs_single(c,10.0,"aug_10deg",cfg,cfg["seed"]+6)],ignore_index=True)
    tvs.to_csv(outdir/"two_vs_best_single_under_misalignment.csv",index=False)
    clean_penalty(c,cfg,cfg["seed"]+7).to_csv(outdir/"clean_accuracy_penalty.csv",index=False)

    base=c[c.test_angle_deg==0][["subject","target","task","training_regime","rmse"]].rename(columns={"rmse":"rmse_at_0"})
    deg=c.merge(base,on=["subject","target","task","training_regime"])
    deg["degradation_deg"]=deg.rmse-deg.rmse_at_0;deg["relative_degradation_pct"]=(deg.rmse/deg.rmse_at_0-1)*100
    deg.to_csv(outdir/"participant_degradation.csv",index=False)
    summary=deg.groupby(["task","target","training_regime","test_angle_deg"],as_index=False).agg(rmse=("rmse","mean"),degradation_deg=("degradation_deg","mean"),relative_degradation_pct=("relative_degradation_pct","mean"),n_subjects=("subject","nunique"))
    summary.to_csv(outdir/"robustness_summary.csv",index=False)
    print(primary.to_string(index=False))

if __name__=="__main__":main()
