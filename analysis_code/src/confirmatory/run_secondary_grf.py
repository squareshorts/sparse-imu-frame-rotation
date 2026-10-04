from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .data import load_study_data
from .folds import inner_splits_for_outer_train, load_outer_folds
from .metrics import basic_metrics, cycle_prediction_table, participant_metrics
from .modeling import fit_predict_family
from .utils import load_json, project_root, set_global_seed

GRF_TASKS={
    "vgrf_peak1_pelvis":{"sensors":(0,),"targets":(0,),"target_names":("vGRF_peak1_BW",)},
    "vgrf_midstance_thigh":{"sensors":(1,),"targets":(1,),"target_names":("vGRF_midstance_BW",)},
    "vgrf_peak2_foot":{"sensors":(3,),"targets":(2,),"target_names":("vGRF_peak2_BW",)},
    "ap_braking_pelvis":{"sensors":(0,),"targets":(3,),"target_names":("AP_braking_peak_mag_BW",)},
    "ap_propulsive_foot":{"sensors":(3,),"targets":(4,),"target_names":("AP_propulsive_peak_mag_BW",)},
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/confirmatory.json")
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    root=project_root();cfg=load_json(root/args.config);set_global_seed(cfg["seed"])
    d=load_study_data(root/cfg["data_file"])
    outer=load_outer_folds(root/"protocol/outer_folds.csv",d.subjects)
    sel=load_json(root/"results/confirmatory/model_selection/selected_models.json")
    out=root/"results/confirmatory/secondary_grf";out.mkdir(parents=True,exist_ok=True)
    pr=[];pm=[]
    for fold in sorted(np.unique(outer)):
        tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold)
        family=sel[str(fold)]["family"];params=dict(sel[str(fold)]["params"])
        for task_name,task in GRF_TASKS.items():
            if family=="tcn":
                if args.quick:
                    params["max_epochs"]=3;params["patience"]=2
                fit_local,stop_local=next(inner_splits_for_outer_train(d.subjects[tr],3))
                fit_idx,stop_idx=tr[fit_local],tr[stop_local]
                _,pre=fit_predict_family(
                    family,params,
                    d.X[fit_idx],d.T[fit_idx][:,task["targets"]],d.durations[fit_idx],d.speeds[fit_idx],
                    d.X[te],d.durations[te],d.speeds[te],task["sensors"],
                    cfg["seed"]+fold*100+len(task_name),
                    val=(d.X[stop_idx],d.T[stop_idx][:,task["targets"]],d.durations[stop_idx],d.speeds[stop_idx],d.subjects[stop_idx]),
                )
                p2=dict(params);p2["max_epochs"]=max(1,int(pre.best_epoch or params["max_epochs"]));p2["patience"]=p2["max_epochs"]+1
                pred,_=fit_predict_family(
                    family,p2,d.X[tr],d.T[tr][:,task["targets"]],d.durations[tr],d.speeds[tr],
                    d.X[te],d.durations[te],d.speeds[te],task["sensors"],cfg["seed"]+fold*100+len(task_name)+50000,
                )
            else:
                pred,_=fit_predict_family(
                    family,params,d.X[tr],d.T[tr][:,task["targets"]],d.durations[tr],d.speeds[tr],
                    d.X[te],d.durations[te],d.speeds[te],task["sensors"],cfg["seed"]+fold*100+len(task_name),
                )
            x=cycle_prediction_table(d.T[te][:,task["targets"]],pred,d.subjects[te],te,task["target_names"])
            x["outer_fold"]=fold;x["task"]=task_name;x["family"]=family;pr.append(x)
            m=participant_metrics(d.T[te][:,task["targets"]],pred,d.subjects[te],task["target_names"])
            m["outer_fold"]=fold;m["task"]=task_name;m["family"]=family;pm.append(m)
    pred=pd.concat(pr,ignore_index=True);parts=pd.concat(pm,ignore_index=True)
    pred.to_csv(out/"cycle_predictions.csv",index=False);parts.to_csv(out/"participant_metrics.csv",index=False)
    rows=[]
    for target,g in pred.groupby("target"):
        rows.append({"target":target,**basic_metrics(g.y_true.to_numpy(),g.y_pred.to_numpy())})
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)

if __name__=="__main__":main()
