from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .constants import MODEL_SELECTION_TASKS, PRIMARY_TASKS
from .data import load_study_data
from .folds import load_outer_folds
from .metrics import cycle_prediction_table, participant_metrics
from .run_clean_oof import fit_outer_model
from .utils import load_json, project_root, set_global_seed


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/confirmatory.json");ap.add_argument("--quick",action="store_true");args=ap.parse_args()
    root=project_root();cfg=load_json(root/args.config);set_global_seed(cfg["seed"]);d=load_study_data(root/cfg["data_file"]);outer=load_outer_folds(root/"protocol/outer_folds.csv",d.subjects)
    selected=load_json(root/"results/confirmatory/model_selection/selected_models.json")
    out=root/"results/confirmatory/model_comparison";out.mkdir(parents=True,exist_ok=True)
    pr=[];pm=[]
    for fold in sorted(np.unique(outer)):
        tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold)
        for family in ("ridge","histgb","tcn"):
            params=selected[str(fold)]["all_family_best"][family]["params"]
            for task_name in MODEL_SELECTION_TASKS:
                task=PRIMARY_TASKS[task_name]
                pred,_=fit_outer_model(family,params,d,tr,te,task,cfg["seed"]+fold*1000+len(family)*10+len(task_name),quick=args.quick)
                x=cycle_prediction_table(d.T[te][:,task["targets"]],pred,d.subjects[te],te,task["target_names"])
                x["outer_fold"]=fold;x["family"]=family;x["task"]=task_name;pr.append(x)
                m=participant_metrics(d.T[te][:,task["targets"]],pred,d.subjects[te],task["target_names"])
                m["outer_fold"]=fold;m["family"]=family;m["task"]=task_name;pm.append(m)
    pred=pd.concat(pr,ignore_index=True);parts=pd.concat(pm,ignore_index=True)
    pred.to_csv(out/"cycle_predictions.csv",index=False);parts.to_csv(out/"participant_metrics.csv",index=False)
    parts.groupby(["family","task","target"],as_index=False).agg(rmse=("rmse","mean"),mae=("mae","mean"),n_subjects=("subject","nunique")).to_csv(out/"summary.csv",index=False)

if __name__=="__main__":main()
