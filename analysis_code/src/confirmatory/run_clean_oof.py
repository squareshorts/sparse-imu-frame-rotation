from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .constants import PRIMARY_TASKS
from .data import load_study_data
from .folds import inner_splits_for_outer_train, load_outer_folds
from .metrics import cycle_prediction_table, participant_metrics
from .modeling import fit_predict_family
from .utils import load_json, project_root, set_global_seed


def fit_outer_model(family, params, d, tr, te, task, seed, quick=False):
    params = dict(params)
    if family != "tcn":
        return fit_predict_family(
            family, params,
            d.X[tr], d.T[tr][:, task["targets"]], d.durations[tr], d.speeds[tr],
            d.X[te], d.durations[te], d.speeds[te], task["sensors"], seed=seed,
        )
    if quick:
        params["max_epochs"] = min(3, int(params.get("max_epochs", 3)))
        params["patience"] = 2
    fit_local, stop_local = next(inner_splits_for_outer_train(d.subjects[tr], 3))
    fit_idx, stop_idx = tr[fit_local], tr[stop_local]
    _, preliminary = fit_predict_family(
        family, params,
        d.X[fit_idx], d.T[fit_idx][:, task["targets"]], d.durations[fit_idx], d.speeds[fit_idx],
        d.X[te], d.durations[te], d.speeds[te], task["sensors"], seed=seed,
        val=(d.X[stop_idx], d.T[stop_idx][:, task["targets"]], d.durations[stop_idx], d.speeds[stop_idx], d.subjects[stop_idx]),
    )
    refit = dict(params)
    refit["max_epochs"] = max(1, int(preliminary.best_epoch or params["max_epochs"]))
    refit["patience"] = refit["max_epochs"] + 1
    return fit_predict_family(
        family, refit,
        d.X[tr], d.T[tr][:, task["targets"]], d.durations[tr], d.speeds[tr],
        d.X[te], d.durations[te], d.speeds[te], task["sensors"], seed=seed + 50000, val=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/confirmatory.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outer-fold", type=int, default=None)
    args = ap.parse_args()
    root = project_root(); cfg = load_json(root / args.config); set_global_seed(cfg["seed"])
    d = load_study_data(root / cfg["data_file"])
    outer = load_outer_folds(root / "protocol/outer_folds.csv", d.subjects)
    selected = load_json(root / "results/confirmatory/model_selection/selected_models.json")
    outdir = root / "results/confirmatory/clean_oof"; outdir.mkdir(parents=True, exist_ok=True)
    pred_rows=[]; part_rows=[]
    folds = [args.outer_fold] if args.outer_fold else sorted(np.unique(outer))
    for ofold in folds:
        tr=np.flatnonzero(outer!=ofold); te=np.flatnonzero(outer==ofold)
        sel=selected[str(ofold)]; family=sel["family"]; params=dict(sel["params"])
        for task_name,task in PRIMARY_TASKS.items():
            pred,_=fit_outer_model(family,params,d,tr,te,task,cfg["seed"]+ofold*100+len(task_name),quick=args.quick)
            cp=cycle_prediction_table(d.T[te][:,task["targets"]],pred,d.subjects[te],te,task["target_names"])
            cp["outer_fold"]=ofold;cp["task"]=task_name;cp["family"]=family
            pred_rows.append(cp)
            pm=participant_metrics(d.T[te][:,task["targets"]],pred,d.subjects[te],task["target_names"])
            pm["outer_fold"]=ofold;pm["task"]=task_name;pm["family"]=family
            part_rows.append(pm)
    pd.concat(pred_rows,ignore_index=True).to_csv(outdir/"cycle_predictions.csv",index=False)
    pd.concat(part_rows,ignore_index=True).to_csv(outdir/"participant_metrics.csv",index=False)

if __name__=="__main__": main()
