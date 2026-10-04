from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import PRIMARY_TASKS
from .data import load_study_data
from .features import magnitude_only_features
from .folds import load_outer_folds
from .orientation import perturb_by_subject
from .utils import load_json, project_root, set_global_seed


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/confirmatory.json");args=ap.parse_args()
    root=project_root();cfg=load_json(root/args.config);set_global_seed(cfg["seed"])
    d=load_study_data(root/cfg["data_file"]);outer=load_outer_folds(root/"protocol/outer_folds.csv",d.subjects)
    rows=[];invariance=[];rng=np.random.default_rng(cfg["seed"])
    for fold in sorted(np.unique(outer)):
        tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold)
        for task_name in ("knee_two","ankle_two"):
            task=PRIMARY_TASKS[task_name];Ytr=d.T[tr][:,task["targets"]];Yte=d.T[te][:,task["targets"]]
            # Duration/speed-only baseline.
            Ftr=np.column_stack([d.durations[tr],d.speeds[tr]]);Fte=np.column_stack([d.durations[te],d.speeds[te]])
            model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(Ftr,Ytr);p=model.predict(Fte)
            for j,t in enumerate(task["target_names"]):rows.append({"fold":fold,"task":task_name,"target":t,"control":"duration_speed_only","rmse":mean_squared_error(Yte[:,j],p[:,j])**0.5})

            # Magnitude-only orientation-invariant reference.
            Mtr=magnitude_only_features(d.X[tr],task["sensors"],d.durations[tr],d.speeds[tr]);Mte=magnitude_only_features(d.X[te],task["sensors"],d.durations[te],d.speeds[te])
            model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(Mtr,Ytr);p_clean=model.predict(Mte)
            Xrot,_=perturb_by_subject(d.X[te],d.subjects[te],task["sensors"],10.0,cfg["seed"]+fold*100+len(task_name),exact_angle=True)
            Mrot=magnitude_only_features(Xrot,task["sensors"],d.durations[te],d.speeds[te]);p_rot=model.predict(Mrot)
            invariance.append({"fold":fold,"task":task_name,"max_abs_feature_difference":float(np.max(np.abs(Mte-Mrot))),"max_abs_prediction_difference":float(np.max(np.abs(p_clean-p_rot)))})
            for j,t in enumerate(task["target_names"]):
                rows.append({"fold":fold,"task":task_name,"target":t,"control":"magnitude_only_clean","rmse":mean_squared_error(Yte[:,j],p_clean[:,j])**0.5})
                rows.append({"fold":fold,"task":task_name,"target":t,"control":"magnitude_only_10deg","rmse":mean_squared_error(Yte[:,j],p_rot[:,j])**0.5})

            # Target-shuffle negative control, training labels shuffled only.
            sh=Ytr.copy();rng.shuffle(sh,axis=0)
            p_shuffle=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(Mtr,sh).predict(Mte)
            for j,t in enumerate(task["target_names"]):rows.append({"fold":fold,"task":task_name,"target":t,"control":"target_shuffle","rmse":mean_squared_error(Yte[:,j],p_shuffle[:,j])**0.5})
    out=root/"results/confirmatory/statistics";out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"negative_controls.csv",index=False)
    pd.DataFrame(invariance).to_csv(out/"magnitude_invariance_check.csv",index=False)

if __name__=="__main__":main()
