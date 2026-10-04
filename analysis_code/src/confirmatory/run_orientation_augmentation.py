from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .constants import PRIMARY_TASKS
from .data import load_study_data
from .folds import inner_splits_for_outer_train, load_outer_folds
from .metrics import cycle_prediction_table, participant_metrics
from .modeling import fit_predict_family
from .orientation import make_augmented_training_set, perturb_by_subject
from .utils import load_json, project_root, set_global_seed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/confirmatory.json")
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--outer-fold",type=int,default=None)
    args=ap.parse_args()
    root=project_root();cfg=load_json(root/args.config);set_global_seed(cfg["seed"])
    d=load_study_data(root/cfg["data_file"])
    outer=load_outer_folds(root/"protocol/outer_folds.csv",d.subjects)
    selected=load_json(root/"results/confirmatory/model_selection/selected_models.json")
    outdir=root/"results/confirmatory/orientation_augmentation";outdir.mkdir(parents=True,exist_ok=True)
    pred_rows=[];part_rows=[];rot_rows=[]
    mc=int(cfg["quick"]["mc_realizations"] if args.quick else cfg["orientation"]["mc_realizations"])
    folds=[args.outer_fold] if args.outer_fold else sorted(np.unique(outer))

    for ofold in folds:
        tr=np.flatnonzero(outer!=ofold);te=np.flatnonzero(outer==ofold)
        family=selected[str(ofold)]["family"];base_params=dict(selected[str(ofold)]["params"])
        if args.quick and family=="tcn":
            base_params["max_epochs"]=int(cfg["quick"]["tcn_max_epochs"]);base_params["patience"]=int(cfg["quick"]["tcn_patience"])
        for task_name,task in PRIMARY_TASKS.items():
            for regime,max_aug in cfg["orientation"]["augmentation_regimes"].items():
                Xtr,Ytr,Gtr,Dtr,Str=make_augmented_training_set(
                    d.X[tr],d.T[tr][:,task["targets"]],d.subjects[tr],d.durations[tr],d.speeds[tr],
                    task["sensors"],float(max_aug),int(cfg["orientation"]["static_aug_copies"]),
                    seed=int(cfg["seed"])+ofold*10000+len(task_name)*100,
                )
                val=None;fit_X,fit_Y,fit_D,fit_S=Xtr,Ytr,Dtr,Str
                if family=="tcn":
                    # Early stopping subjects are selected from CLEAN outer-train subjects, then all augmented copies are partitioned by those subject IDs.
                    _, stop_local = next(inner_splits_for_outer_train(d.subjects[tr],3))
                    stop_subjects=np.unique(d.subjects[tr][stop_local])
                    is_stop=np.isin(Gtr,stop_subjects);is_fit=~is_stop
                    fit_X,fit_Y,fit_D,fit_S=Xtr[is_fit],Ytr[is_fit],Dtr[is_fit],Str[is_fit]
                    val=(Xtr[is_stop],Ytr[is_stop],Dtr[is_stop],Str[is_stop],Gtr[is_stop])
                # Train once per outer fold/task/regime; test multiple perturbation realizations.
                train_seed=int(cfg["seed"])+ofold*1000+len(task_name)*10
                clean_pred,model=fit_predict_family(
                    family,base_params,fit_X,fit_Y,fit_D,fit_S,
                    d.X[te],d.durations[te],d.speeds[te],task["sensors"],
                    seed=train_seed,val=val,
                )
                if family=="tcn":
                    # Early stopping determines epoch count only. Refit on every outer-training subject/copy.
                    refit_params=dict(base_params)
                    refit_params["max_epochs"]=max(1,int(model.best_epoch or base_params.get("max_epochs",150)))
                    refit_params["patience"]=refit_params["max_epochs"]+1
                    clean_pred,model=fit_predict_family(
                        family,refit_params,Xtr,Ytr,Dtr,Str,
                        d.X[te],d.durations[te],d.speeds[te],task["sensors"],
                        seed=train_seed+50000,val=None,
                    )
                # Evaluate 0-degree once; nonzero angles via deterministic MC.
                for angle in cfg["orientation"]["test_angles_deg"]:
                    reps=1 if float(angle)==0 else mc
                    for rep in range(reps):
                        if float(angle)==0:
                            Xte=d.X[te];records=[]
                        else:
                            pseed=int(cfg["seed"])+ofold*100000+len(task_name)*1000+int(round(float(angle)*10))*20+rep
                            Xte,records=perturb_by_subject(d.X[te],d.subjects[te],task["sensors"],float(angle),pseed,exact_angle=True)
                            for r in records:
                                r.update({"outer_fold":ofold,"task":task_name,"regime":regime,"test_angle_deg":float(angle),"realization":rep})
                            rot_rows.extend(records)
                        if float(angle)==0:
                            pred=clean_pred
                        else:
                            if family in ("ridge","histgb"):
                                from .features import engineered_features
                                Fte=engineered_features(Xte,task["sensors"],d.durations[te],d.speeds[te])
                                pred=model.predict(Fte)
                            else:
                                from .tcn import prepare_sequence
                                seq=prepare_sequence(Xte,task["sensors"])
                                scal=np.column_stack([d.durations[te],d.speeds[te]]).astype(np.float32)
                                pred=model.predict(seq,scal)
                        cp=cycle_prediction_table(d.T[te][:,task["targets"]],pred,d.subjects[te],te,task["target_names"])
                        cp["outer_fold"]=ofold;cp["task"]=task_name;cp["family"]=family;cp["training_regime"]=regime;cp["test_angle_deg"]=float(angle);cp["realization"]=rep
                        pred_rows.append(cp)
                        pm=participant_metrics(d.T[te][:,task["targets"]],pred,d.subjects[te],task["target_names"])
                        pm["outer_fold"]=ofold;pm["task"]=task_name;pm["family"]=family;pm["training_regime"]=regime;pm["test_angle_deg"]=float(angle);pm["realization"]=rep
                        part_rows.append(pm)
                        print("done",ofold,task_name,regime,angle,rep,flush=True)
    pd.concat(pred_rows,ignore_index=True).to_csv(outdir/"cycle_predictions.csv",index=False)
    pd.concat(part_rows,ignore_index=True).to_csv(outdir/"participant_metrics.csv",index=False)
    pd.DataFrame(rot_rows).to_csv(outdir/"rotation_parameters.csv",index=False)

if __name__=="__main__":main()
