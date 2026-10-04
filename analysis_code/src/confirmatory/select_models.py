from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import MODEL_SELECTION_TASKS, PRIMARY_TASKS
from .data import load_study_data
from .folds import inner_splits_for_outer_train, load_outer_folds
from .metrics import composite_participant_rmse
from .modeling import fit_predict_family, grid_histgb, grid_ridge, grid_tcn
from .utils import load_json, project_root, save_json, set_global_seed


def candidate_grid(family: str, cfg: dict, seed: int, quick: bool):
    if family == "ridge":
        grid = list(grid_ridge(cfg["models"]["ridge"]))
        return grid[:1] if quick else grid
    if family == "histgb":
        grid = list(grid_histgb(cfg["models"]["histgb"], seed))
        if quick:
            grid = grid[:1]
            grid[0]["max_iter"] = 25
        return grid
    if family == "tcn":
        grid = list(grid_tcn(cfg["models"]["tcn"], quick=quick))
        return grid[:1] if quick else grid
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/confirmatory.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outer-fold", type=int, default=None, help="Run one outer fold only")
    args = ap.parse_args()
    root = project_root()
    cfg = load_json(root / args.config)
    set_global_seed(int(cfg["seed"]))
    d = load_study_data(root / cfg["data_file"])
    outer = load_outer_folds(root / "protocol/outer_folds.csv", d.subjects)
    outdir = root / "results/confirmatory/model_selection"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = {}
    fold_values = [args.outer_fold] if args.outer_fold else sorted(np.unique(outer))

    for ofold in fold_values:
        outer_train_idx = np.flatnonzero(outer != ofold)
        g_outer = d.subjects[outer_train_idx]
        family_best = {}
        for family in ("ridge", "histgb", "tcn"):
            best = (np.inf, None)
            for pi, params in enumerate(candidate_grid(family, cfg, int(cfg["seed"]) + ofold, args.quick)):
                scores = []
                for ifold, (itr_local, iva_local) in enumerate(
                    inner_splits_for_outer_train(g_outer, int(cfg["inner_splits"])), start=1
                ):
                    itr = outer_train_idx[itr_local]
                    iva = outer_train_idx[iva_local]
                    task_scores = []
                    for task_name in MODEL_SELECTION_TASKS:
                        task = PRIMARY_TASKS[task_name]
                        ytr = d.T[itr][:, task["targets"]]
                        yva = d.T[iva][:, task["targets"]]
                        # For TCN, use a participant-disjoint subset of inner-training as early-stopping validation.
                        val = None
                        if family == "tcn":
                            sub_splits = list(inner_splits_for_outer_train(d.subjects[itr], 3))
                            fit_local, stop_local = sub_splits[0]
                            fit_idx, stop_idx = itr[fit_local], itr[stop_local]
                            _, preliminary = fit_predict_family(
                                family, params,
                                d.X[fit_idx], d.T[fit_idx][:, task["targets"]], d.durations[fit_idx], d.speeds[fit_idx],
                                d.X[iva], d.durations[iva], d.speeds[iva], task["sensors"],
                                seed=int(cfg["seed"]) + ofold*1000 + ifold*100 + pi,
                                val=(d.X[stop_idx], d.T[stop_idx][:, task["targets"]], d.durations[stop_idx], d.speeds[stop_idx], d.subjects[stop_idx]),
                            )
                            refit_params = dict(params)
                            refit_params["max_epochs"] = max(1, int(preliminary.best_epoch or params.get("max_epochs", 150)))
                            refit_params["patience"] = refit_params["max_epochs"] + 1
                            pred, _ = fit_predict_family(
                                family, refit_params,
                                d.X[itr], ytr, d.durations[itr], d.speeds[itr],
                                d.X[iva], d.durations[iva], d.speeds[iva], task["sensors"],
                                seed=int(cfg["seed"]) + ofold*1000 + ifold*100 + pi + 50000,
                                val=None,
                            )
                        else:
                            pred, _ = fit_predict_family(
                                family, params,
                                d.X[itr], ytr, d.durations[itr], d.speeds[itr],
                                d.X[iva], d.durations[iva], d.speeds[iva], task["sensors"],
                                seed=int(cfg["seed"]) + ofold*1000 + ifold*100 + pi,
                            )
                        task_scores.append(composite_participant_rmse(yva, pred, d.subjects[iva]))
                    scores.append(float(np.mean(task_scores)))
                mean_score = float(np.mean(scores))
                rows.append({
                    "outer_fold": int(ofold), "family": family, "param_index": pi,
                    "params_json": json.dumps(params, sort_keys=True), "inner_composite_rmse": mean_score,
                })
                if mean_score < best[0]:
                    best = (mean_score, params)
            family_best[family] = {"score": best[0], "params": best[1]}
        chosen_family = min(family_best, key=lambda x: family_best[x]["score"])
        selected[str(ofold)] = {
            "family": chosen_family,
            "params": family_best[chosen_family]["params"],
            "all_family_best": family_best,
        }
        print("outer fold", ofold, "selected", chosen_family, family_best[chosen_family]["score"])

    pd.DataFrame(rows).to_csv(outdir / "inner_model_scores.csv", index=False)
    save_json(selected, outdir / "selected_models.json")

if __name__ == "__main__":
    main()
