from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    e = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else np.nan,
    }


def participant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    target_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    rows = []
    for sid in np.unique(subjects):
        idx = subjects == sid
        for j, target in enumerate(target_names):
            m = basic_metrics(y_true[idx, j], y_pred[idx, j])
            rows.append({"subject": int(sid), "target": target, "n_cycles": int(idx.sum()), **m})
    return pd.DataFrame(rows)


def cycle_prediction_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    cycle_indices: np.ndarray,
    target_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    rows = []
    for i in range(len(y_true)):
        for j, target in enumerate(target_names):
            rows.append({
                "cycle_index": int(cycle_indices[i]),
                "subject": int(subjects[i]),
                "target": target,
                "y_true": float(y_true[i, j]),
                "y_pred": float(y_pred[i, j]),
            })
    return pd.DataFrame(rows)


def composite_participant_rmse(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    for sid in np.unique(subjects):
        idx = subjects == sid
        e = y_pred[idx] - y_true[idx]
        vals.extend(np.sqrt(np.mean(e ** 2, axis=0)).tolist())
    return float(np.mean(vals))


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot = np.empty(reps)
    for i in range(reps):
        boot[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
