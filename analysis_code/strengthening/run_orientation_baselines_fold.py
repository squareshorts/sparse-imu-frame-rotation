from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.confirmatory.constants import PRIMARY_TASKS
from src.confirmatory.data import load_study_data
from src.confirmatory.features import engineered_features
from src.confirmatory.folds import load_outer_folds
from src.confirmatory.orientation import perturb_by_subject
from src.confirmatory.tabular_models import TabularWrapper

ANGLES = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0], dtype=float)
MC = 50
SEED = 20260907
TASKS = ("knee_two", "ankle_two")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if (not np.isfinite(n)) or n < 1e-10:
        raise ValueError("Degenerate axis")
    return v / n


def _phase_signature(block: np.ndarray) -> np.ndarray:
    """Participant/sensor gait signature, normalized separately for accel and gyro."""
    a = np.asarray(block[..., :3], dtype=float).mean(axis=0)
    w = np.asarray(block[..., 3:], dtype=float).mean(axis=0)
    sa = float(np.sqrt(np.mean(a * a)))
    sw = float(np.sqrt(np.mean(w * w)))
    a = a / max(sa, 1e-9)
    w = w / max(sw, 1e-9)
    return np.vstack([a, w])


def _kabsch_row(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return proper R minimizing ||X R - Y||_F for row-vector 3-D data."""
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def build_training_templates(X: np.ndarray, subjects: np.ndarray, sensor_ids: tuple[int, ...]) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for sensor in sensor_ids:
        signatures = []
        for sid in np.unique(subjects):
            signatures.append(_phase_signature(X[subjects == sid, sensor]))
        templates[int(sensor)] = np.mean(signatures, axis=0)
    return templates


def template_register(
    X: np.ndarray,
    subjects: np.ndarray,
    sensor_ids: tuple[int, ...],
    templates: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Training-template SO(3) registration.

    One static rotation is estimated per participant and sensor from the complete
    phase-normalized gait signature, without endpoint labels, and applied to both
    accelerometer and gyroscope vectors.
    """
    out = X.copy()
    for sid in np.unique(subjects):
        ii = np.flatnonzero(subjects == sid)
        for sensor in sensor_ids:
            V = _phase_signature(X[ii, sensor])
            R = _kabsch_row(V, templates[int(sensor)])
            b = X[ii, sensor]
            out[ii, sensor, :, :3] = b[..., :3] @ R
            out[ii, sensor, :, 3:] = b[..., 3:] @ R
    return out


def functional_frame(block: np.ndarray) -> np.ndarray:
    """
    Gait-functional sensor-to-segment frame.

    The mean specific-force vector defines the longitudinal/gravity axis. The
    dominant gyroscope covariance direction, projected perpendicular to gravity,
    defines a functional transverse axis. Its sign is fixed using the phase-locked
    mean angular-velocity excursion; the third axis completes a right-handed frame.
    """
    A = np.asarray(block[..., :3], dtype=float)
    W = np.asarray(block[..., 3:], dtype=float)

    z = _unit(A.reshape(-1, 3).mean(axis=0))
    wf = W.reshape(-1, 3)
    wc = wf - wf.mean(axis=0)
    C = wc.T @ wc / max(1, len(wc) - 1)
    vals, vecs = np.linalg.eigh(C)

    x = None
    for j in np.argsort(vals)[::-1]:
        cand = vecs[:, j]
        cand = cand - z * np.dot(cand, z)
        if np.linalg.norm(cand) > 1e-6:
            x = _unit(cand)
            break
    if x is None:
        raise ValueError("Degenerate gyroscope covariance")

    phase = W.mean(axis=0) @ x
    j = int(np.argmax(np.abs(phase)))
    if phase[j] < 0:
        x = -x

    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    return np.stack([x, y, z], axis=0)


def functional_register(X: np.ndarray, subjects: np.ndarray, sensor_ids: tuple[int, ...]) -> np.ndarray:
    out = X.copy()
    for sid in np.unique(subjects):
        ii = np.flatnonzero(subjects == sid)
        for sensor in sensor_ids:
            R = functional_frame(X[ii, sensor])
            b = X[ii, sensor]
            out[ii, sensor, :, :3] = b[..., :3] @ R.T
            out[ii, sensor, :, 3:] = b[..., 3:] @ R.T
    return out


def subject_rmses(y: np.ndarray, p: np.ndarray, g: np.ndarray, names: tuple[str, ...]) -> list[tuple[int, str, float]]:
    rows = []
    for sid in np.unique(g):
        ii = g == sid
        for j, name in enumerate(names):
            rows.append((int(sid), name, float(np.sqrt(np.mean((p[ii, j] - y[ii, j]) ** 2)))))
    return rows


def main(fold: int) -> None:
    t0 = time.time()
    d = load_study_data(ROOT / "data/processed/strict_parameters.npz")
    outer = load_outer_folds(ROOT / "protocol/outer_folds.csv", d.subjects)
    selected = json.loads((ROOT / "results/confirmatory/model_selection/selected_models.json").read_text())
    outdir = ROOT / "strengthening" / "orientation_baselines"
    outdir.mkdir(parents=True, exist_ok=True)

    ofold = int(fold)
    tr = np.flatnonzero(outer != ofold)
    te = np.flatnonzero(outer == ofold)
    family = selected[str(ofold)]["family"]
    params = selected[str(ofold)]["params"]
    if family not in ("ridge", "histgb"):
        raise RuntimeError(f"This baseline runner expects a tabular selected family, got {family!r}")

    accum: dict[tuple[str, str, str, float, int], tuple[float, int]] = {}
    inv_rows: list[dict] = []
    fit_rows: list[dict] = []

    for task_name in TASKS:
        task = PRIMARY_TASKS[task_name]
        sensor_ids = task["sensors"]
        ytr = d.T[tr][:, task["targets"]]
        yte = d.T[te][:, task["targets"]]
        gte = d.subjects[te]

        templates = build_training_templates(d.X[tr], d.subjects[tr], sensor_ids)

        clean_registered: dict[str, np.ndarray] = {
            "template_registration": template_register(d.X[te], gte, sensor_ids, templates),
            "gait_functional_calibration": functional_register(d.X[te], gte, sensor_ids),
        }
        train_registered: dict[str, np.ndarray] = {
            "template_registration": template_register(d.X[tr], d.subjects[tr], sensor_ids, templates),
            "gait_functional_calibration": functional_register(d.X[tr], d.subjects[tr], sensor_ids),
        }

        models = {}
        for method, Xtr_reg in train_registered.items():
            Ftr = engineered_features(Xtr_reg, sensor_ids, d.durations[tr], d.speeds[tr])
            models[method] = TabularWrapper(family, params).fit(Ftr, ytr)
            fit_rows.append({
                "outer_fold": ofold,
                "task": task_name,
                "method": method,
                "family": family,
                "params_json": json.dumps(params, sort_keys=True),
                "n_train_rows": int(len(tr)),
                "feature_dim": int(Ftr.shape[1]),
            })

        for test_ang in ANGLES:
            reps = 1 if test_ang == 0 else MC
            for rep in range(reps):
                if test_ang == 0:
                    Xp = d.X[te]
                else:
                    # Same perturbation seed schedule as the full augmentation matrix.
                    pseed = (
                        SEED
                        + ofold * 1_000_000
                        + len(task_name) * 10_000
                        + int(round(float(test_ang) * 10)) * 100
                        + rep
                    )
                    Xp, _ = perturb_by_subject(
                        d.X[te], gte, sensor_ids, float(test_ang), pseed, exact_angle=True
                    )

                for method in ("template_registration", "gait_functional_calibration"):
                    if method == "template_registration":
                        Xreg = template_register(Xp, gte, sensor_ids, templates)
                    else:
                        Xreg = functional_register(Xp, gte, sensor_ids)

                    clean_reg = clean_registered[method]
                    max_abs = max(
                        float(np.max(np.abs(Xreg[:, s] - clean_reg[:, s])))
                        for s in sensor_ids
                    )
                    inv_rows.append({
                        "outer_fold": ofold,
                        "task": task_name,
                        "method": method,
                        "test_rotation_deg": float(test_ang),
                        "realization": int(rep),
                        "max_abs_registered_signal_difference_vs_clean": max_abs,
                    })

                    Fte = engineered_features(Xreg, sensor_ids, d.durations[te], d.speeds[te])
                    pred = models[method].predict(Fte)
                    for sid, target, rmse in subject_rmses(yte, pred, gte, task["target_names"]):
                        key = (method, task_name, target, float(test_ang), sid)
                        s, n = accum.get(key, (0.0, 0))
                        accum[key] = (s + rmse, n + 1)

            print("fold", ofold, "task", task_name, "test", test_ang, "done", flush=True)

    rows = []
    for (method, task, target, angle, sid), (s, n) in accum.items():
        rows.append({
            "outer_fold": ofold,
            "method": method,
            "task": task,
            "target": target,
            "test_rotation_deg": angle,
            "subject": sid,
            "rmse_deg": s / n,
            "mc_reps": n,
        })

    pd.DataFrame(rows).to_csv(outdir / f"participant_metrics_fold{ofold}.csv", index=False)
    pd.DataFrame(inv_rows).to_csv(outdir / f"invariance_fold{ofold}.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(outdir / f"fit_log_fold{ofold}.csv", index=False)
    print("elapsed", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    main(args.fold)
