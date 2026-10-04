from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.confirmatory.features import engineered_features
from src.confirmatory.tabular_models import TabularWrapper
from src.confirmatory.metrics import bootstrap_mean_ci

SEED = 20260907
BOOT = 5000
PARAMS = {"alpha": 10.0}
SENSOR_PREFIXES = ("CHEST", "R_THIGH", "R_SHANK", "R_FOOT")

# Body masses from the public Camargo metadata used by StanfordMIMI/SSL_IMU.
BODY_MASS_KG = {
    "AB06": 74.8, "AB07": 55.3, "AB09": 63.5, "AB10": 83.9,
    "AB11": 77.1, "AB12": 86.2, "AB13": 59.0, "AB14": 58.4,
    "AB15": 96.2, "AB16": 55.8, "AB17": 61.2, "AB19": 68.0,
    "AB21": 58.1, "AB23": 76.8, "AB24": 72.6, "AB25": 52.2,
    "AB27": 68.0, "AB28": 62.1, "AB30": 77.0,
}

GRF_NAMES = (
    "vGRF_peak1_BW", "vGRF_midstance_BW", "vGRF_peak2_BW",
    "AP_braking_peak_mag_BW", "AP_propulsive_peak_mag_BW",
)
CONFIGS = {
    "thigh_shank": (1, 2),
    "shank_foot": (2, 3),
}


def phase_map(a: np.ndarray, cols: list[str]):
    ph = np.asarray(a[cols.index("heel_strike")], dtype=float)
    drops = np.flatnonzero(np.diff(ph) < -50.0) + 1
    if len(drops) == 0:
        raise ValueError("No phase reset")
    reset = int(drops[np.argmin(np.abs(drops - 40))])
    cur = np.arange(reset, len(ph))
    curmax = float(ph[cur].max())
    tail = np.arange(reset)[ph[:reset] > curmax]
    idx = np.concatenate([cur, tail])
    phase = np.concatenate([ph[cur], ph[tail]])
    order = np.argsort(phase)
    phase, idx = phase[order], idx[order]
    phase, u = np.unique(phase, return_index=True)
    idx = idx[u]
    return phase, idx


def derive_force_target(a: np.ndarray, cols: list[str], mass_kg: float) -> tuple[np.ndarray, dict]:
    phase, idx = phase_map(a, cols)
    q = np.linspace(0.0, 100.0, 101)
    # StanfordMIMI's Camargo mapping defines fy as vertical and fx as anterior-posterior.
    v = np.interp(q, phase, a[cols.index("fy"), idx]) / (mass_kg * 9.81)
    ap = np.interp(q, phase, a[cols.index("fx"), idx]) / (mass_kg * 9.81)

    contact = np.flatnonzero(v > 0.05)
    if len(contact) == 0:
        raise ValueError("No vGRF contact")
    groups = np.split(contact, np.where(np.diff(contact) > 1)[0] + 1)
    groups = [g for g in groups if len(g)]
    stance = max(groups, key=len)
    on, off = int(stance[0]), int(stance[-1])
    if not (on <= 12 and 35 <= off <= 80 and len(stance) >= 25):
        raise ValueError(f"Contact QC failed: on={on}, off={off}, n={len(stance)}")

    stv = v[on:off + 1]
    sta = ap[on:off + 1]
    n = len(stv)
    p1 = float(stv[:max(2, int(0.45 * n))].max())
    mid = float(stv[int(0.30 * n):max(int(0.70 * n), int(0.30 * n) + 1)].min())
    p2 = float(stv[int(0.55 * n):].max())
    h = max(2, int(0.50 * n))
    brake = float(np.max(np.abs(sta[:h])))
    prop = float(np.max(np.abs(sta[h:])))
    return np.asarray([p1, mid, p2, brake, prop], np.float32), {
        "contact_on_pct": on, "contact_off_pct": off, "contact_samples": int(len(stance)),
    }


def load_force_targets(h5_path: Path, prepared_path: Path):
    z = np.load(prepared_path, allow_pickle=True)
    subjects = z["subject_names"].astype(str)
    windows = z["source_window"].astype(int)
    targets, qc = [], []
    with h5py.File(h5_path, "r") as f:
        cols = json.loads(f.attrs["columns"])
        for sid, ci in zip(subjects, windows):
            t, info = derive_force_target(np.asarray(f[sid][ci], float), cols, BODY_MASS_KG[sid])
            targets.append(t)
            qc.append({"subject": sid, "source_window": int(ci), **info})
    return np.stack(targets).astype(np.float32), pd.DataFrame(qc)


def main(h5_path: Path, prepared_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(prepared_path, allow_pickle=True)
    X, G, names, D, S, folds = z["X"], z["subjects"], z["subject_names"].astype(str), z["durations"], z["speeds"], z["outer_fold"]
    Tgrf, force_qc = load_force_targets(h5_path, prepared_path)
    force_qc.to_csv(outdir / "force_contact_qc.csv", index=False)
    pd.DataFrame(Tgrf, columns=GRF_NAMES).assign(subject=names).to_csv(outdir / "force_targets.csv", index=False)

    cycle_rows = []
    participant_rows = []
    summary_rows = []

    for config, sensors in CONFIGS.items():
        pred_all = np.full_like(Tgrf, np.nan, dtype=np.float32)
        for fold in range(1, 6):
            tr = folds != fold
            te = folds == fold
            Ftr = engineered_features(X[tr], sensors, D[tr], S[tr])
            Fte = engineered_features(X[te], sensors, D[te], S[te])
            model = TabularWrapper("ridge", PARAMS).fit(Ftr, Tgrf[tr])
            pred_all[te] = model.predict(Fte)

        for i in range(len(Tgrf)):
            for j, target in enumerate(GRF_NAMES):
                cycle_rows.append({
                    "config": config, "subject": int(G[i]), "subject_name": names[i],
                    "outer_fold": int(folds[i]), "target": target,
                    "y_true": float(Tgrf[i, j]), "y_pred": float(pred_all[i, j]),
                })

        for j, target in enumerate(GRF_NAMES):
            e = pred_all[:, j] - Tgrf[:, j]
            summary_rows.append({
                "config": config, "target": target,
                "rmse_BW": float(np.sqrt(np.mean(e * e))),
                "mae_BW": float(np.mean(np.abs(e))),
                "bias_BW": float(np.mean(e)),
                "r2_oof": float(r2_score(Tgrf[:, j], pred_all[:, j])),
            })
            vals = []
            for sid in np.unique(G):
                ii = G == sid
                rmse = float(np.sqrt(np.mean((pred_all[ii, j] - Tgrf[ii, j]) ** 2)))
                vals.append(rmse)
                participant_rows.append({
                    "config": config, "subject": int(sid), "subject_name": str(names[np.flatnonzero(ii)[0]]),
                    "target": target, "rmse_BW": rmse, "n_windows": int(ii.sum()),
                })
            mean, lo, hi = bootstrap_mean_ci(np.asarray(vals), BOOT, SEED + len(summary_rows) * 101)
            summary_rows[-1].update({
                "mean_participant_rmse_BW": mean,
                "participant_ci95_low_BW": lo,
                "participant_ci95_high_BW": hi,
            })

    pd.DataFrame(cycle_rows).to_csv(outdir / "cycle_predictions.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(outdir / "participant_metrics.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    main(Path(a.h5), Path(a.prepared), Path(a.outdir))
