from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SEED = 20260907
EXPECTED_SHA256 = "c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a"
SENSOR_PREFIXES = ("CHEST", "R_THIGH", "R_SHANK", "R_FOOT")
EXCLUDED_SUBJECTS = {"AB18"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def phase_normalize_window(a: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray, float, dict]:
    phase = np.asarray(a[cols.index("heel_strike")], dtype=float)
    drops = np.flatnonzero(np.diff(phase) < -50.0) + 1
    if len(drops) == 0:
        raise ValueError("No heel-strike phase reset")
    # The upstream segmentation places the focal heel strike about 40 samples into the window.
    reset = int(drops[np.argmin(np.abs(drops - 40))])

    cur_idx = np.arange(reset, len(phase))
    cur_max = float(np.max(phase[cur_idx]))
    # Fill any missing late-cycle phase with the immediately preceding cycle tail already present in the window.
    tail_idx = np.arange(0, reset)[phase[:reset] > cur_max]
    use_idx = np.concatenate([cur_idx, tail_idx])
    use_phase = np.concatenate([phase[cur_idx], phase[tail_idx]])
    order = np.argsort(use_phase)
    use_phase = use_phase[order]
    use_idx = use_idx[order]
    use_phase, uniq_pos = np.unique(use_phase, return_index=True)
    use_idx = use_idx[uniq_pos]

    if use_phase[0] > 1.5 or use_phase[-1] < 97.0:
        raise ValueError(f"Insufficient phase coverage: {use_phase[0]:.3f}..{use_phase[-1]:.3f}")

    q = np.linspace(0.0, 100.0, 101)
    X = np.empty((4, 101, 6), dtype=np.float32)
    for s, prefix in enumerate(SENSOR_PREFIXES):
        idx = [cols.index(f"{prefix}_Accel_{ax}") for ax in "XYZ"] + [cols.index(f"{prefix}_Gyro_{ax}") for ax in "XYZ"]
        for k, j in enumerate(idx):
            X[s, :, k] = np.interp(q, use_phase, a[j, use_idx]).astype(np.float32)

    knee = np.interp(q, use_phase, -a[cols.index("knee_angle_r"), use_idx])
    ankle = np.interp(q, use_phase, a[cols.index("ankle_angle_r"), use_idx])
    T = np.asarray([
        float(np.max(knee[:51])),
        float(np.max(knee[55:])),
        float(np.max(ankle[15:56])),
        float(np.min(ankle[45:76])),
    ], dtype=np.float32)

    dphase = np.diff(phase)
    positive = dphase[(dphase > 0) & (dphase < 10)]
    if len(positive) < 10:
        raise ValueError("Insufficient positive phase increments")
    phase_inc_pct_per_sample = float(np.median(positive))
    # Source file is 100 Hz after preprocessing: cycle duration = (100/inc samples) / 100 Hz = 1/inc seconds.
    duration = 1.0 / phase_inc_pct_per_sample

    info = {
        "reset_sample": reset,
        "n_phase_resets_in_window": int(len(drops)),
        "current_cycle_last_phase": cur_max,
        "phase_min_used": float(use_phase[0]),
        "phase_max_used": float(use_phase[-1]),
        "duration_s": float(duration),
    }
    return X, T, float(duration), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    src = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    digest = sha256(src)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected source SHA-256: {digest}")

    Xs, Ts, subj_names, cycle_ids, durations = [], [], [], [], []
    qc_rows = []

    with h5py.File(src, "r") as f:
        cols = json.loads(f.attrs["columns"])
        required = [
            "label", "heel_strike", "knee_angle_r", "ankle_angle_r",
            *[f"{p}_Accel_{ax}" for p in SENSOR_PREFIXES for ax in "XYZ"],
            *[f"{p}_Gyro_{ax}" for p in SENSOR_PREFIXES for ax in "XYZ"],
        ]
        missing = [c for c in required if c not in cols]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        relevant_idx = [cols.index(c) for c in required if c not in {"label", "heel_strike"}]

        for sid in sorted(f.keys()):
            data = np.asarray(f[sid][:], dtype=float)
            for ci, a in enumerate(data):
                row = {"subject": sid, "source_window": int(ci), "included": False, "reason": ""}
                if sid in EXCLUDED_SUBJECTS:
                    row["reason"] = "upstream_subject_qc_exclusion"
                    qc_rows.append(row)
                    continue
                if not np.all(a[cols.index("label")] == 0):
                    row["reason"] = "contains_turning_label"
                    qc_rows.append(row)
                    continue
                if not np.isfinite(a[relevant_idx]).all():
                    row["reason"] = "nonfinite_relevant_channel"
                    qc_rows.append(row)
                    continue
                try:
                    X, T, duration, info = phase_normalize_window(a, cols)
                except Exception as e:
                    row["reason"] = f"phase_normalization_failure:{type(e).__name__}:{e}"
                    qc_rows.append(row)
                    continue
                if not np.isfinite(X).all() or not np.isfinite(T).all() or not np.isfinite(duration):
                    row["reason"] = "nonfinite_after_normalization"
                    qc_rows.append(row)
                    continue
                row.update(info)
                row["included"] = True
                row["reason"] = "included"
                qc_rows.append(row)
                Xs.append(X)
                Ts.append(T)
                subj_names.append(sid)
                cycle_ids.append(ci)
                durations.append(duration)

    X = np.stack(Xs).astype(np.float32)
    T = np.stack(Ts).astype(np.float32)
    subject_names = np.asarray(subj_names, dtype="U8")
    unique_subjects = np.unique(subject_names)
    subject_to_int = {s: i for i, s in enumerate(unique_subjects)}
    subjects = np.asarray([subject_to_int[s] for s in subject_names], dtype=int)
    duration = np.asarray(durations, dtype=np.float32)
    speed = np.zeros(len(X), dtype=np.float32)
    cycle_ids = np.asarray(cycle_ids, dtype=int)

    rng = np.random.default_rng(SEED)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    fold_by_subject = {s: int(i % 5 + 1) for i, s in enumerate(shuffled)}
    folds = np.asarray([fold_by_subject[s] for s in subject_names], dtype=int)

    np.savez_compressed(
        outdir / "camargo_external.npz",
        X=X,
        T=T,
        subjects=subjects,
        subject_names=subject_names,
        durations=duration,
        speeds=speed,
        source_window=cycle_ids,
        outer_fold=folds,
        target_names=np.asarray([
            "knee_stance_flex_deg", "knee_swing_flex_deg",
            "ankle_dorsiflex_deg", "ankle_plantarflex_deg",
        ], dtype="U32"),
    )

    pd.DataFrame(
        [{"subject": s, "subject_int": subject_to_int[s], "outer_fold": fold_by_subject[s]} for s in unique_subjects]
    ).sort_values("outer_fold").to_csv(outdir / "folds.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(outdir / "qc_windows.csv", index=False)

    tdf = pd.DataFrame(T, columns=[
        "knee_stance_flex_deg", "knee_swing_flex_deg", "ankle_dorsiflex_deg", "ankle_plantarflex_deg"
    ])
    tdf.insert(0, "duration_s", duration)
    tdf.insert(0, "subject", subject_names)
    tdf.to_csv(outdir / "derived_targets.csv", index=False)

    summary = {
        "source_sha256": digest,
        "n_source_subjects": 20,
        "excluded_subjects": sorted(EXCLUDED_SUBJECTS),
        "n_included_subjects": int(len(unique_subjects)),
        "n_included_windows": int(len(X)),
        "n_excluded_windows": int(len(qc_rows) - len(X)),
        "shape_X": list(X.shape),
        "shape_T": list(T.shape),
        "duration_s": {"min": float(duration.min()), "median": float(np.median(duration)), "max": float(duration.max())},
        "target_mean_sd": {
            c: {"mean": float(tdf[c].mean()), "sd": float(tdf[c].std(ddof=1))}
            for c in tdf.columns if c not in {"subject", "duration_s"}
        },
    }
    (outdir / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
