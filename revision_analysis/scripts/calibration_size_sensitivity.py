"""Disjoint calibration-size sensitivity for submitted frame normalizers.

Uses the submitted feature, perturbation, and frame-estimation functions. Only the
new calibration/evaluation split and application of a calibration-derived frame to
separate evaluation records are implemented here. Outputs stay in the revision area.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FINAL_CODE = ROOT / "submitted_final/analysis/analysis_code"
sys.dont_write_bytecode = True  # Keep the submitted source tree unchanged.
sys.path.insert(0, str(FINAL_CODE))

from src.confirmatory.constants import PRIMARY_TASKS  # noqa: E402
from src.confirmatory.data import load_study_data  # noqa: E402
from src.confirmatory.features import engineered_features  # noqa: E402
from src.confirmatory.folds import load_outer_folds  # noqa: E402
from src.confirmatory.orientation import perturb_by_subject  # noqa: E402
from strengthening.run_orientation_baselines_fold import (  # noqa: E402
    _kabsch_row,
    _phase_signature,
    build_training_templates,
    functional_frame,
    functional_register,
    template_register,
)


OUT = ROOT / "results/revision_2026_10"
SEED = 20260907
SUBSET_SEED_OFFSET = 30_000_000
BOOT_SEED_OFFSET = 40_000_000
BOOT = 5000
ALPHA = 100.0
METHODS = ("template_registration", "gait_functional_calibration")
TASK_ORDER = ("knee_two", "ankle_two")
PHYSICAL_TASKS = {
    "knee_two": {
        "sensors": (1, 2),
        "targets": (0, 1),
        "target_names": ("knee_stance_flex_deg", "knee_swing_flex_deg"),
    },
    "ankle_two": {
        "sensors": (2, 3),
        "targets": (2, 3),
        "target_names": ("ankle_dorsiflex_deg", "ankle_plantarflex_deg"),
    },
}


@dataclass(frozen=True)
class InputData:
    X: np.ndarray
    T: np.ndarray
    G: np.ndarray
    D: np.ndarray
    S: np.ndarray
    F: np.ndarray
    tasks: dict
    calibration_sizes: tuple[int, ...]


class WeightedRidge:
    """Same scaler/Ridge fit as final run_rotation_matrix_v2.py:25-34."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "WeightedRidge":
        w = np.ones(len(X), float)
        self.scaler.fit(X, sample_weight=w)
        Z = self.scaler.transform(X)
        self.model.fit(Z, Y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(X)), dtype=np.float32)


def load_inputs(cohort: str) -> InputData:
    if cohort == "virtual":
        d = load_study_data(ROOT / "data/processed/strict_parameters.npz")
        folds = load_outer_folds(FINAL_CODE / "protocol/outer_folds.csv", d.subjects)
        if (len(d.subjects), len(np.unique(d.subjects))) != (747, 138):
            raise RuntimeError("Virtual cohort does not match final 747-cycle analysis")
        return InputData(
            d.X, d.T, d.subjects, d.durations, d.speeds, folds,
            {t: PRIMARY_TASKS[t] for t in TASK_ORDER}, (1, 3),
        )

    if cohort == "physical_camargo":
        p = OUT / "camargo_prepared/camargo_external.npz"
        with np.load(p, allow_pickle=True) as z:
            d = {key: z[key] for key in ("X", "T", "subjects", "durations", "speeds", "outer_fold")}
        if (len(d["subjects"]), len(np.unique(d["subjects"]))) != (248, 19):
            raise RuntimeError("Physical cohort does not match final 248-window analysis")
        return InputData(
            d["X"], d["T"], d["subjects"], d["durations"],
            d["speeds"], d["outer_fold"], PHYSICAL_TASKS, (1, 3, 5),
        )
    raise ValueError(cohort)


def disjoint_indices(
    g_test: np.ndarray, sizes: tuple[int, ...]
) -> dict[int, tuple[dict[int, np.ndarray], np.ndarray]]:
    """One stable participant shuffle; first k calibration records form nested sets."""
    order: dict[int, np.ndarray] = {}
    for sid in np.unique(g_test):
        ix = np.flatnonzero(g_test == sid)
        rng = np.random.default_rng(SEED + SUBSET_SEED_OFFSET + int(sid))
        order[int(sid)] = rng.permutation(ix)

    selections = {}
    for k in sizes:
        calibration = {sid: ix[:k] for sid, ix in order.items() if len(ix) > k}
        evaluation = np.sort(np.concatenate([ix[k:] for ix in order.values() if len(ix) > k]))
        if not calibration or not len(evaluation):
            raise RuntimeError(f"No eligible participants for k={k}")
        c_all = np.concatenate(list(calibration.values()))
        if np.intersect1d(c_all, evaluation).size:
            raise AssertionError("Calibration/evaluation overlap")
        selections[k] = calibration, evaluation
    return selections


def apply_from_disjoint_calibration(
    X: np.ndarray,
    G: np.ndarray,
    sensor_ids: tuple[int, ...],
    templates: dict[int, np.ndarray],
    method: str,
    calibration: dict[int, np.ndarray],
    evaluation: np.ndarray,
) -> np.ndarray:
    """Estimate each final-method frame on calibration records; apply to evaluation."""
    out = X[evaluation].copy()
    eval_subjects = G[evaluation]
    for sid, cidx in calibration.items():
        positions = np.flatnonzero(eval_subjects == sid)
        eidx = evaluation[positions]
        if len(positions) == 0 or np.intersect1d(cidx, eidx).size:
            raise AssertionError("Invalid disjoint subset")
        for sensor in sensor_ids:
            if method == "template_registration":
                signature = _phase_signature(X[cidx, sensor])
                R = _kabsch_row(signature, templates[int(sensor)])
                out[positions, sensor, :, :3] = X[eidx, sensor, :, :3] @ R
                out[positions, sensor, :, 3:] = X[eidx, sensor, :, 3:] @ R
            elif method == "gait_functional_calibration":
                R = functional_frame(X[cidx, sensor])
                out[positions, sensor, :, :3] = X[eidx, sensor, :, :3] @ R.T
                out[positions, sensor, :, 3:] = X[eidx, sensor, :, 3:] @ R.T
            else:
                raise ValueError(method)
    return out


def perturb_seed(cohort: str, fold: int, task_name: str, task_index: int, rep: int) -> int:
    if cohort == "virtual":
        return SEED + fold * 3_000_000 + task_index * 100_000 + 100 * 100 + rep
    return SEED + fold * 1_000_000 + len(task_name) * 10_000 + 100 * 100 + rep


def evaluate(cohort: str, mc: int, folds_to_run: tuple[int, ...], tag: str) -> Path:
    d = load_inputs(cohort)
    accumulator: dict[tuple, list[float]] = {}
    eval_sizes: dict[tuple, int] = {}

    for fold in folds_to_run:
        tr = np.flatnonzero(d.F != fold)
        te = np.flatnonzero(d.F == fold)
        Xte, Gte, Yte = d.X[te], d.G[te], d.T[te]
        subsets = disjoint_indices(Gte, d.calibration_sizes)
        for task_index, task_name in enumerate(TASK_ORDER):
            task = d.tasks[task_name]
            sensors = task["sensors"]
            templates = build_training_templates(d.X[tr], d.G[tr], sensors)
            models = {}
            for method in METHODS:
                Xtr = (
                    template_register(d.X[tr], d.G[tr], sensors, templates)
                    if method == "template_registration"
                    else functional_register(d.X[tr], d.G[tr], sensors)
                )
                Ftr = engineered_features(Xtr, sensors, d.D[tr], d.S[tr])
                models[method] = WeightedRidge(ALPHA).fit(Ftr, d.T[tr][:, task["targets"]])

            for angle in (0.0, 10.0):
                reps = 1 if angle == 0 else mc
                for rep in range(reps):
                    if angle == 0:
                        Xp = Xte
                    else:
                        Xp, _ = perturb_by_subject(
                            Xte, Gte, sensors, angle,
                            perturb_seed(cohort, fold, task_name, task_index, rep),
                            exact_angle=True,
                        )
                    for k, (calibration, evaluation) in subsets.items():
                        g_eval = Gte[evaluation]
                        for method in METHODS:
                            Xreg = apply_from_disjoint_calibration(
                                Xp, Gte, sensors, templates, method, calibration, evaluation
                            )
                            Fte = engineered_features(
                                Xreg, sensors, d.D[te][evaluation], d.S[te][evaluation]
                            )
                            y = Yte[evaluation][:, task["targets"]]
                            e2 = (models[method].predict(Fte) - y) ** 2
                            for sid in calibration:
                                ix = g_eval == sid
                                n = int(ix.sum())
                                for j, target in enumerate(task["target_names"]):
                                    key = (method, k, task_name, target, angle, int(sid), fold)
                                    if key not in accumulator:
                                        accumulator[key] = [0.0, 0.0]
                                        eval_sizes[key] = n
                                    elif eval_sizes[key] != n:
                                        raise AssertionError("Evaluation count changed across realizations")
                                    accumulator[key][0] += float(e2[ix, j].sum())
                                    accumulator[key][1] += n
                    if angle == 10.0 and (rep + 1) % 10 == 0:
                        print(cohort, "fold", fold, task_name, "rotation", rep + 1, "/", reps, flush=True)
                print(cohort, "fold", fold, task_name, "angle", angle, "done", flush=True)

    rows = []
    for key, (sumsq, count) in accumulator.items():
        method, k, task, target, angle, sid, fold = key
        rows.append(
            {
                "cohort": cohort,
                "method": method,
                "calibration_size_k": k,
                "task": task,
                "target": target,
                "test_rotation_deg": angle,
                "subject": sid,
                "outer_fold": fold,
                "evaluation_records": eval_sizes[key],
                "pooled_error_count": int(count),
                "rmse_deg": float(np.sqrt(sumsq / count)),
            }
        )
    out = OUT / f"calibration_size_participants_{cohort}_{tag}.csv"
    pd.DataFrame(rows).sort_values(
        ["method", "calibration_size_k", "task", "target", "test_rotation_deg", "subject"]
    ).to_csv(out, index=False)
    print("saved", out, "participant rows", len(rows), flush=True)
    return out


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    resampled = np.empty(BOOT, float)
    for offset in range(0, BOOT, 500):
        n = min(500, BOOT - offset)
        index = rng.integers(0, len(values), size=(n, len(values)))
        resampled[offset : offset + n] = values[index].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(resampled, 0.025)),
        float(np.quantile(resampled, 0.975)),
    )


def summarize() -> None:
    frames = []
    for cohort in ("virtual", "physical_camargo"):
        p = OUT / f"calibration_size_participants_{cohort}_full.csv"
        frames.append(pd.read_csv(p))
    participant = pd.concat(frames, ignore_index=True)
    rows = []
    group_columns = [
        "cohort", "method", "calibration_size_k", "task", "target", "test_rotation_deg"
    ]
    for i, (keys, group) in enumerate(participant.groupby(group_columns, sort=True)):
        cohort, method, k, task, target, angle = keys
        mean, lo, hi = bootstrap_mean(
            group.rmse_deg.to_numpy(), SEED + BOOT_SEED_OFFSET + i * 37
        )
        expected = 138 if cohort == "virtual" else 19
        rows.append(
            {
                "cohort": cohort,
                "method": method,
                "calibration_size_k": int(k),
                "task": task,
                "target": target,
                "test_rotation_deg": angle,
                "mean_participant_rmse_deg": mean,
                "ci95_low_deg": lo,
                "ci95_high_deg": hi,
                "n_participants": int(group.subject.nunique()),
                "participants_lost_vs_original": expected - int(group.subject.nunique()),
                "n_scored_records": int(group.evaluation_records.sum()),
                "mc_realizations": 1 if angle == 0 else 50,
                "bootstrap_reps": BOOT,
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary) != 80:
        raise AssertionError(f"Expected 80 endpoint rows, got {len(summary)}")
    for (cohort, k), expected in {
        ("virtual", 1): (136, 609),
        ("virtual", 3): (113, 348),
        ("physical_camargo", 1): (19, 229),
        ("physical_camargo", 3): (19, 191),
        ("physical_camargo", 5): (19, 153),
    }.items():
        q = summary[(summary.cohort == cohort) & (summary.calibration_size_k == k)]
        if not ((q.n_participants == expected[0]) & (q.n_scored_records == expected[1])).all():
            raise AssertionError(f"Unexpected participant/evaluation count for {cohort} k={k}")
    out = OUT / "calibration_size_sensitivity.csv"
    summary.to_csv(out, index=False)
    print("saved", out, "summary rows", len(summary), flush=True)


def write_split_manifest() -> None:
    rows = []
    for cohort in ("virtual", "physical_camargo"):
        d = load_inputs(cohort)
        if cohort == "physical_camargo":
            with np.load(OUT / "camargo_prepared/camargo_external.npz") as z:
                source_windows = z["source_window"]
                subject_names = z["subject_names"]
        else:
            source_windows = np.full(len(d.G), -1)
            subject_names = np.full(len(d.G), "", dtype="U1")
        for fold in (1, 2, 3, 4, 5):
            te = np.flatnonzero(d.F == fold)
            g_test = d.G[te]
            subsets = disjoint_indices(g_test, d.calibration_sizes)
            for sid in np.unique(g_test):
                local = np.flatnonzero(g_test == sid)
                rng = np.random.default_rng(SEED + SUBSET_SEED_OFFSET + int(sid))
                ordered = rng.permutation(local)
                for rank, local_idx in enumerate(ordered, start=1):
                    global_idx = int(te[local_idx])
                    row = {
                        "cohort": cohort,
                        "outer_fold": fold,
                        "subject": int(sid),
                        "subject_name": str(subject_names[global_idx]),
                        "global_record_index_zero_based": global_idx,
                        "source_window": (
                            int(source_windows[global_idx]) if cohort == "physical_camargo" else ""
                        ),
                        "deterministic_subset_rank": rank,
                        "participant_record_count": len(local),
                    }
                    for k, (calibration, evaluation) in subsets.items():
                        status = (
                            "ineligible" if len(local) <= k else
                            "calibration" if local_idx in calibration[int(sid)] else
                            "evaluation"
                        )
                        if status == "evaluation" and local_idx not in evaluation:
                            raise AssertionError("Split manifest/evaluation disagreement")
                        row[f"status_k{k}"] = status
                    rows.append(row)
    out = OUT / "calibration_split_manifest.csv"
    pd.DataFrame(rows).sort_values(
        ["cohort", "subject", "deterministic_subset_rank"]
    ).to_csv(out, index=False)
    print("saved", out, "records", len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("virtual", "physical_camargo"))
    ap.add_argument("--mc", type=int, default=50)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--splits", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    if args.splits:
        write_split_manifest()
        return
    if args.cohort is None:
        ap.error("--cohort is required unless --summarize is given")
    folds = (args.fold,) if args.fold else (1, 2, 3, 4, 5)
    evaluate(args.cohort, args.mc, folds, args.tag)


if __name__ == "__main__":
    main()
