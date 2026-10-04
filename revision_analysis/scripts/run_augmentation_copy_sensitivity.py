"""Nested-copy Ridge sensitivity using the submitted virtual protocol.

Writes only to results/revision_2026_10. The submitted five-copy result is
checked against source_data/virtual_rotation_matrix_summary.csv before output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "submitted_final" / "analysis"
sys.path.insert(0, str(FINAL / "analysis_code"))
from src.confirmatory.constants import PRIMARY_TASKS  # noqa: E402
from src.confirmatory.data import load_study_data  # noqa: E402
from src.confirmatory.features import engineered_features  # noqa: E402
from src.confirmatory.folds import load_outer_folds  # noqa: E402
from src.confirmatory.orientation import (  # noqa: E402
    make_augmented_training_set,
    perturb_by_subject,
)

OUT = ROOT / "results" / "revision_2026_10"
SEED = 20260907
MC = 50
BOOT = 5000
CAP = 10.0
COUNTS = (1, 3, 5, 10)
TASKS = ("knee_two", "ankle_two")


class WeightedRidge:
    """Exact estimator/scaler weighting from run_rotation_matrix_v2.py:27-37."""

    def __init__(self, alpha: float):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray):
        self.scaler.fit(x, sample_weight=weight)
        self.model.fit(self.scaler.transform(x), y, sample_weight=weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(x)), dtype=np.float32)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    vals = np.asarray(values, float)
    n = len(vals)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOT, float)
    for start in range(0, BOOT, 500):
        end = min(start + 500, BOOT)
        ix = rng.integers(0, n, size=(end - start, n))
        draws[start:end] = vals[ix].mean(axis=1)
    return float(vals.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_study_data(ROOT / "data" / "processed" / "strict_parameters.npz")
    outer = load_outer_folds(FINAL / "analysis_code" / "protocol" / "outer_folds.csv", d.subjects)
    submitted = pd.read_csv(FINAL / "source_data" / "virtual_rotation_matrix_summary.csv")
    assert d.n_subjects == 138 and d.n_cycles == 747
    assert set(np.unique(outer)) == {1, 2, 3, 4, 5}
    # All >=5-degree caps use alpha=100 in all folds of the submitted source.
    alphas = pd.read_csv(FINAL / "source_data" / "virtual_selected_alphas_v2.csv")
    selected = alphas[alphas.train_augmentation_max_deg == CAP]
    assert len(selected) == 5 and (selected.selected_alpha == 100.0).all()

    # key: (copy_count, task, target, angle, participant) -> [sum squared errors, count]
    accum: dict[tuple[int, str, str, float, int], list[float]] = {}
    for fold in sorted(np.unique(outer)):
        fold = int(fold)
        tr, te = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        for ti, task_name in enumerate(TASKS):
            task = PRIMARY_TASKS[task_name]
            n_train = len(tr)
            train_seed = SEED + fold * 2_000_000 + int(round(CAP * 10)) * 10_000 + ti
            # Existing augmentation function creates clean + ten copies in order.
            # Each first-k prefix exactly shares the final analysis's first five copies.
            xa, ya, _, da, sa = make_augmented_training_set(
                d.X[tr], d.T[tr][:, task["targets"]], d.subjects[tr],
                d.durations[tr], d.speeds[tr], task["sensors"], CAP, 10, train_seed,
            )
            fa = engineered_features(xa, task["sensors"], da, sa)
            models = {}
            for k in COUNTS:
                stop = (k + 1) * n_train
                weight = np.full(stop, 1.0 / (k + 1), float)
                assert np.isclose(weight.sum(), n_train)
                models[k] = WeightedRidge(100.0).fit(fa[:stop], ya[:stop], weight)

            y = d.T[te][:, task["targets"]]
            g = d.subjects[te]
            participants = np.unique(g)
            for angle in (0.0, 10.0):
                for rep in range(1 if angle == 0 else MC):
                    if angle == 0:
                        xte = d.X[te]
                    else:
                        pseed = SEED + fold * 3_000_000 + ti * 100_000 + int(round(angle * 10)) * 100 + rep
                        xte, _ = perturb_by_subject(d.X[te], g, task["sensors"], angle, pseed, exact_angle=True)
                    fte = engineered_features(xte, task["sensors"], d.durations[te], d.speeds[te])
                    for k, model in models.items():
                        e2 = (model.predict(fte) - y) ** 2
                        for sid in participants:
                            ix = g == sid
                            sums, n = e2[ix].sum(axis=0), int(ix.sum())
                            for j, target in enumerate(task["target_names"]):
                                key = (k, task_name, target, angle, int(sid))
                                value = accum.setdefault(key, [0.0, 0.0])
                                value[0] += float(sums[j])
                                value[1] += n
                print("copy sensitivity", fold, task_name, angle, flush=True)

    participant_rows = [
        {"copy_count": k, "task": task, "target": target,
         "test_rotation_deg": angle, "subject": sid,
         "rmse_deg": float(np.sqrt(sum_sq / n)), "pooled_error_count": int(n)}
        for (k, task, target, angle, sid), (sum_sq, n) in accum.items()
    ]
    part = pd.DataFrame(participant_rows).sort_values(
        ["copy_count", "task", "target", "test_rotation_deg", "subject"]
    )
    # The five-copy rows use the exact submitted bootstrap seeds, indexed by
    # the complete sorted matrix in summarize_rotation_matrix_v2.py.
    source_order = submitted.sort_values(
        ["task", "target", "train_augmentation_max_deg", "test_rotation_deg"]
    ).reset_index(drop=True)
    source_boot_seed = {
        (row.task, row.target, float(row.train_augmentation_max_deg), float(row.test_rotation_deg)):
        SEED + 5_000_000 + i
        for i, row in source_order.iterrows()
    }
    summary = []
    for i, (keys, q) in enumerate(part.groupby(
        ["copy_count", "task", "target", "test_rotation_deg"], sort=True
    )):
        k, task, target, angle = keys
        boot_seed = (source_boot_seed[(task, target, CAP, angle)] if k == 5
                     else SEED + 12_000_000 + i)
        mean, low, high = bootstrap_mean(q.rmse_deg.to_numpy(), boot_seed)
        summary.append({"copy_count": k, "task": task, "target": target,
                        "test_rotation_deg": angle, "mean_participant_rmse_deg": mean,
                        "ci_low": low, "ci_high": high, "n_participants": len(q)})
    sm = pd.DataFrame(summary)

    # Gate the new analysis against the submitted five-copy primary means.
    five = sm[sm.copy_count == 5]
    expected = submitted[(submitted.train_augmentation_max_deg == CAP) &
                         (submitted.test_rotation_deg.isin([0.0, 10.0]))]
    joined = five.merge(expected, on=["task", "target", "test_rotation_deg"],
                        suffixes=("_new", "_submitted"), validate="one_to_one")
    assert len(joined) == 8
    delta = np.abs(joined.mean_participant_rmse_deg_new -
                   joined.mean_participant_rmse_deg_submitted)
    max_delta = float(delta.max())
    if max_delta > 0.005:
        raise RuntimeError(f"Five-copy baseline disagrees with submitted source: max mean delta {max_delta:.6f} deg")
    ci_delta = max(float(np.abs(joined.ci_low_new - joined.ci_low_submitted).max()),
                   float(np.abs(joined.ci_high_new - joined.ci_high_submitted).max()))
    if ci_delta > 0.005:
        raise RuntimeError(f"Five-copy bootstrap disagrees with submitted source: max CI delta {ci_delta:.6f} deg")

    worst = sm[sm.test_rotation_deg == 10].groupby("copy_count").ci_high.max().to_dict()
    sm["worst_endpoint_upper_ci_at_10_deg"] = sm.copy_count.map(worst)
    sm["all_endpoint_4p5_criterion_at_10_deg"] = sm.copy_count.map(
        {k: bool(v <= 4.5) for k, v in worst.items()}
    )
    sm["max_five_copy_mean_delta_vs_submitted_deg"] = max_delta
    sm["max_five_copy_ci_delta_vs_submitted_deg"] = ci_delta
    five_p = part[part.copy_count == 5][
        ["task", "target", "test_rotation_deg", "subject", "rmse_deg"]
    ].rename(columns={"rmse_deg": "rmse_5"})
    ten_p = part[part.copy_count == 10][
        ["task", "target", "test_rotation_deg", "subject", "rmse_deg"]
    ].rename(columns={"rmse_deg": "rmse_10"})
    paired = five_p.merge(ten_p, on=["task", "target", "test_rotation_deg", "subject"],
                          validate="one_to_one")
    plateau_rows = []
    for i, (keys, q) in enumerate(paired.groupby(
        ["task", "target", "test_rotation_deg"], sort=True
    )):
        task, target, angle = keys
        benefit, low, high = bootstrap_mean(
            q.rmse_5.to_numpy() - q.rmse_10.to_numpy(), SEED + 15_000_000 + i
        )
        plateau_rows.append({"task": task, "target": target, "test_rotation_deg": angle,
                             "mean_rmse_benefit_of_10_vs_5_deg": benefit,
                             "ci_low": low, "ci_high": high,
                             "n_participants": len(q)})
    part.to_csv(OUT / "augmentation_copy_participant_metrics.csv", index=False)
    sm.to_csv(OUT / "augmentation_copy_sensitivity.csv", index=False)
    pd.DataFrame(plateau_rows).to_csv(OUT / "augmentation_copy_paired_5_vs_10.csv", index=False)
    print("five-copy max mean delta vs submitted:", max_delta)
    print("five-copy max CI delta vs submitted:", ci_delta)
    print(sm[sm.test_rotation_deg == 10].to_string(index=False))


if __name__ == "__main__":
    main()
