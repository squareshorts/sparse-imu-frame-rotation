"""Reapply the submitted virtual operating rule to three CI ceilings.

Reads only the final submission's matrix summary. All outputs are revision files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submitted_final/analysis/source_data/virtual_rotation_matrix_summary.csv"
BASELINE = ROOT / "submitted_final/analysis/source_data/virtual_deployment_rule.csv"
OUTPUT = ROOT / "results/revision_2026_10/threshold_sensitivity.csv"
CEILINGS = (4.0, 4.5, 5.0)
ENDPOINTS = {
    ("ankle_two", "ankle_dorsiflex_deg"),
    ("ankle_two", "ankle_plantarflex_deg"),
    ("knee_two", "knee_stance_flex_deg"),
    ("knee_two", "knee_swing_flex_deg"),
}


def main() -> None:
    df = pd.read_csv(SOURCE)
    caps = sorted(df.train_augmentation_max_deg.unique())
    angles = sorted(df.test_rotation_deg.unique())
    expected = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    assert caps == expected and angles == expected, (caps, angles)
    assert set(zip(df.task, df.target)) == ENDPOINTS
    assert len(df) == len(caps) * len(angles) * len(ENDPOINTS)
    assert not df.duplicated(["task", "target", "train_augmentation_max_deg", "test_rotation_deg"]).any()
    assert (df.n_subjects == 138).all()
    assert df[["mean_participant_rmse_deg", "ci_low", "ci_high"]].notna().all().all()

    clean = df[df.test_rotation_deg == 0].groupby("train_augmentation_max_deg").agg(
        clean_worst_endpoint_mean_rmse_deg=("mean_participant_rmse_deg", "max"),
        clean_worst_endpoint_upper_ci_deg=("ci_high", "max"),
    )
    assert clean.clean_worst_endpoint_mean_rmse_deg.is_monotonic_increasing
    assert (clean.clean_worst_endpoint_mean_rmse_deg.diff().dropna() > 0).all()

    rows = []
    for ceiling in CEILINGS:
        for bound in angles:
            qualifying = []
            for cap in caps:
                q = df[(df.train_augmentation_max_deg == cap) & (df.test_rotation_deg <= bound)]
                assert len(q) == len(ENDPOINTS) * sum(angle <= bound for angle in angles)
                worst_upper = float(q.ci_high.max())
                if worst_upper <= ceiling:
                    qualifying.append((cap, worst_upper))

            # The manuscript rule is lowest worst-endpoint clean mean RMSE;
            # smaller augmentation cap is the tie break.
            chosen = min(
                qualifying,
                key=lambda c: (float(clean.loc[c[0], "clean_worst_endpoint_mean_rmse_deg"]), c[0]),
                default=None,
            )
            rows.append({
                "criterion_upper_ci_deg": ceiling,
                "field_rotation_bound_deg": bound,
                "qualifying_augmentation_caps_deg": ";".join(f"{cap:g}" for cap, _ in qualifying),
                "n_qualifying_caps": len(qualifying),
                "selected_augmentation_cap_deg": chosen[0] if chosen else np.nan,
                "selected_worst_upper_ci_through_bound_deg": chosen[1] if chosen else np.nan,
                "selected_worst_clean_mean_rmse_deg": (
                    float(clean.loc[chosen[0], "clean_worst_endpoint_mean_rmse_deg"])
                    if chosen else np.nan
                ),
                "decision": "QUALIFIES" if chosen else "NO_TESTED_CAP_QUALIFIES",
            })

    out = pd.DataFrame(rows)
    maximum = out[out.decision == "QUALIFIES"].groupby("criterion_upper_ci_deg")[
        "field_rotation_bound_deg"
    ].max()
    out["maximum_supported_field_rotation_bound_deg"] = out.criterion_upper_ci_deg.map(maximum)

    baseline = pd.read_csv(BASELINE)
    original = out[out.criterion_upper_ci_deg == 4.5].reset_index(drop=True)
    baseline = baseline.sort_values("field_q95_rotation_deg").reset_index(drop=True)
    assert np.array_equal(original.field_rotation_bound_deg.to_numpy(), baseline.field_q95_rotation_deg.to_numpy())
    assert np.array_equal(original.n_qualifying_caps.to_numpy(), baseline.n_qualifying_rows.to_numpy())
    assert np.allclose(
        original.selected_augmentation_cap_deg,
        baseline.selected_train_augmentation_max_deg,
        equal_nan=True,
    )
    assert np.allclose(
        original.selected_worst_upper_ci_through_bound_deg,
        baseline.worst_ci_high_deg_through_q95,
        equal_nan=True,
    )
    assert np.allclose(
        original.selected_worst_clean_mean_rmse_deg,
        baseline.worst_clean_mean_rmse_deg,
        equal_nan=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print("CLEAN WORST-ENDPOINT MEAN RMSE BY CAP")
    print(clean.to_string())
    print("4.5-degree rows exactly match submitted virtual_deployment_rule.csv")
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
