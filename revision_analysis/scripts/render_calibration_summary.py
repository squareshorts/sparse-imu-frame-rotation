"""Render the full calibration-size endpoint table from the numeric CSV."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision_2026_10"
df = pd.read_csv(BASE / "calibration_size_sensitivity.csv")
targets = (
    "ankle_dorsiflex_deg", "ankle_plantarflex_deg",
    "knee_stance_flex_deg", "knee_swing_flex_deg",
)
labels = ("Ankle DF", "Ankle PF", "Knee stance", "Knee swing")
lines = [
    "# Disjoint calibration-size endpoint results",
    "",
    "Each cell is mean participant RMSE [95% participant-bootstrap CI] in degrees. "
    "The 10-degree condition pools squared errors over 50 exact-rotation realizations "
    "before participant RMSE is calculated. The complete machine-readable results "
    "are in `calibration_size_sensitivity.csv`.",
    "",
    "| Cohort | Method | k | Test rotation | Participants | Ankle DF | Ankle PF | Knee stance | Knee swing |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for (cohort, method, k, angle), group in df.groupby(
    ["cohort", "method", "calibration_size_k", "test_rotation_deg"], sort=True
):
    q = group.set_index("target")
    vals = []
    for target in targets:
        r = q.loc[target]
        vals.append(
            f"{r.mean_participant_rmse_deg:.3f} "
            f"[{r.ci95_low_deg:.3f}, {r.ci95_high_deg:.3f}]"
        )
    lines.append(
        f"| {cohort} | {method} | {int(k)} | {angle:g}° | "
        f"{int(group.n_participants.iloc[0])} | " + " | ".join(vals) + " |"
    )
lines += [
    "",
    "The calibration records are excluded from scoring. Cohort eligibility and "
    "scored-record counts are in `calibration_audit.csv` and "
    "`calibration_size_sensitivity.csv`; the exact nested split is in "
    "`calibration_split_manifest.csv`. Differences between k values reflect both "
    "the calibration size and the changing evaluation set, including virtual "
    "participant loss at k=3.",
    "",
]
out = BASE / "calibration_size_table.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(out, "rows", len(lines) - 9, "targets", ", ".join(labels))
