"""Extract the submitted physical-IMU Ridge 0/10 degree source rows, without refitting."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source_data"
OUT = ROOT / "revision_analysis/results/physical_zero_rotation.csv"
TARGETS = (
    ("ankle_two", "ankle_dorsiflex_deg"),
    ("ankle_two", "ankle_plantarflex_deg"),
    ("knee_two", "knee_stance_flex_deg"),
    ("knee_two", "knee_swing_flex_deg"),
)
EXPECTED_PAIRED_10 = (0.3823575, 0.4820858, 0.6634664, 0.6707565)
CRITERION = 4.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique(rows: list[dict[str, str]], task: str, target: str, angle: int) -> dict[str, str]:
    matches = [r for r in rows if r["task"] == task and r["target"] == target
               and float(r["test_rotation_deg"]) == angle]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {task}/{target}/{angle}, got {len(matches)}")
    return matches[0]


def main() -> None:
    matrix = [r for r in read_csv(SOURCE / "external_rotation_matrix_summary.csv")
              if float(r["train_augmentation_max_deg"]) == 0]
    paired = read_csv(SOURCE / "external_unaugmented_degradation.csv")
    selected = [(unique(matrix, task, target, 0), unique(matrix, task, target, 10),
                 unique(paired, task, target, 10)) for task, target in TARGETS]
    if any(int(float(z[0]["n_participants"])) != 19 or int(float(z[1]["n_participants"])) != 19
           or int(float(z[2]["n_participants"])) != 19 for z in selected):
        raise ValueError("Unexpected participant count")

    worst0 = max(float(z[0]["ci95_high_deg"]) for z in selected)
    worst10 = max(float(z[1]["ci95_high_deg"]) for z in selected)
    records = []
    for (task, target), (clean, rotated, delta), expected in zip(TARGETS, selected, EXPECTED_PAIRED_10):
        d = float(delta["mean_paired_degradation_deg"])
        if abs(d - expected) > 0.002:
            raise ValueError(f"Paired increase differs from submitted claim audit: {target}: {d}")
        if abs(float(rotated["mean_participant_rmse_deg"]) -
               float(clean["mean_participant_rmse_deg"]) - d) > 1e-9:
            raise ValueError(f"Absolute means and paired increase disagree: {target}")
        records.append({
            "task": task,
            "endpoint": target,
            "n_participants": 19,
            "rmse_0_deg": clean["mean_participant_rmse_deg"],
            "ci95_low_0_deg": clean["ci95_low_deg"],
            "ci95_high_0_deg": clean["ci95_high_deg"],
            "rmse_10_deg": rotated["mean_participant_rmse_deg"],
            "ci95_low_10_deg": rotated["ci95_low_deg"],
            "ci95_high_10_deg": rotated["ci95_high_deg"],
            "paired_increase_10_minus_0_deg": delta["mean_paired_degradation_deg"],
            "paired_ci95_low_deg": delta["ci95_low_deg"],
            "paired_ci95_high_deg": delta["ci95_high_deg"],
            "worst_upper_ci_0_deg_all_endpoints": worst0,
            "all_endpoint_4p5_criterion_at_0_deg": str(worst0 <= CRITERION).lower(),
            "worst_upper_ci_10_deg_all_endpoints": worst10,
            "all_endpoint_4p5_criterion_at_10_deg": str(worst10 <= CRITERION).lower(),
            "source_matrix": "source_data/external_rotation_matrix_summary.csv",
            "source_paired": "source_data/external_unaugmented_degradation.csv",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} endpoints to {OUT}")
    print(f"Worst upper CI: 0°={worst0:.6f}°, 10°={worst10:.6f}°")


if __name__ == "__main__":
    main()
