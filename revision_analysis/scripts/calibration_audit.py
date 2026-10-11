"""Inventory calibration exposure in the submitted frame-normalization runners.

This computes batch sizes only. The overlap and implementation facts are established
by the cited final runner source, not inferred from the data distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_analysis/results/calibration_audit.csv"
VIRTUAL = ROOT / "analysis_code/data/processed/strict_parameters.npz"
PHYSICAL_QC = ROOT / "analysis_code/strengthening/camargo_external/qc_windows.csv"
VIRTUAL_RUNNER = (
    "analysis_code/strengthening/"
    "run_orientation_baselines_v2.py"
)
PHYSICAL_RUNNER = (
    "analysis_code/strengthening/"
    "camargo_external/run_camargo_baselines_v3_pooled_fold.py"
)
TRANSFORMS = (
    "analysis_code/strengthening/"
    "run_orientation_baselines_fold.py"
)


def main() -> None:
    if VIRTUAL.is_file():
        with np.load(VIRTUAL, allow_pickle=True) as z:
            virtual_subjects = z["subjects"]
        virtual_source = VIRTUAL
    else:
        # Exact cycle/subject manifest was archived with the frozen analysis.
        # This fallback audits calibration group *sizes*, not signal values.
        manifest = ROOT / "analysis_code/protocol/outer_folds.csv"
        f = pd.read_csv(manifest)
        assert len(f) == 747 and f.cycle_index.tolist() == list(range(747))
        virtual_subjects = f.subject.to_numpy()
        virtual_source = manifest
    _, virtual_counts = np.unique(virtual_subjects, return_counts=True)

    qc = pd.read_csv(PHYSICAL_QC)
    included = qc[qc["included"].astype(str).str.lower().eq("true")]
    physical_counts = included.groupby("subject").size().to_numpy()

    if (len(virtual_subjects), len(virtual_counts)) != (747, 138):
        raise RuntimeError("Virtual source is not the submitted 747-cycle/138-person cohort")
    if (len(included), len(physical_counts)) != (248, 19):
        raise RuntimeError("Physical QC is not the submitted 248-window/19-person cohort")

    rows = []
    for cohort, unit, counts, input_path, runner in (
        ("virtual", "cycle", virtual_counts, virtual_source, VIRTUAL_RUNNER),
        ("physical_camargo", "window", physical_counts, PHYSICAL_QC, PHYSICAL_RUNNER),
    ):
        for method in ("template_registration", "gait_functional_calibration"):
            rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "calibration_unit": unit,
                    "n_participants": len(counts),
                    "n_scored_units_baseline": int(counts.sum()),
                    "calibration_units_per_participant_min": int(counts.min()),
                    "calibration_units_per_participant_q1": float(np.quantile(counts, 0.25)),
                    "calibration_units_per_participant_median": float(np.median(counts)),
                    "calibration_units_per_participant_q3": float(np.quantile(counts, 0.75)),
                    "calibration_units_per_participant_max": int(counts.max()),
                    "calibration_and_scored_units_overlap": True,
                    "transductive_calibration": True,
                    "recomputed_for_each_rotation_realization": True,
                    "endpoint_labels_used_for_calibration": False,
                    "calibration_scope": "all held-out participant units",
                    "data_source": str(input_path.relative_to(ROOT)).replace("\\", "/"),
                    "runner_source": runner,
                    "transform_source": TRANSFORMS,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main()
