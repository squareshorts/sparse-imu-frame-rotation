from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_study_data
from .orientation import perturb_by_subject, assert_magnitude_preservation
from .utils import load_json, project_root, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/confirmatory.json")
    args = ap.parse_args()
    root = project_root()
    cfg = load_json(root / args.config)
    path = root / cfg["data_file"]
    d = load_study_data(path)
    Xr, records = perturb_by_subject(d.X[:40], d.subjects[:40], (1, 2), 10.0, cfg["seed"], exact_angle=True)
    assert_magnitude_preservation(d.X[:40, [1, 2]], Xr[:, [1, 2]])

    counts = pd.Series(d.subjects).value_counts()
    report = f"""# Data integrity report

- File: `{path.relative_to(root)}`
- SHA-256: `{sha256_file(path)}`
- Cycles: {d.n_cycles}
- Participants: {d.n_subjects}
- X shape: {d.X.shape}
- T shape: {d.T.shape}
- Target names: {list(map(str, d.targets))}
- Age range: {np.nanmin(d.ages):.2f} to {np.nanmax(d.ages):.2f} years
- Cycles/participant: median {counts.median():.1f}, range {counts.min()} to {counts.max()}
- Finite X: {bool(np.isfinite(d.X).all())}
- Finite T: {bool(np.isfinite(d.T).all())}
- Orientation magnitude-preservation test: PASS
- Orientation test rotations checked: {len(records)}
"""
    out = root / "protocol/data_integrity_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
