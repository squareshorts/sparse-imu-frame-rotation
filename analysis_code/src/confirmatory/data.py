from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import SENSOR_NAMES


@dataclass(frozen=True)
class StudyData:
    X: np.ndarray
    T: np.ndarray
    subjects: np.ndarray
    ages: np.ndarray
    durations: np.ndarray
    speeds: np.ndarray
    sides: np.ndarray
    trials: np.ndarray
    targets: np.ndarray
    contacts: np.ndarray

    @property
    def n_cycles(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.subjects).size)


def load_study_data(path: str | Path) -> StudyData:
    z = np.load(Path(path), allow_pickle=True)
    required = {
        "X", "T", "subjects", "ages", "durations", "speeds",
        "sides", "trials", "targets", "contacts",
    }
    missing = required - set(z.files)
    if missing:
        raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")
    data = StudyData(**{k: z[k] for k in required})
    validate_study_data(data)
    return data


def validate_study_data(d: StudyData) -> None:
    if d.X.ndim != 4 or d.X.shape[1:] != (4, 101, 6):
        raise ValueError(f"Expected X [N,4,101,6], got {d.X.shape}")
    if d.T.ndim != 2 or d.T.shape[0] != d.X.shape[0]:
        raise ValueError(f"Target shape mismatch: X={d.X.shape}, T={d.T.shape}")
    n = d.X.shape[0]
    for name in ("subjects", "ages", "durations", "speeds", "sides", "trials", "contacts"):
        arr = getattr(d, name)
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} != {n}")
    if list(SENSOR_NAMES) != ["pelvis", "thigh", "shank", "foot"]:
        raise RuntimeError("Unexpected sensor convention")
    if not np.isfinite(d.X).all():
        raise ValueError("X contains non-finite values")
    if not np.isfinite(d.T).all():
        raise ValueError("T contains non-finite values")
