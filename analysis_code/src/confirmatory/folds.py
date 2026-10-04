from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_outer_folds(subjects: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    # GroupKFold is deterministic for a fixed ordered input. We explicitly save the result.
    X_dummy = np.zeros((len(subjects), 1), dtype=np.float32)
    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, test_idx) in enumerate(splitter.split(X_dummy, groups=subjects), start=1):
        for i in test_idx:
            rows.append({"cycle_index": int(i), "subject": int(subjects[i]), "outer_fold": fold})
    out = pd.DataFrame(rows).sort_values("cycle_index").reset_index(drop=True)
    if len(out) != len(subjects) or out["cycle_index"].nunique() != len(subjects):
        raise RuntimeError("Outer fold assignment is incomplete or duplicated")
    return out


def save_outer_folds(folds: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)


def load_outer_folds(path: str | Path, subjects: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path).sort_values("cycle_index")
    if not np.array_equal(df["cycle_index"].to_numpy(), np.arange(len(subjects))):
        raise ValueError("Fold file cycle indices do not match current dataset")
    if not np.array_equal(df["subject"].to_numpy(), subjects.astype(int)):
        raise ValueError("Fold file subjects do not match current dataset")
    folds = df["outer_fold"].to_numpy(dtype=int)
    leakage_audit(folds, subjects)
    return folds


def leakage_audit(folds: np.ndarray, subjects: np.ndarray) -> None:
    for fold in np.unique(folds):
        te_subjects = set(subjects[folds == fold].tolist())
        tr_subjects = set(subjects[folds != fold].tolist())
        overlap = te_subjects & tr_subjects
        if overlap:
            raise RuntimeError(f"Subject leakage in fold {fold}: {sorted(overlap)}")


def inner_splits_for_outer_train(subjects_train: np.ndarray, n_splits: int = 3):
    splitter = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(subjects_train), 1), dtype=np.float32)
    yield from splitter.split(X_dummy, groups=subjects_train)
