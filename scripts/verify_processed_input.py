"""Verify the exact frozen virtual-IMU analysis input and participant folds.

This verifies provenance and shape. It does not claim to regenerate the arrays
from original C3D recordings. Do not load pickle-capable NPZ before hash check.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e'
EXPECTED_FOLDS = '8aaf78f5c51f0e5d28d88e17c1bbbb0563fdeb7cc09d3e39fe666d03972ca782'
DEFAULT_DATA = ROOT / 'analysis_code/data/processed/strict_parameters.npz'
FOLDS = ROOT / 'analysis_code/protocol/outer_folds.csv'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    a = ap.parse_args()
    data = a.data.resolve()
    if sha256(FOLDS) != EXPECTED_FOLDS:
        raise SystemExit('ERROR: fold CSV hash differs from frozen confirmatory protocol')
    f = pd.read_csv(FOLDS)
    assert f.columns.tolist() == ['cycle_index', 'subject', 'outer_fold']
    assert f.cycle_index.tolist() == list(range(747))
    assert len(f) == 747 and f.subject.nunique() == 138
    assert sorted(f.outer_fold.unique().tolist()) == [1, 2, 3, 4, 5]
    assert (f.groupby('subject').outer_fold.nunique() == 1).all()
    print('FOLD MANIFEST: VERIFIED (747 cycles, 138 participants, participant-disjoint)')
    if not data.is_file():
        print(f'PROCESSED INPUT: MISSING: {data}')
        print('Find the frozen strict_parameters.npz on the original analysis computer.')
        return 2
    digest = sha256(data)
    if digest != EXPECTED:
        print('PROCESSED INPUT: HASH MISMATCH')
        print('Observed SHA-256:', digest)
        print('Expected SHA-256:', EXPECTED)
        return 3
    sys.path.insert(0, str(ROOT / 'analysis_code'))
    from src.confirmatory.data import load_study_data
    d = load_study_data(data)
    assert d.X.shape == (747, 4, 101, 6), d.X.shape
    assert d.T.shape == (747, 9), d.T.shape
    assert d.n_subjects == 138 and d.n_cycles == 747
    assert np.array_equal(d.subjects.astype(int), f.subject.to_numpy())
    print('PROCESSED INPUT: VERIFIED')
    print('File:', data)
    print('SHA-256:', digest)
    print('X:', d.X.shape, 'T:', d.T.shape)
    print('All frozen cycle-subject/fold assignments: VERIFIED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
