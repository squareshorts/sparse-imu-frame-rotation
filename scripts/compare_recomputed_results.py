"""Check regenerated virtual matrix outputs against frozen submission result tables.

Requires completing all five outer fold runs. Passing means computation from the
frozen processed NPZ matches published summary tables within specified tolerances;
it does not establish raw-C3D-to-NPZ reproducibility.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
GOLD=ROOT/'source_data'
VM=ROOT/'analysis_code/strengthening/rotation_matrix_v2'
B=ROOT/'analysis_code/strengthening/orientation_baselines_v2'


def compare(a: Path,b: Path,key: list[str],metric: list[str],tolerance: float)->None:
    if not a.is_file() or not b.is_file():
        raise FileNotFoundError(f'Missing file: {a if not a.is_file() else b}')
    ref=pd.read_csv(a);got=pd.read_csv(b)
    assert not ref.duplicated(key).any() and not got.duplicated(key).any()
    m=ref.merge(got,on=key,how='outer',validate='one_to_one',indicator=True,suffixes=('_ref','_got'))
    if len(m)!=len(ref) or len(m)!=len(got) or (m['_merge']!='both').any():
        raise AssertionError(f'Key mismatch: {a} versus {b}')
    maximum=0.0
    for col in metric:
        lhs=m[col+'_ref'].to_numpy(float);rhs=m[col+'_got'].to_numpy(float)
        if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
            raise AssertionError(f'Non-finite results in {col}')
        delta=float(np.max(np.abs(lhs-rhs)))
        maximum=max(maximum,delta)
        print(f'   {col}: max absolute discrepancy={delta:.9f} deg')
    if maximum>tolerance:
        raise AssertionError(f'FAILED: {b}; max discrepancy {maximum:.6g} > {tolerance:.6g} deg')
    print(f'PASS: {len(m)} cells compared to {a.name}, max={maximum:.8f} deg')


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tolerance',type=float,default=0.005)
    p.add_argument('--include-normalization',action='store_true')
    args=p.parse_args()
    common=['task','target','train_augmentation_max_deg','test_rotation_deg']
    compare(GOLD/'virtual_rotation_matrix_summary.csv',VM/'matrix_summary.csv',common,
            ['mean_participant_rmse_deg','ci_low','ci_high'],args.tolerance)
    if args.include_normalization:
        common=['method','task','target','test_rotation_deg']
        compare(GOLD/'virtual_orientation_baselines.csv',B/'summary.csv',common,
                ['mean_participant_rmse_deg','ci95_low_deg','ci95_high_deg'],args.tolerance)


if __name__=='__main__':main()
