"""Verify frozen data/fold hashes and stored rerun evidence against frozen tables.

NOTE: This checks saved rerun outputs. Run fold scripts to actually refit models.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
H={
 'analysis_code/data/processed/strict_parameters.npz':'f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e',
 'analysis_code/protocol/outer_folds.csv':'8aaf78f5c51f0e5d28d88e17c1bbbb0563fdeb7cc09d3e39fe666d03972ca782',
}
for path,want in H.items():
 have=hashlib.sha256((R/path).read_bytes()).hexdigest()
 assert have==want,(path,have,want)
 print('HASH VERIFIED:',path)
def cmp(reference_path,rerun_path,keys,metric,tol):
 a=pd.read_csv(R/reference_path);b=pd.read_csv(R/rerun_path)
 m=a[keys+metric].merge(b[keys+metric],on=keys,how='outer',validate='one_to_one',indicator=True,suffixes=('_frozen','_rerun'))
 assert len(m)==len(a)==len(b) and (m._merge=='both').all(),(reference_path,rerun_path)
 mx=max(float(np.max(np.abs(m[c+'_frozen'].to_numpy(float)-m[c+'_rerun'].to_numpy(float)))) for c in metric)
 assert mx<=tol,(reference_path,rerun_path,mx)
 print(f'PASS {len(m):4d} rows; max delta {mx:.9g} deg: {rerun_path}')
cmp('source_data/virtual_rotation_matrix_summary.csv','reproduction_evidence/virtual/rotation_matrix_v2/matrix_summary.csv', ['task','target','train_augmentation_max_deg','test_rotation_deg'],['mean_participant_rmse_deg','ci_low','ci_high'],1e-8)
cmp('source_data/virtual_orientation_baselines.csv','reproduction_evidence/virtual/orientation_baselines_v2/summary.csv', ['method','task','target','test_rotation_deg'],['mean_participant_rmse_deg','ci95_low_deg','ci95_high_deg'],1e-8)
cmp('revision_analysis/results/augmentation_copy_sensitivity.csv','reproduction_evidence/reviewer/augmentation_copy_sensitivity.csv',['copy_count','task','target','test_rotation_deg'],['mean_participant_rmse_deg','ci_low','ci_high'],1e-4)
cmp('revision_analysis/results/calibration_size_sensitivity.csv','reproduction_evidence/reviewer/calibration_size_sensitivity.csv',['cohort','method','calibration_size_k','task','target','test_rotation_deg'],['mean_participant_rmse_deg','ci95_low_deg','ci95_high_deg'],1e-4)
print('STORED EVIDENCE: VERIFIED; these tests do not refit the estimator')
