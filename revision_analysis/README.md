# Completed reviewer-requested revision analyses

This directory contains the reviewer-requested analyses used in the revised manuscript and response letter. Archived CSV outputs are under `results/`; standalone scripts are under `scripts/`. Scripts now resolve the root from their own location and no longer require the original `submitted_final/analysis` workspace.

Reproduction from the processed virtual input requires the *exact* frozen `analysis_code/data/processed/strict_parameters.npz`. See `../DATA_PROVENANCE.md`.


## Threshold sensitivity
`results/threshold_sensitivity.csv` evaluates the virtual operating rule under 4.0, 4.5, and 5.0 degree upper-CI ceilings. The largest supported tested field bounds are 7.5, 10, and 15 degrees, respectively.

## Physical zero-rotation performance
`results/physical_zero_rotation.csv` reports the four physical-IMU endpoint RMSEs at 0 degrees and 10 degrees. The 4.5-degree virtual criterion is already exceeded by some physical endpoints at zero rotation, so the virtual operating bounds are not transferred to the external cohort.

## Augmentation-copy sensitivity
`results/augmentation_copy_sensitivity.csv` evaluates 1, 3, 5, and 10 rotated copies at the boundary-defining exact 10-degree stress condition, with the clean condition retained as a reference. Five copies are the smallest tested count whose worst-endpoint upper CI at exact 10 degrees is below 4.5 degrees. This analysis does not recompute the complete intermediate-angle operating envelope for every copy count.

## Calibration audit and size sensitivity
`results/calibration_audit.csv` documents that the submitted normalization analyses used unlabeled held-out walking records for frame estimation and scoring, so the primary comparison is transductive with respect to gait records. Endpoint labels were not used for frame estimation.

`results/calibration_size_sensitivity.csv` separates calibration and scoring records and evaluates small calibration sets. Virtual tests use 1 and 3 calibration cycles; physical tests use 1, 3, and 5 calibration windows. Both normalization methods retain numerical invariance to the imposed static 10-degree coordinate rotation under these disjoint splits. Calibration and scoring records come from the same walking condition; calibration-to-use gait mismatch remains untested and is stated as a limitation.

## Excluded partial work
Incomplete HistGradientBoosting/TCN rotation runs are not included. The revised manuscript follows the reviewers' alternative recommendation and scopes the quantitative robustness claims to the selected Ridge/feature framework.

## 2026-10-10 independent reruns
The frozen result tables in this directory are preserved. Independently recomputed virtual participant and summary tables from the restored NPZ are under `../reproduction_evidence/reviewer/` and numerical differences are documented in `../REPRODUCIBILITY_QA.md`.
