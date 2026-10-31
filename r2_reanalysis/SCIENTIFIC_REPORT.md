# IEEE Sensors Journal R2: computational robustness audit

**Manuscript:** Operating Limits of Ridge-Based Sparse IMU Gait Estimation Under Sensor-Frame Rotation  
**Original repository:** `squareshorts/sparse-imu-frame-rotation`, commit `3609b490194670bf69f82c5a166ffdfa5aadb3f3`  
**Analysis input:** authenticated virtual dataset `strict_parameters.npz`, SHA-256 `f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e`  
**Sample:** 747 cycles from 138 participants. Frozen input and original result tables were not edited.  
**Inference:** participant-level pooled RMSE, 50 static rotation-axis Monte Carlo realizations for nonzero test angles, 5,000 participant-bootstrap resamples; pass criterion upper 95% CI <= 4.5 degrees for every evaluated endpoint and test angle in the envelope.

## Main finding

The specific **10-degree operating-limit decision is unstable** under changes to augmentation seed and participant-disjoint training/test partition. It passed in **5 of 8** independent augmentation-seed repetitions of the fixed five-fold partition, and **0 of 5** alternative participant-disjoint five-fold partitions. **All 13** additional configurations satisfied the study-defined criterion at every tested angle from 0 through **7.5 degrees**. This was the largest tested magnitude consistently satisfying the rule across those internal sensitivity analyses; the maximum worst-endpoint upper CI at or below 7.5 degrees across all 13 configurations was **4.1740 degrees**.

This empirical stability statement is conditional on the study's Ridge/feature system, virtual signals, fixed 10-degree training-augmentation cap, five augmented copies, Monte Carlo test protocol and 4.5-degree decision threshold. It is **not a physical IMU remounting tolerance**, a clinical accuracy standard, or an independent confirmation cohort. The 7.5-degree stability observation is retrospective, so it should be reported as a sensitivity finding and not as an independently established universal bound.

## Operating decision by repetition

| Analysis | Run | Worst upper 95% CI at/below 10 deg | 10-deg decision | Largest passing tested angle |
|---|---:|---:|---|---:|
| Frozen folds, new augmentation seeds | 1 | 4.490053 deg | Pass | 10 deg |
| Frozen folds, new augmentation seeds | 2 | 4.462230 deg | Pass | 10 deg |
| Frozen folds, new augmentation seeds | 3 | 4.513385 deg | Fail | 7.5 deg |
| Frozen folds, new augmentation seeds | 4 | 4.481888 deg | Pass | 10 deg |
| Frozen folds, new augmentation seeds | 5 | 4.456161 deg | Pass | 10 deg |
| Frozen folds, new augmentation seeds | 6 | 4.529864 deg | Fail | 7.5 deg |
| Frozen folds, new augmentation seeds | 7 | 4.470185 deg | Pass | 10 deg |
| Frozen folds, new augmentation seeds | 8 | 4.510973 deg | Fail | 7.5 deg |
| Alternative participant-disjoint partition | 1 | 4.518519 deg | Fail | 7.5 deg |
| Alternative participant-disjoint partition | 2 | 4.604428 deg | Fail | 7.5 deg |
| Alternative participant-disjoint partition | 3 | 4.514771 deg | Fail | 7.5 deg |
| Alternative participant-disjoint partition | 4 | 4.531629 deg | Fail | 7.5 deg |
| Alternative participant-disjoint partition | 5 | 4.518290 deg | Fail | 7.5 deg |

The limiting endpoint is typically knee stance flexion; one alternative partition is limited by knee swing flexion. Pass fractions are **descriptive over computational repetitions**, not binomial estimates from independent participant cohorts.

## Exact original-result reproduction

Before interpreting the new analyses, the frozen five-copy model was refitted and checked at 0- and 10-degree test rotations. The eight source mean/CI cells (four endpoints by two angles) reproduced to numerical precision: maximum absolute mean difference **0.0 degrees**, maximum upper/lower CI difference **4.4 x 10^-16 degrees**. The original worst upper confidence limit at 10 degrees was **4.455609 degrees**, just **0.044391 degrees** below the 4.5-degree criterion. The original five-copy result passed, but the expanded sensitivity tests show it is not stable.

## Walking-speed ablation

The virtual Ridge models were refitted after deleting only the walking-speed feature, preserving gait-cycle duration and all directional IMU features; alpha, training augmentation, outer folds, and test rotation streams were held constant. At 10-degree test rotation, removing speed changed the participant-mean RMSE by **-0.0148 to +0.0506 degrees** across four unaugmented endpoints. For five-copy augmentation, the change ranged from **-0.0038 to +0.0314 degrees**. The unaugmented perturbation penalty remained approximately **6.63-7.22 degrees**, so walking-speed availability does not explain most of the much smaller rotation effect in the physical cohort. A separate model and signal-domain mechanism is still needed to explain the virtual/physical discrepancy.

Ablation evaluations were performed at **0 and 10 degrees only**. The two-angle pass/fail checks in the source runner are not complete 0-10-degree operating-envelope decisions.

## Common-subject, common-record virtual calibration

Both template registration and gait-functional calibration were evaluated using **the same 113 eligible held-out participants and 348 scored cycles** for one versus three calibration cycles. The first three shuffled cycles per participant form the nested calibration pool; the scored set excludes all three cycles in every condition. At 0 degrees, adding calibration cycles reduced RMSE in all four endpoints:

| Endpoint | Template registration 3 minus 1 cycle | Functional calibration 3 minus 1 cycle |
|---|---:|---:|
| Ankle dorsiflexion | -0.356 deg | -0.332 deg |
| Ankle plantarflexion | -0.337 deg | -0.346 deg |
| Knee stance flexion | -0.301 deg | -0.251 deg |
| Knee swing flexion | -0.292 deg | -0.447 deg |

Individual paired participant-bootstrap 95% intervals exclude zero for these contrasts (not adjusted for multiplicity). The same mean differences appear at 10 degrees due to numerical invariance to imposed coordinate rotation. This design removes the changing-participant and changing-evaluation-set confounds from the previous calibration-size comparison; it does not test calibration generalization across speeds, tasks, or remounting conditions.

## External physical-IMU protocol audit

The archived source manifest has **540** Camargo source windows, **248 included windows** from **19 participants** (9-15 included windows per participant), **262** windows excluded for turning, and **30** from excluded participant AB18. There are no duplicate subject/window IDs in the manifest. Source-code inspection confirms that models were **refitted on the physical training participants**, transferring only Ridge regularization from the virtual study. The preparation code sets the physical walking-speed feature to constant zero. The physical cohort's baseline RMSEs are higher than those of the virtual cohort, so the virtual 4.5-degree threshold cannot be directly imposed on the physical evaluation.

The raw file `Camargo_levelground.h5` was **not included** in the supplied ZIP. Consequently, three tests could not be performed: source-HDF5 checksum verification and raw-window overlap checking; independent execution of the physical-cohort model on the prepared tensor; and the physical-cohort common-subject/common-window calibration comparison. No physical signal or participant values have been imputed.

## Implications for manuscript revision

1. Replace the unqualified 10-degree operating-limit conclusion with a criterion-dependent numerical result and directly report the augmentation-seed and partition sensitivity.
2. Describe 7.5 degrees as the largest tested magnitude consistently satisfying the **internal** numerical rule across these sensitivity repetitions, while retaining the original result as an observed initial configuration.
3. Incorporate the walking-speed ablation as evidence that speed availability contributes little to the virtual/physical discrepancy within the virtual pipeline; keep the cause of that discrepancy unresolved.
4. Incorporate the matched virtual calibration-size contrasts. Explain the calibration time/condition requirements and distinguish frame-rotation invariance from anatomical frame recovery.
5. Explicitly state the external physical-model refitting procedure and report the raw physical data analyses as incomplete until the source HDF5 is obtained.
6. Do not represent repeated folds of the previously examined virtual cohort as untouched independent confirmation. A new independent cohort or a substantially more limited operating claim is required to address selection uncertainty.

## Contents

- `MANIFEST.json` records the authenticated data hash, reference reproduction, analysis counts, and missing physical HDF5.
- `tables/stability_decisions.csv`, `stability_pass_fractions.csv`, `stability_endpoint_summary.csv`, and `stability_participant_rmse.csv` record the complete 13-scenario internal robustness evidence (35,880 participant-level scenario/angle/endpoint observations).
- `tables/speed_*` and `calibration_common_virtual_*` preserve the two new paired analyses and their participant-level products.
- `tables/physical_*`, `virtual_physical_relative_changes.csv`, and `virtual_sensor_characteristics.csv` document the source-available physical audit and sensor comparisons.
- `code/` contains the exact Python execution scripts; `logs/` contains the successful-run logs. The aborted all-in-one execution and its incomplete intermediate tables are deliberately **excluded**.

## Cleanup

The delivered ZIP contains **no virtual environment, duplicate repository checkout, or obsolete input archive**. Keep the original GitHub repository and original raw/processed research data. On the Windows computer, the nonworking `C:\Users\mirro\IMU_R2_work\.venv` may be removed after this result archive has been downloaded and verified. The earlier locally generated `IMU_R2_results.zip` may also be removed if it was created during the failed PowerShell run; it does not contain the successful analyses performed here. Deletion from the user's Windows filesystem was not performed remotely.