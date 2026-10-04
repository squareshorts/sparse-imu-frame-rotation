Claim audit for IEEE Sensors Journal submission

The final manuscript claims were checked programmatically against the CSV source tables in source_data/. The executable check is scripts/claim_audit.py; its current output is saved as CLAIM_AUDIT.txt.

Claims directly supported by the source tables

Virtual clean two-sensor RMSE: 0.89-1.40 deg.

Exact 10-deg per-sensor rotation, unaugmented virtual model: 8.08-8.24 deg RMSE.

Exact 10-deg rotation, matched 10-deg augmentation: 3.66-4.27 deg RMSE; largest upper 95% CI 4.456 deg.

Under the all-endpoint operating rule fixed before evaluation of the complete matrix, 10-deg augmentation qualifies through a 10-deg tested bound; no tested augmentation qualifies through 12.5 or 15 deg.

Matched 10-deg augmentation rescue at exact 10 deg: 4.46, 4.38, 3.87, and 3.91 deg across the four endpoints; every paired 95% CI excludes zero.

Clean-condition penalty from 10-deg augmentation: 1.46-1.59 deg.

Virtual template SO(3) registration at exact 10 deg: 3.19-3.92 deg RMSE, with near-exact invariance to the imposed static rotation.

Independent physical-IMU external test at exact 10 deg: unaugmented degradation 0.382, 0.482, 0.663, and 0.671 deg; every paired 95% CI excludes zero.

External matched 10-deg augmentation improves 3 of 4 endpoints with paired 95% CI excluding zero.

External template registration improves all 4 endpoints versus unaugmented 10-deg testing, with all paired 95% CIs excluding zero.

Template registration versus matched augmentation in the external cohort is inconclusive for three endpoints and favors template registration for knee swing by 0.536 deg (95% CI 0.031-1.288).

Guardrails retained in the manuscript

The 4.5-deg ceiling is an operational analysis criterion, fixed before evaluation of the complete matrix and selected within the approximately 2-5-deg gait-kinematic error range discussed in the literature. It is not presented as a universal clinical threshold.

The independent external test uses recordings from physical IMUs with controlled mathematical coordinate rotations. It is not described as a repeated physical sensor-remounting experiment.

The virtual 10-deg operating result is not extrapolated to 12.5 or 15 deg.

Augmentation is not described as universally transferable: the external plantarflexion improvement is statistically inconclusive.

Explicit frame registration requires an unlabeled calibration walking batch and trades clean-frame accuracy for stronger rotation invariance.

Current executable result: PASS.

Additional mechanically audited claims

Relative-frame mechanism: in clean-trained virtual models, a common 10-degree rotation changes RMSE by less than 1 degree for every endpoint, whereas independent adjacent-sensor rotations increase RMSE by 6.81-7.15 degrees.

Pair geometry at nominal 10 degrees per sensor: mean relative frame disagreement is 13.317 degrees, the 97.5th percentile is 19.745 degrees, and the sampled maximum approaches 20 degrees.

External data provenance: the exact physical-IMU analysis file is identified by SHA-256 in the manuscript and protocol; mathematical rotations must not be described as repeated physical remounting.