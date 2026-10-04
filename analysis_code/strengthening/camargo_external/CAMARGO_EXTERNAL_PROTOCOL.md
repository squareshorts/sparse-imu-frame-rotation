# Camargo independent physical-IMU stress-test protocol

Protocol date: 2026-09-07.

## Purpose
Independently test whether static sensor-coordinate-frame perturbations degrade sparse gait estimation when the base signals come from physical IMUs, and compare rotation augmentation with label-free frame normalization. This provides an external test of the frame-rotation robustness mechanism; it does not redefine the absolute virtual-study operating threshold and it is not a repeated physical removal/remounting experiment.

## Source and analysis set
- Source file: `Camargo_levelground.h5`
- SHA-256: `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`
- Exclude AB18 because the upstream source code flags that recording as erroneous.
- Retain only windows labeled as straight level-ground walking at every sample.
- Require finite right-thigh, right-shank, right-foot IMU channels and finite right-knee/right-ankle kinematics.
- Phase-normalize retained windows to 101 gait-cycle points using the supplied heel-strike phase.
- Final set: 248 windows from 19 participants.

## Sensors and targets
Primary sensor pairs are right thigh + right shank for knee endpoints and right shank + right foot for ankle endpoints. Each sensor contributes three-axis accelerometer and three-axis gyroscope signals. Targets use the same phase windows as the virtual analysis: knee stance flexion maximum (0--50%), knee swing flexion maximum (55--100%), ankle dorsiflexion maximum (15--55%), and ankle plantarflexion minimum (45--75%). Knee sign is harmonized with the virtual-study flexion-positive convention before endpoint extraction.

## Model transfer
Use the frozen engineered representation and Ridge family from the virtual study. There is no external hyperparameter search. Transfer the virtual-selected penalties exactly:
- unaugmented and 2.5-degree augmentation: `alpha = 10`;
- augmentation caps >=5 degrees: `alpha = 100`;
- training-template SO(3) registration: `alpha = 100`;
- gait-functional frame normalization: `alpha = 100`.

Models are fitted only on training participants within each of five fixed participant-disjoint external folds.

## Orientation matrix
Training augmentation caps and exact per-sensor test rotations are 0, 2.5, 5, 7.5, 10, 12.5, and 15 degrees. For each nonzero test rotation, use 50 deterministic participant-consistent Monte Carlo rotation-axis realizations. Within a realization, each participant/sensor receives one static random-axis rotation of the exact requested magnitude, shared across that participant's windows. Training augmentation uses the clean training window plus five participant-consistent rotated copies sampled uniformly from zero to the training cap, with the same total-weight normalization used by the virtual analysis.

## Frame-normalization comparators
1. Training-template proper SO(3) registration by Kabsch alignment of phase-aligned acceleration/angular-velocity signatures to the outer-training template.
2. Gait-functional frame normalization from mean acceleration direction and dominant gyroscope covariance, with temporal sign anchoring and a right-handed third axis.

Both may use an unlabeled gait calibration batch from a held-out participant; no kinematic or force target is used for calibration.

## Primary inference
The direct external comparison at the central stress condition uses exact 10-degree rotation and matched 10-degree augmentation. Report:
1. paired unaugmented degradation relative to zero rotation;
2. paired improvement of matched augmentation relative to unaugmented testing;
3. paired improvement of each frame-normalization method relative to unaugmented testing;
4. paired comparison of frame normalization with matched augmentation;
5. clean-domain cost and rotation-invariance diagnostics.

The virtual 4.5-degree criterion is not reapplied as an external pass/fail threshold because the external cohort has different baseline difficulty and source preprocessing.

## RMSE and uncertainty
Participant-level RMSE is the unit of inference. At nonzero rotations, pool squared errors across all retained participant windows and all 50 rotation-axis realizations before taking the square root. This pooled-squared-error definition is the primary metric and exactly matches the virtual analysis. A realization-averaged RMSE calculation is retained only as a sensitivity analysis.

Mean participant RMSE and paired deltas use 95% percentile bootstrap confidence intervals with 5,000 participant-level resamples and seed 20260907.
