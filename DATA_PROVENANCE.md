# Input provenance and boundaries of reproducibility

## Virtual IMU

The publicly released Van Criekinge et al. (2023) gait dataset is the scientific source of the virtual sensors. Original C3D raw data: https://doi.org/10.6084/m9.figshare.24192480.v1 (CC0; ~442 MB). Scientific Data DOI: 10.1038/s41597-023-02767-y.

The **exact frozen derived analysis input** is included in this release as `analysis_code/data/processed/strict_parameters.npz`, SHA-256:

`f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e`

It contains 747 cycles from 138 participants. The `X` matrix is `[cycles, 4 sensor positions, 101 phase-normalized samples, 6 inertial channels]`: the positions are pelvis, thigh, shank and foot, with three acceleration and three angular-velocity channels. `T` has nine cycle-level gait targets; `Y` stores reference joint-kinematic sequences. There are subject IDs and nonidentifying covariates, including age, speed, cycle duration, side, and trial labels.

The fixed disjoint outer fold CSV `analysis_code/protocol/outer_folds.csv` has SHA-256:

`8aaf78f5c51f0e5d28d88e17c1bbbb0563fdeb7cc09d3e39fe666d03972ca782`

**Verified:** The archive enables reproducible refitting and numerical validation of the 7x7 Ridge orientation experiment, template registration, gait-functional normalization, and reviewer-requested virtual sensitivity analyses from the exact derived NPZ.

**Not verified:** The exact original exploratory C3D-to-NPZ converter, covering gait-event selection, segment placement, virtual angular-velocity/acceleration construction, coordinate conventions, sample interpolation, and target extraction, has not been recovered. Raw-to-NPZ re-derivation is not guaranteed by this archive. Supplying an independently redesigned preprocessing converter would change the experiment and must not be represented as reproducing the historical frozen array.

## Physical IMU

The external study uses public `Camargo_levelground.h5` from StanfordMIMI/SSL_IMU (original HDF5 SHA-256 `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`). Preparation source code is at `analysis_code/strengthening/camargo_external/prepare_camargo_external.py`, producing a subset of 248 straight level-ground windows from 19 participants. The HDF5 is not redistributed and was not reprocessed during this v1.0.1 audit. The already published physical result tables remain available for source-table checks.

## Statistical and deployment interpretation

Participant bootstrap CIs represent uncertainty in mean participant RMSE after pooling Monte Carlo test rotations. They do not describe the tail of any empirically observed placement-error distribution. Ridge results at a 4.5-degree decision criterion are specific to the frozen virtual cohort and selected features. The pipeline assumes externally determined gait cycles and walking speed; it has not been validated as an autonomous continuously operating wearable system.
