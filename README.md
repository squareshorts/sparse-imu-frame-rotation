# Operating Limits of Ridge-Based Sparse IMU Gait Estimation Under Sensor-Frame Rotation

Reproducibility archive for IEEE Sensors Journal manuscript Sensors-113886-2026.

The archive contains the final analysis code, source tables, figure-generation scripts, participant-level summaries, and completed reviewer-requested sensitivity analyses.

The virtual benchmark evaluates sparse two-IMU Ridge estimation of sagittal knee and ankle extrema under mathematical sensor-frame rotation. Knee estimation uses thigh and shank signals; ankle estimation uses shank and foot signals.

## Repository structure

- analysis_code/ - final analysis implementation
- source_data/ - tables supporting manuscript results and figures
- scripts/ - figure generation and numerical claim audit
- figures/ - publication figures
- revision_analysis/ - completed reviewer-requested sensitivity analyses
- CLAIM_AUDIT.md - numerical claim audit

## External data

The large source datasets are not redistributed.

The physical-IMU analysis uses Camargo_levelground.h5 from StanfordMIMI/SSL_IMU.

SHA-256:
c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a

The virtual gait signals derive from the public Van Criekinge et al. gait dataset.

## Scope

The reported operating envelopes apply to the selected Ridge estimator and 330-feature representation under mathematical coordinate-frame rotation. They are not empirical tolerance limits for repeated physical sensor removal and remounting.

## Verification

Run:

python scripts/claim_audit.py

Expected output:

CLAIM AUDIT: PASS
