# R2 robustness reanalysis (31 October 2026)

This directory contains additional analysis of the frozen virtual-IMU cohort and an audit of the physical-IMU protocol. **The original submitted results and release remain unchanged.**

- Original fixed-fold, five-copy virtual result reproduced exactly (worst 10° upper 95% CI 4.455609°).
- Eight independent augmentation seeds: 5/8 satisfied the 4.5° decision rule at 10°.
- Five alternative participant-disjoint fold assignments: 0/5 satisfied the rule at 10°.
- All thirteen sensitivity configurations satisfied the rule at 7.5°; this is a *retrospective internal sensitivity finding*, not a validated physical remounting tolerance.
- Walking-speed ablation and common-evaluation-set virtual calibration comparisons are reported in the tables and scientific report.
- The physical audit checked source code and archived QC records only. Raw physical data were **not** re-executed.

## Source file for completing physical analyses

The exact original public file is [StanfordMIMI/SSL_IMU/example_usage/Camargo_levelground.h5](https://github.com/StanfordMIMI/SSL_IMU/blob/main/example_usage/Camargo_levelground.h5), 23,347,200 bytes, expected SHA-256 `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`.

The data file is hosted in the Stanford repository, not in this one. The GitHub connector used for this update cannot deliver that binary to the scientific runtime, and local outbound requests to GitHub fail DNS resolution. It has consequently **not been downloaded, hash-verified, or processed here**. The physical raw-window overlap audit and common-evaluation-set calibration rerun remain pending.

Full executed code, logs, participant tables and manifest are bundled in the conversation's verified reanalysis ZIP. This repository directory presently publishes the scientific report and key numerical tables, not the entire execution archive.
