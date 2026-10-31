# R2 robustness and physical-IMU reanalysis (31 October 2026)

The original v1.0.1 dataset, manuscript and frozen source results are unchanged. These additional analyses document robustness across computational repetitions and independently reproduce the Camargo physical-IMU evaluation.

## Virtual-IMU results
- Original five-copy 10-degree test reproduced: worst endpoint upper CI 4.455609 degrees.
- Eight augmentation seeds: five passed the study-defined 4.5-degree rule at 10 degrees.
- Five alternative participant-disjoint partitions: none passed at 10 degrees.
- All 13 additional configurations passed at 7.5 degrees (post-hoc numerical observation, not a physical remounting tolerance).
- Walking-speed ablation: 10-degree unaugmented RMSE change from -0.0148 to +0.0506 degrees; augmented change from -0.0038 to +0.0314 degrees.
- Matched virtual calibration: 113 subjects and 348 identical scoring cycles; three calibration cycles improved RMSE over one for both registration methods.

## Completed external physical-IMU evaluation
- Source: [Stanford Camargo HDF5](https://github.com/StanfordMIMI/SSL_IMU/blob/main/example_usage/Camargo_levelground.h5); 23,347,200 bytes, SHA-256 `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`.
- HDF5 processed independently into 248 retained windows from 19 participants; all 540 original QC entries reproduced.
- Participant-disjoint Ridge refits reproduced all 196 rotation-matrix source result cells to within 0.0000381 degrees and all 56 frame-normalization summary cells exactly.
- At 10-degree rotation, the unaugmented physical RMSE increases were 0.382-0.671 degrees, compared with virtual increases of 6.68-7.23 degrees. The observed discrepancy remains.
- The Camargo `treadmill_speed` field is constant -1 in retained windows; it is not a usable speed feature.
- Raw-window overlap check found one pair with 31 shared samples (AB25 windows 0 and 1). Matched calibration was repeated excluding the affected scored window; 19 participants and 152 identical scoring windows remain. All 32 paired contrast means changed by at most 0.000421 degrees.
- Physical calibration benefit depends on the endpoint and normalization method. It does not demonstrate cross-condition calibration transfer.

See [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md), [physical_camargo](physical_camargo), and the [key virtual tables](tables). The full execution archive, including participant-level outputs, is attached in the project conversation. The original raw HDF5 is not redistributed.
