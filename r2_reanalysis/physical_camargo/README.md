# Completed Camargo physical-IMU reanalysis

The source HDF5 is the public Stanford `example_usage/Camargo_levelground.h5`, SHA-256 `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`. The raw dataset is not redistributed.

A fresh preprocessing run reproduced 248 windows from 19 participants and the archived 540-window QC manifest. Independent physical Ridge refits reproduced 196 source rotation-matrix summary cells to within 0.0000381 degrees, and all 56 orientation-normalization summary cells exactly.

At an imposed 10-degree rotation, the unaugmented physical error increased by 0.382-0.671 degrees across endpoints. The four baseline RMSEs at zero rotation were 4.145, 4.320, 5.187 and 5.898 degrees. The physical results do not establish the virtual 4.5-degree operating limit.

A raw overlap check identified one pair of partially overlapping windows (AB25 windows 0 and 1, 31 common samples). The matched calibration-size comparison excluding the affected scored window retains 19 participants and the same 152 scored windows for all calibration sizes (1, 3, 5). The maximum alteration in any of the 32 paired mean calibration effects was 0.000421 degrees.

All summary outputs are in this folder. The full reproducibility archive with code, participant-level results, executed logs, and source checks is delivered separately in the conversation as `IMU_R2_RESULTS_COMPLETE_2026-10-31.zip`.

The physical raw source provides no usable recorded walking-speed covariate (`treadmill_speed=-1` throughout retained windows); this point is accounted for in the report.
