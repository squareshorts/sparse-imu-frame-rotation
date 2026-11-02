# Direction-resolved sparse-IMU rotation analysis (2 November 2026)

This analysis extends the frozen v1.0.1 participant-disjoint virtual/physical Ridge study. It does not modify the published dataset or the earlier numerical results.

## Nested physical-cohort regularization

Physical-model Ridge penalties were selected in three grouped inner folds separately within each five participant-disjoint outer-training fold, among alpha = 0.01, 0.1, 1, 10, 100, 1000, 10000. For clean models the selections were alpha 100 in two folds and alpha 1000 in three; for 7.5- and 10-degree-augmented models all five folds selected alpha 100. At 10-degree testing, physical retuning changed participant-mean RMSE by -0.923 degrees (ankle dorsiflexion), +0.190 (ankle plantarflexion), -0.949 (knee stance) and -1.738 (knee swing), compared with the transferred penalty. Paired participant-bootstrap intervals excluded zero for the three improvements, but not ankle plantarflexion.

## Virtual-physical prediction response

At 10-degree isotropic-axis rotation, clean-trained mean participant prediction-shift RMS was 8.058, 8.095, 7.984, 7.915 degrees virtually for ankle dorsiflexion, ankle plantarflexion, knee stance and knee swing, versus 1.754, 2.058, 2.519, 2.753 degrees physically. The between-cohort gap reflects smaller physical prediction changes, not only larger physical unrotated error. The mean-square error identity baseline^2 + prediction_shift^2 + 2*(baseline error)*(prediction shift) held at participant level. A 9.80665 scaling of physical acceleration followed by refitted within-fold standardization and Ridge changed predictions by at most 0.000154 degrees, so uniform numerical acceleration rescaling does not resolve the sensitivity gap.

## Signed-axis and upper-tail stress

For the five-copy, 10-degree-cap Ridge model, each of four gait extrema was evaluated under 24 deterministic signed 7.5-degree conditions: 3 local axes * 2 signs * 4 sensor configurations. Among 96 endpoint/configuration comparisons, 88 met the study-defined 4.5-degree upper participant-bootstrap RMSE threshold. All eight failures were opposing-sensor rotations about local y, in both signs and for every endpoint. Knee stance flexion reached 5.711899 degrees mean participant RMSE, and a joint participant bootstrap across all 96 cases gave a 95% interval of 5.621417 to 6.157170 degrees for the maximum of condition means. The 13 of 13 success across additional random-axis training seeds and fold partitions at 7.5 degrees is conditional on that sampling distribution and cannot be interpreted as a universal orientation tolerance.

## Sources and limitations

Virtual: 747 gait cycles, 138 participants, SHA-256 `f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e`.
Physical Camargo: 248 windows, 19 participants, source SHA-256 `c5faa8b545127176765b53f88d5551a10721beea760d3893f263a42154656b0a`.
Original frozen model/data repository commit: `3609b490194670bf69f82c5a166ffdfa5aadb3f3`.

A scientific report, full executed scripts, successful-run logs, and participant/cycle-level outputs were packaged separately as `IMU_MEASUREMENT_REPRODUCIBILITY_2026-11-02.zip`. This repository note preserves the headline numerical outcomes, without redistributing original Camargo HDF5 data. These experiments apply mathematical coordinate rotations to existing sensor records; they do not validate physical removal and reattachment of sensors or provide a traceable instrument measurement-uncertainty budget.
