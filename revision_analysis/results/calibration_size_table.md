# Disjoint calibration-size endpoint results

Each cell is mean participant RMSE [95% participant-bootstrap CI] in degrees. The 10-degree condition pools squared errors over 50 exact-rotation realizations before participant RMSE is calculated. The complete machine-readable results are in `calibration_size_sensitivity.csv`.

| Cohort | Method | k | Test rotation | Participants | Ankle DF | Ankle PF | Knee stance | Knee swing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| physical_camargo | gait_functional_calibration | 1 | 0° | 19 | 4.687 [3.506, 6.139] | 6.017 [4.407, 7.876] | 4.732 [3.747, 5.866] | 5.204 [4.431, 5.975] |
| physical_camargo | gait_functional_calibration | 1 | 10° | 19 | 4.687 [3.530, 6.149] | 6.017 [4.402, 7.890] | 4.732 [3.760, 5.859] | 5.204 [4.469, 5.972] |
| physical_camargo | gait_functional_calibration | 3 | 0° | 19 | 4.256 [3.010, 5.908] | 4.819 [3.756, 6.251] | 4.650 [3.602, 5.805] | 4.650 [3.779, 5.517] |
| physical_camargo | gait_functional_calibration | 3 | 10° | 19 | 4.256 [3.030, 5.837] | 4.819 [3.759, 6.312] | 4.650 [3.575, 5.772] | 4.650 [3.791, 5.551] |
| physical_camargo | gait_functional_calibration | 5 | 0° | 19 | 3.649 [2.846, 4.491] | 4.728 [3.601, 6.347] | 4.815 [3.678, 6.107] | 4.916 [4.047, 5.834] |
| physical_camargo | gait_functional_calibration | 5 | 10° | 19 | 3.649 [2.849, 4.490] | 4.728 [3.597, 6.208] | 4.815 [3.699, 6.096] | 4.916 [4.019, 5.831] |
| physical_camargo | template_registration | 1 | 0° | 19 | 3.894 [3.092, 4.767] | 3.889 [3.317, 4.497] | 4.525 [3.496, 5.657] | 4.269 [3.542, 5.142] |
| physical_camargo | template_registration | 1 | 10° | 19 | 3.894 [3.088, 4.725] | 3.889 [3.321, 4.492] | 4.525 [3.491, 5.710] | 4.269 [3.555, 5.149] |
| physical_camargo | template_registration | 3 | 0° | 19 | 3.754 [2.950, 4.579] | 3.991 [3.458, 4.551] | 4.565 [3.542, 5.745] | 4.437 [3.641, 5.335] |
| physical_camargo | template_registration | 3 | 10° | 19 | 3.754 [2.959, 4.562] | 3.991 [3.454, 4.549] | 4.565 [3.531, 5.705] | 4.437 [3.637, 5.299] |
| physical_camargo | template_registration | 5 | 0° | 19 | 3.678 [2.912, 4.465] | 4.007 [3.446, 4.570] | 4.702 [3.642, 5.834] | 4.659 [3.797, 5.640] |
| physical_camargo | template_registration | 5 | 10° | 19 | 3.678 [2.925, 4.483] | 4.007 [3.446, 4.570] | 4.702 [3.680, 5.840] | 4.659 [3.835, 5.634] |
| virtual | gait_functional_calibration | 1 | 0° | 136 | 3.375 [3.055, 3.721] | 3.815 [3.391, 4.259] | 4.253 [3.871, 4.666] | 4.426 [4.018, 4.859] |
| virtual | gait_functional_calibration | 1 | 10° | 136 | 3.375 [3.048, 3.718] | 3.815 [3.394, 4.259] | 4.253 [3.862, 4.654] | 4.426 [4.026, 4.841] |
| virtual | gait_functional_calibration | 3 | 0° | 113 | 3.113 [2.708, 3.548] | 3.364 [2.920, 3.834] | 3.962 [3.559, 4.375] | 3.975 [3.545, 4.418] |
| virtual | gait_functional_calibration | 3 | 10° | 113 | 3.113 [2.707, 3.542] | 3.364 [2.930, 3.842] | 3.962 [3.575, 4.364] | 3.975 [3.540, 4.440] |
| virtual | template_registration | 1 | 0° | 136 | 3.413 [3.076, 3.775] | 3.663 [3.260, 4.079] | 4.148 [3.774, 4.563] | 4.161 [3.753, 4.581] |
| virtual | template_registration | 1 | 10° | 136 | 3.413 [3.070, 3.775] | 3.663 [3.270, 4.068] | 4.148 [3.759, 4.546] | 4.161 [3.762, 4.595] |
| virtual | template_registration | 3 | 0° | 113 | 3.055 [2.658, 3.476] | 3.130 [2.718, 3.553] | 3.719 [3.350, 4.076] | 3.836 [3.416, 4.279] |
| virtual | template_registration | 3 | 10° | 113 | 3.055 [2.666, 3.468] | 3.130 [2.715, 3.566] | 3.719 [3.365, 4.086] | 3.836 [3.424, 4.268] |

The calibration records are excluded from scoring. Cohort eligibility and scored-record counts are in `calibration_audit.csv` and `calibration_size_sensitivity.csv`; the exact nested split is in `calibration_split_manifest.csv`. Differences between k values reflect both the calibration size and the changing evaluation set, including virtual participant loss at k=3.
