# v1.0.1 reproducibility audit (2026-10-10)

The following checks were actually executed using the exact frozen `strict_parameters.npz` in a fresh patched workspace. This report distinguishes original-model refitting from inspection of archived result tables.

| Gate | Result | Detail |
| --- | --- | --- |
| Derived data SHA-256 | PASS | `f75bc8ca9c9cad0a6d4b853b7fd55d3637e5a485dbfdb9f27fa18405cdd2db9e` |
| Frozen fold SHA-256 | PASS | `8aaf78f5c51f0e5d28d88e17c1bbbb0563fdeb7cc09d3e39fe666d03972ca782`; 747 cycles, 138 participants, five disjoint folds |
| Original numerical claim audit | PASS | `python scripts/claim_audit.py` |
| Full Ridge augmentation/test matrix | **PASS (model rerun)** | 5 outer folds, 70 fitted Ridge models, 196 cells, max difference in mean RMSE and CI bounds < 1e-9 degrees |
| Template + gait-functional frame normalization | **PASS (model rerun)** | 5 outer folds, 56 cells, max difference in mean RMSE and CI bounds < 1e-9 degrees |
| Augmentation-copy sensitivity (1,3,5,10 copies) | **PASS (model rerun)** | 32 result cells; max summary numeric difference 2.65061491e-06 degrees |
| Disjoint calibration-size sensitivity: virtual subset | **PASS (model rerun)** | 32 independently rerun virtual summary cells and 48 unchanged archived physical summary cells; maximum numerical difference 3.88688337e-06 degrees; virtual participant table independently regenerated |
| Full physical-IMU model rerun | NOT RUN | Requires independently downloaded checksum-matched Camargo HDF5 |
| Original raw C3D to exact NPZ conversion | NOT VERIFIED | Historical exploratory preprocessing implementation missing; exact derived input is supplied instead |

**Executed environment:** Python 3.13.5; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0; h5py 3.15.1; matplotlib 3.10.8.

The versioned reference-source CSVs were not changed. New reproductions are in `reproduction_evidence/` and are identified as distinct artifacts. Reruns with other numerical library versions may differ at microdegree precision; the supplied comparison script defaults to a 0.005-degree tolerance, which is a numerical reproduction criterion and not a clinical decision rule.
