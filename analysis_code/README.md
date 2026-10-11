# Analysis code supplied for peer review

This directory contains the code used for the corrected orientation-robustness analyses reported in the manuscript.

## Primary virtual-IMU analysis

Required inputs:

- `data/processed/strict_parameters.npz`, the frozen derived input from the public Van Criekinge et al. source dataset. The exact verified frozen NPZ is included in v1.0.1. See top-level `DATA_PROVENANCE.md`.
- `protocol/outer_folds.csv`, included in this archive. Its SHA-256 is fixed by the original confirmatory protocol.

Run `python scripts/verify_processed_input.py` from the repository root before fitting any models. This hash-verification step distinguishes reproducing models from reproducing preprocessing.

The corrected full matrix uses participant-disjoint outer folds, five weight-normalized augmented copies plus the clean cycle, cap-specific Ridge penalties, 50 Monte Carlo test rotations for each nonzero condition, and participant RMSE computed by pooling squared errors across cycles and Monte Carlo realizations before taking the square root.

From the `analysis_code` directory, the principal scripts are:

```text
python strengthening/run_rotation_matrix_v2.py
python strengthening/summarize_rotation_matrix_v2.py
python strengthening/run_orientation_baselines_v2.py
```

Fold-wise variants are also included for parallel execution.

## Independent physical-IMU external test

The Camargo source HDF5 file is not redistributed. Obtain it from the public source dataset, then prepare the analysis representation with:

```text
python strengthening/camargo_external/prepare_camargo_external.py --input <Camargo_levelground.h5> --outdir <external_dir>
```

Run the five subject-disjoint folds for the matrix and frame-normalization analyses:

```text
python strengthening/camargo_external/run_camargo_matrix_v3_pooled_fold.py --fold 1 --data <external_dir>/camargo_external.npz --outdir <external_dir>/matrix_v3_pooled
python strengthening/camargo_external/run_camargo_baselines_v3_pooled_fold.py --fold 1 --data <external_dir>/camargo_external.npz --outdir <external_dir>/baselines_v3_pooled
```

Repeat `--fold 1` through `--fold 5`, then summarize with:

```text
python strengthening/camargo_external/summarize_camargo_external_v3_pooled.py --base <external_dir>
```

The primary external endpoint is the same pooled squared-error participant RMSE definition used in the virtual analysis. The secondary ground-reaction-force analysis is in `run_camargo_grf_secondary.py`.

## Dependencies

Python 3.11+ with NumPy, pandas, SciPy, and scikit-learn. The original confirmatory source modules required by these runners are included under `src/confirmatory`.

## Important implementation notes

- Training augmentation is weight normalized: the clean cycle and each of five rotated copies receive weight `1/6`.
- The StandardScaler is also fit with those sample weights.
- Clean and 2.5-degree augmentation use Ridge `alpha=10`; augmentation caps of 5 degrees or more use `alpha=100`, as selected in grouped inner validation.
- Frame-normalization methods use their reselected Ridge penalty (`alpha=100` in all five primary folds).
- Test rotations are exact magnitudes about independent isotropic axes and are fixed within participant for each realization.
