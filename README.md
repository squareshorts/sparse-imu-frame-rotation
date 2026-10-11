# Operating Limits of Ridge-Based Sparse IMU Gait Estimation Under Sensor-Frame Rotation

Reproducibility archive for **IEEE Sensors Journal** manuscript **Sensors-113886-2026**.

**v1.0.1 update:** The exact frozen virtual-IMU analysis dataset is now included, allowing all five participant-disjoint outer Ridge folds and both frame-normalization comparators to be refit and independently checked against the reported results. Reviewer-requested virtual sensitivity analyses have also been rerun. The numerical manuscript source tables remain unchanged.

## Reproduction levels — do not conflate them

1. **Input provenance:** Included derived virtual IMU file `analysis_code/data/processed/strict_parameters.npz`, verified by SHA-256 and checked against the original 747-cycle, 138-participant frozen fold manifest.
2. **Model-level re-execution: VERIFIED.** Refitting 70 Ridge models and recomputing the full virtual training-by-test rotation matrix matches 196 archived result cells, including participant-bootstrap intervals. The two calibration/registration comparators match another 56 cells. See `reproduction_evidence/manifest.json`.
3. **Reviewer-requested virtual sensitivity re-execution: VERIFIED.** Augmentation-copy (32 cells) and disjoint virtual calibration-size (32 cells; physical 48 archived cells preserved) analyses agree with the frozen outputs to within 0.000004 degrees at summary level.
4. **Raw C3D-to-derived-input reconstruction: NOT VERIFIED.** The original exploratory preprocessing implementation is unavailable. The included NPZ is the authenticated entry point for model replication; no claim is made that it can be recreated bit-for-bit from public raw C3D recordings. See `DATA_PROVENANCE.md`.
5. **External physical IMU rerun: NOT PERFORMED HERE.** Physical Camargo input is not redistributed. Archived physical results and its preparation/analysis source are supplied; the source HDF5 checksum is documented.

## Quick integrity and source-table checks

Use Python 3.13.5 and dependencies in `requirements-reproduction.txt` for the verified environment, or use the broader `requirements.txt` for compatible environments.

```bash
python -m pip install -r requirements-reproduction.txt
python scripts/verify_processed_input.py
python scripts/claim_audit.py
python scripts/verify_reproduction_evidence.py
```

The evidence verification compares stored rerun products to original frozen source tables; it **does not** refit models. To reproduce model computations independently, run the fold scripts below.

## Refit all five folds (the verification that matters)

From this repository root, run these commands, each for folds 1 through 5:

```bash
python analysis_code/strengthening/run_rotation_matrix_v2_fold.py --fold 1
python analysis_code/strengthening/run_rotation_matrix_v2_fold.py --fold 2
python analysis_code/strengthening/run_rotation_matrix_v2_fold.py --fold 3
python analysis_code/strengthening/run_rotation_matrix_v2_fold.py --fold 4
python analysis_code/strengthening/run_rotation_matrix_v2_fold.py --fold 5
python analysis_code/strengthening/summarize_rotation_matrix_v2.py
python scripts/compare_recomputed_results.py
```

Repeat with `run_orientation_baselines_v2_fold.py --fold {1..5}`, then:

```bash
python analysis_code/strengthening/summarize_orientation_baselines_v2.py
python scripts/compare_recomputed_results.py --include-normalization
```

Model output is written under `analysis_code/strengthening/rotation_matrix_v2/` and `analysis_code/strengthening/orientation_baselines_v2/`. Reference reruns, generated during the pre-release QA, are preserved separately in `reproduction_evidence/virtual/`. Thus presence of stored evidence is never mistaken for a fresh fit.

Reviewers' virtual augmentation-copy and calibration-size experiments can be rerun via:

```bash
python revision_analysis/scripts/run_augmentation_copy_sensitivity.py
python revision_analysis/scripts/calibration_size_sensitivity.py --cohort virtual
python revision_analysis/scripts/calibration_size_sensitivity.py --summarize
```

The original reviewer-analysis CSVs are retained unmodified in `revision_analysis/results/` (recomputed values in `reproduction_evidence/reviewer/`).

## Scope and datasets

The two-sensor 330-feature Ridge representation uses thigh-shank IMUs for knee and shank-foot for ankle, plus externally supplied walking speed and cycle duration. Test perturbations are mathematical sensor-coordinate rotations, not repeated physical sensor remounting. Virtual 4.5-degree RMSE operating bounds are not physical remounting tolerance bounds.

- Van Criekinge et al. (2023), *Scientific Data* 10, 852. DOI: 10.1038/s41597-023-02767-y. CC0 source C3D DOI: 10.6084/m9.figshare.24192480.v1.
- Camargo physical-IMU source: StanfordMIMI/SSL_IMU, `Camargo_levelground.h5`; SHA-256 in `DATA_PROVENANCE.md`.

See `DATA_LICENSE.md`, `REPRODUCIBILITY_QA.md`, and `reproduction_evidence/manifest.json`.
