# v1.0.1 GitHub/Zenodo release instructions

1. Use a **clean local clone** of `https://github.com/squareshorts/sparse-imu-frame-rotation`. Do not create a new release from this patch ZIP alone: Zenodo archives the **repository snapshot at the GitHub tag**, not uploaded release-asset contents.
2. Extract `sparse_imu_v1.0.1_RELEASE_PATCH.zip` **into the clone root**, overwriting only the named files. This preserves all other v1.0.0 repository files, including published figures. Alternatively use the accompanying `apply_imu_repro_patch.ps1` helper from outside the clone.
3. Run from the repository root:
   ```bash
   python scripts/verify_processed_input.py
   python scripts/claim_audit.py
   python scripts/verify_reproduction_evidence.py
   ```
   All must pass. Confirm `git status` shows the expected new data and documentation only. The exact NPZ must be tracked (`git ls-files analysis_code/data/processed/strict_parameters.npz`).
4. Commit and push the verified update to the GitHub repository. Then publish release tag **`v1.0.1`** with title **`v1.0.1 - Frozen-input Ridge reproducibility and portable rerun`**. Avoid publishing from a branch/tag lacking the input NPZ.
5. The enabled GitHub-Zenodo integration should create a new **version-specific DOI**. Verify the published Zenodo archive contains `analysis_code/data/processed/strict_parameters.npz` and that it resolves to v1.0.1. Retain the v1.0.0 historical record.
6. Only after the actual DOI exists, surgically update the manuscript's data/code availability citations in **both** clean and marked source PDFs/TeX, review reviewer-response references, and rebuild the submission package.

**DO NOT CLAIM:** original raw C3D-to-NPZ conversion was rebuilt, or raw physical Camargo HDF5 results were refitted in this audit. Model-level replication of the complete virtual analyses is established using the exact bundled derived input.
