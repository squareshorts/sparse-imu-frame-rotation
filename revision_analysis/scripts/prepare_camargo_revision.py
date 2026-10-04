"""Run the submitted Camargo preparation code with revision-local h5py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
PREP = ROOT / (
    "submitted_final/analysis/analysis_code/strengthening/"
    "camargo_external/prepare_camargo_external.py"
)
sys.argv = [str(PREP), *sys.argv[1:]]
runpy.run_path(str(PREP), run_name="__main__")
