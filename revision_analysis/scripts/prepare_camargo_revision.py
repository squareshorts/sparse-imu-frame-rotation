"""Run the submitted Camargo preparation code with revision-local h5py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# Standard installed h5py is used; no original-workspace vendor directory required.

PREP = ROOT / "analysis_code/strengthening/camargo_external/prepare_camargo_external.py"
sys.argv = [str(PREP), *sys.argv[1:]]
runpy.run_path(str(PREP), run_name="__main__")
