"""Shared test setup.

The pipeline scripts create working folders and log files when they are
imported, so every test session points them at a temporary folder first.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE = REPO_ROOT / "code"

_tmp = Path(tempfile.mkdtemp(prefix="mce-tests-"))
os.environ["PIPELINE_WORK_DIR"] = str(_tmp / "work")
os.environ["FIGURES_DIR"] = str(_tmp / "figures")

for sub in ("", "structure_pipeline"):
    path = str(CODE / sub)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_script(relative_path):
    """Import a script by file path (several names start with a digit)."""
    path = CODE / relative_path
    name = "script_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tmp_figures():
    return Path(os.environ["FIGURES_DIR"])
