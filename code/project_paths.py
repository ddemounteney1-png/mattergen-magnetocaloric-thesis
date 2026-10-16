"""Where the code reads and writes files.

Every script imports its folders from here instead of hardcoding paths, so the
repository runs on any machine without editing the scripts.

Folders inside the repository (data that is committed) are fixed. Working
folders for the long-running pipelines live outside version control; by default
they go under ``work/`` at the repository root, and each can be moved with an
environment variable:

=========================  ==============================  ==================================
Variable                   Default                         Used by
=========================  ==============================  ==================================
PIPELINE_WORK_DIR          <repo>/work                     parent of the folders below
PREPROCESSING_DIR          $PIPELINE_WORK_DIR/preprocessing  code/data_parsing, SQS step
CLEAN_PREPROCESSING_DIR    $PIPELINE_WORK_DIR/clean_preprocessing  relaxation + dataset export
BASE_DATASET_DIR           $PIPELINE_WORK_DIR/base_dataset   code/base_dataset (stages 01-08)
ANALYSIS_WORK_DIR          $PIPELINE_WORK_DIR/analysis       raw SLURM logs, convex-hull data
MATTERGEN_DIR              <repo>/../mattergen               a clone of microsoft/mattergen
FIGURES_DIR                <repo>/results/figures            where figure scripts save output
=========================  ==============================  ==================================
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Committed data and results
DATA = REPO_ROOT / "data"
SOURCE_DATA = DATA / "source"
FINETUNE_DATA = DATA / "finetune"
SUPERCONDUCTOR_DATA = DATA / "superconductor"
BASE_METADATA = DATA / "base_metadata"
DOCS = REPO_ROOT / "docs"
RESULTS = REPO_ROOT / "results"
METRICS = RESULTS / "metrics"
LOSS_CURVES = RESULTS / "loss_curves"


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Figures are written here; tests point this at a temporary folder
FIGURES = _env_dir("FIGURES_DIR", RESULTS / "figures")

# Working folders (not committed)
WORK = _env_dir("PIPELINE_WORK_DIR", REPO_ROOT / "work")
PREPROCESSING_WORK = _env_dir("PREPROCESSING_DIR", WORK / "preprocessing")
CLEAN_PREPROCESSING_WORK = _env_dir("CLEAN_PREPROCESSING_DIR", WORK / "clean_preprocessing")
BASE_DATASET_WORK = _env_dir("BASE_DATASET_DIR", WORK / "base_dataset")
ANALYSIS_WORK = _env_dir("ANALYSIS_WORK_DIR", WORK / "analysis")
MATTERGEN_DIR = _env_dir("MATTERGEN_DIR", REPO_ROOT.parent / "mattergen")
