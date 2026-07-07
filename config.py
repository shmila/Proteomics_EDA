"""
Centralized path configuration for the WSI Protein Expression pipeline.

Set the THESIS_DATA_DIR environment variable to point to the local
Thesis directory, or modify DEFAULT_DATA_DIR below.

Expected external directory structure:
    DATA_DIR/
    ├── 2021-01-17/
    │   └── Tiffs/           # Whole-slide images (per patient)
    ├── CSVs/                # Proteomics data
    └── dataset/             # Generated tile datasets (created by the pipeline)
"""

from pathlib import Path
import os

DEFAULT_DATA_DIR = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis"
DATA_DIR = Path(os.environ.get("THESIS_DATA_DIR", DEFAULT_DATA_DIR))
TIFFS_DIR = DATA_DIR / "2021-01-17" / "Tiffs"
CSVS_DIR = DATA_DIR / "CSVs"
DATASET_DIR = DATA_DIR / "dataset"
