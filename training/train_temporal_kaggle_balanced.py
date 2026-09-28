from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get(
        "SATQUERYAI_PROJECT_ROOT",
        SCRIPT_DIR.parent,
    )
).resolve()

DATA_ROOT = Path(
    os.environ.get(
        "SATQUERYAI_DATA_ROOT",
        PROJECT_ROOT / "data",
    )
).resolve()

OUTPUT_ROOT = Path(
    os.environ.get(
        "SATQUERYAI_OUTPUT_ROOT",
        PROJECT_ROOT,
    )
).resolve()

# Kaggle bundle contains temporal_change_vqa.py at its dataset root.
# The environment variable is preferred; otherwise use the established
# SatQueryAI temporal bundle path when running on Kaggle.
_bundle_env = os.environ.get("SATQUERYAI_BUNDLE_ROOT")
if _bundle_env:
    BUNDLE_ROOT = Path(_bundle_env).resolve()
elif Path("/kaggle/input/datasets/rajdeepbiswas1/satqueryai-temporal-change-vqa-bundle").exists():
    BUNDLE_ROOT = Path(
        "/kaggle/input/datasets/rajdeepbiswas1/"
        "satqueryai-temporal-change-vqa-bundle"
    ).resolve()
else:
    BUNDLE_ROOT = PROJECT_ROOT

TRAIN_MANIFEST = (
    DATA_ROOT
    / "processed"
    / "temporal_split"
    / "train_manifest.csv"
)

VAL_MANIFEST = (
    DATA_ROOT
    / "processed"
    / "temporal_split"
    / "val_manifest.csv"
)

FEATURE_DIR = (
    DATA_ROOT
    / "processed"
    / "temporal_features"
)

TRAIN_QUESTION_DIR = (
    DATA_ROOT
    / "processed"
    / "temporal_question_embeddings"
)

VAL_QUESTION_DIR = (
    DATA_ROOT
    / "processed"
    / "temporal_val_question_embeddings"
)

CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "models"
    / "checkpoints"
    / "temporal_change_vqa_balanced"
)
