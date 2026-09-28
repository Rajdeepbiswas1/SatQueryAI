from pathlib import Path
import json

import pandas as pd


PROJECT = Path(r"D:\SatQueryAI")


# ============================================================
# PATHS
# ============================================================

POINT_SPLIT_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "point_grounding_split"
)

REFERENCE_DIR = (
    PROJECT
    / "kaggle"
    / "reference_grounding_end_to_end"
)

GROUNDING_DATA_DIR = (
    PROJECT
    / "data"
    / "processed"
)

REFERENCE_PACKAGE = (
    PROJECT
    / "kaggle"
    / "reference_grounding"
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def inspect_file(path):
    print()
    print(f"FILE: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        return

    print(f"SIZE: {path.stat().st_size:,} bytes")

    if path.suffix.lower() == ".parquet":

        try:
            df = pd.read_parquet(
                path,
                engine="pyarrow",
            )

            print(f"ROWS: {len(df):,}")
            print(f"COLUMNS ({len(df.columns)}):")

            for column in df.columns:
                print(
                    f"  {column}: "
                    f"{df[column].dtype}"
                )

            print()
            print("FIRST 3 ROWS:")

            with pd.option_context(
                "display.max_columns",
                None,
                "display.width",
                220,
            ):
                print(
                    df.head(3).to_string(
                        index=False
                    )
                )

        except Exception as exc:
            print(
                f"PARQUET READ FAILED: "
                f"{type(exc).__name__}: {exc}"
            )


def inspect_directory(path, recursive=False):
    print()
    print(f"DIRECTORY: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        return

    if recursive:
        items = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
        )
    else:
        items = sorted(
            p
            for p in path.iterdir()
        )

    print(f"ITEM COUNT: {len(items)}")

    for item in items[:200]:

        if item.is_file():
            print(
                f"  FILE   "
                f"{item.relative_to(path)}"
                f"   "
                f"{item.stat().st_size:,} bytes"
            )

        else:
            print(
                f"  DIR    "
                f"{item.relative_to(path)}"
            )

    if len(items) > 200:
        print(
            f"  ... {len(items) - 200} more items"
        )


# ============================================================
# START
# ============================================================

print_header(
    "SATQUERYAI — PHASE B REAL GROUNDING DATA INSPECTION"
)

print(f"PROJECT: {PROJECT}")
print("TRAINING: NOT STARTED")
print("KAGGLE: NOT USED")
print("GPU: OFF")
print("DEVICE: CPU")


# ============================================================
# 1. POINT VALIDATION DATA
# ============================================================

print_header("1. POINT VALIDATION DATA")

point_train = (
    POINT_SPLIT_DIR
    / "point_train.parquet"
)

point_val = (
    POINT_SPLIT_DIR
    / "point_val.parquet"
)

point_metadata = (
    POINT_SPLIT_DIR
    / "split_metadata.json"
)

inspect_file(point_train)
inspect_file(point_val)
inspect_file(point_metadata)


# ============================================================
# 2. REFERENCE VALIDATION DATA
# ============================================================

print_header("2. REFERENCE VALIDATION DATA")

reference_train = (
    REFERENCE_DIR
    / "reference_train.parquet"
)

reference_val = (
    REFERENCE_DIR
    / "reference_val.parquet"
)

inspect_file(reference_train)
inspect_file(reference_val)


# ============================================================
# 3. GROUNDING PROCESSED DATA
# ============================================================

print_header("3. PROCESSED GROUNDING DATA")

inspect_directory(
    GROUNDING_DATA_DIR,
    recursive=False,
)


# ============================================================
# 4. GROUNDING TEXT CACHE
# ============================================================

print_header("4. GROUNDING TEXT CACHE")

text_cache_candidates = [
    GROUNDING_DATA_DIR
    / "grounding_text_embeddings",

    GROUNDING_DATA_DIR,

    PROJECT
    / "kaggle_grounding_embeddings",

    REFERENCE_PACKAGE
    / "grounding_text_embeddings",
]

for path in text_cache_candidates:

    if path.exists():
        inspect_directory(
            path,
            recursive=True,
        )


# ============================================================
# 5. DOFA FEATURE CACHE
# ============================================================

print_header("5. DOFA FEATURE CACHE")

dofa_candidates = [
    GROUNDING_DATA_DIR
    / "dofa_features",

    PROJECT
    / "kaggle_grounding_embeddings",

    REFERENCE_PACKAGE
    / "dofa_features",
]

for path in dofa_candidates:

    if path.exists():
        inspect_directory(
            path,
            recursive=True,
        )


# ============================================================
# 6. REFERENCE PACKAGE
# ============================================================

print_header("6. LOCAL REFERENCE GROUNDING PACKAGE")

inspect_directory(
    REFERENCE_PACKAGE,
    recursive=False,
)


# ============================================================
# 7. SUMMARY / LIKELY REQUIRED FIELDS
# ============================================================

print_header("7. WHAT WE NEED TO IDENTIFY")

print(
    """
For POINT validation we need to identify:

  patch identifier
  annotation/source-row identifier
  text embedding or text-cache reference
  point coordinates
  target bounding box
  DOFA S1 feature location
  DOFA S2 feature location

For REFERENCE validation we need to identify:

  patch identifier
  annotation/source-row identifier
  reference phrase/text embedding
  target bounding box
  DOFA S1 feature location
  DOFA S2 feature location

Critical historical mapping:

  annotation_index
        ↕
  text-cache source_row_index

The previously validated mapping is direct
for the real Point data. Do not substitute
grounding_row_index for source_row_index.
"""
)


print_header("INSPECTION COMPLETE")

print(
    "No models were loaded."
)

print(
    "No checkpoints were loaded."
)

print(
    "No training was performed."
)

print(
    "No files were modified."
)

print("=" * 72)