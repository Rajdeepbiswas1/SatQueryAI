import os
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ANNOTATIONS = (
    r"D:\SatQueryAI\data\processed"
    r"\training_annotations.parquet"
)

OUTPUT_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\reference_grounding_split"
)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "reference_train.parquet",
)

VAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "reference_val.parquet",
)

MANIFEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "reference_patch_manifest.parquet",
)

SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PATCH-DISJOINT REFERENCE GROUNDING SPLIT")
    print("=" * 70)

    # ========================================================
    # LOAD ANNOTATIONS
    # ========================================================

    print("\nLoading annotations...")

    df = pd.read_parquet(
        ANNOTATIONS
    )

    print(
        f"Total annotation rows: "
        f"{len(df):,}"
    )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "patch_id",
        "type",
        "category",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # ========================================================
    # FILTER REFERENCE GROUNDING
    # ========================================================

    reference_df = df[
        (
            df["type"].astype(str).str.lower()
            == "bounding box"
        )
        &
        (
            df["category"].astype(str).str.lower()
            == "reference"
        )
    ].copy()

    reference_df = reference_df.copy()

    # Preserve the original annotation-row index.
    reference_df["original_row_index"] = reference_df.index

    reference_df.reset_index(
    	drop=True,
    	inplace=True,
    )

    print(
        f"Reference grounding rows: "
        f"{len(reference_df):,}"
    )

    if len(reference_df) == 0:

        raise RuntimeError(
            "No reference grounding examples found."
        )

    # ========================================================
    # PATCH STATISTICS
    # ========================================================

    reference_df["patch_id"] = (
        reference_df["patch_id"]
        .astype(str)
    )

    patch_ids = (
        reference_df["patch_id"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique reference patches: "
        f"{len(patch_ids):,}"
    )

    # ========================================================
    # PATCH-LEVEL RANDOM SPLIT
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    patch_ids = np.array(
        patch_ids,
        dtype=object,
    )

    rng.shuffle(
        patch_ids
    )

    train_patch_count = int(
        len(patch_ids)
        * TRAIN_RATIO
    )

    # Guarantee both sides contain patches
    train_patch_count = max(
        1,
        min(
            train_patch_count,
            len(patch_ids) - 1,
        ),
    )

    train_patches = set(
        patch_ids[
            :train_patch_count
        ]
    )

    val_patches = set(
        patch_ids[
            train_patch_count:
        ]
    )

    # ========================================================
    # ASSIGN ROWS
    # ========================================================

    reference_df["split"] = (
        reference_df["patch_id"]
        .apply(
            lambda x:
                "train"
                if x in train_patches
                else "val"
        )
    )

    # ========================================================
    # BUILD SPLITS
    # ========================================================

    train_df = reference_df[
        reference_df["split"]
        == "train"
    ].copy()

    val_df = reference_df[
        reference_df["split"]
        == "val"
    ].copy()

    train_df.reset_index(
        drop=True,
        inplace=True,
    )

    val_df.reset_index(
        drop=True,
        inplace=True,
    )

    # ========================================================
    # VALIDATE PATCH ISOLATION
    # ========================================================

    train_patch_check = set(
        train_df["patch_id"]
    )

    val_patch_check = set(
        val_df["patch_id"]
    )

    overlap = (
        train_patch_check
        &
        val_patch_check
    )

    print("\n" + "=" * 70)
    print("PATCH ISOLATION CHECK")
    print("=" * 70)

    print(
        f"Training patches: "
        f"{len(train_patch_check):,}"
    )

    print(
        f"Validation patches: "
        f"{len(val_patch_check):,}"
    )

    print(
        f"Overlapping patches: "
        f"{len(overlap):,}"
    )

    if overlap:

        print(
            "\nERROR: Patch overlap detected!"
        )

        for patch_id in list(
            sorted(overlap)
        )[:20]:

            print(
                f"  {patch_id}"
            )

        raise RuntimeError(
            "Patch-disjoint split validation failed."
        )

    print(
        "\nPASS: Training and validation "
        "patches are completely disjoint."
    )

    # ========================================================
    # BASIC DATASET STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    print(
        f"\nReference rows: "
        f"{len(reference_df):,}"
    )

    print(
        f"Training rows:  "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows:"
        f" {len(val_df):,}"
    )

    print(
        f"\nTraining row ratio: "
        f"{len(train_df) / len(reference_df):.4f}"
    )

    print(
        f"Validation row ratio: "
        f"{len(val_df) / len(reference_df):.4f}"
    )

    # ========================================================
    # PHRASE COVERAGE
    # ========================================================

    train_phrases = set(
        train_df["output"].astype(str)
    )

    val_phrases = set(
        val_df["output"].astype(str)
    )

    print("\n" + "=" * 70)
    print("REFERENCE QUERY COVERAGE")
    print("=" * 70)

    print(
        f"Unique training outputs: "
        f"{len(train_phrases):,}"
    )

    print(
        f"Unique validation outputs: "
        f"{len(val_phrases):,}"
    )

    shared_phrases = (
        train_phrases
        &
        val_phrases
    )

    validation_only = (
        val_phrases
        -
        train_phrases
    )

    print(
        f"Outputs appearing in both: "
        f"{len(shared_phrases):,}"
    )

    print(
        f"Validation-only outputs: "
        f"{len(validation_only):,}"
    )

    # ========================================================
    # ROWS PER PATCH
    # ========================================================

    train_rows_per_patch = (
        train_df
        .groupby("patch_id")
        .size()
    )

    val_rows_per_patch = (
        val_df
        .groupby("patch_id")
        .size()
    )

    print("\n" + "=" * 70)
    print("ROWS PER PATCH")
    print("=" * 70)

    print(
        f"Training mean rows/patch: "
        f"{train_rows_per_patch.mean():.2f}"
    )

    print(
        f"Training median rows/patch: "
        f"{train_rows_per_patch.median():.2f}"
    )

    print(
        f"Validation mean rows/patch: "
        f"{val_rows_per_patch.mean():.2f}"
    )

    print(
        f"Validation median rows/patch: "
        f"{val_rows_per_patch.median():.2f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    train_df.to_parquet(
        TRAIN_OUTPUT,
        index=False,
    )

    val_df.to_parquet(
        VAL_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Patch manifest
    # --------------------------------------------------------

    manifest_rows = []

    for patch_id in sorted(
        train_patch_check
    ):

        manifest_rows.append(
            {
                "patch_id": patch_id,
                "split": "train",
            }
        )

    for patch_id in sorted(
        val_patch_check
    ):

        manifest_rows.append(
            {
                "patch_id": patch_id,
                "split": "val",
            }
        )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_parquet(
        MANIFEST_OUTPUT,
        index=False,
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    train_saved = pd.read_parquet(
        TRAIN_OUTPUT
    )

    val_saved = pd.read_parquet(
        VAL_OUTPUT
    )

    saved_train_patches = set(
        train_saved["patch_id"]
    )

    saved_val_patches = set(
        val_saved["patch_id"]
    )

    saved_overlap = (
        saved_train_patches
        &
        saved_val_patches
    )

    if saved_overlap:

        raise RuntimeError(
            "Saved files contain patch overlap."
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\nTraining:"
        f"\n  {TRAIN_OUTPUT}"
    )

    print(
        f"\nValidation:"
        f"\n  {VAL_OUTPUT}"
    )

    print(
        f"\nPatch manifest:"
        f"\n  {MANIFEST_OUTPUT}"
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "\nPATCH-DISJOINT REFERENCE SPLIT "
        "CREATED SUCCESSFULLY."
    )

    print(
        "\nNo patch appears in both "
        "training and validation."
    )

    print(
        "\nThe model architecture has NOT "
        "been changed."
    )

    print(
        "\nTEST COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()