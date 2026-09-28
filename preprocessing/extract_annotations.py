from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PARQUET_FILE = Path(
    r"D:\SatQueryAI\data\raw\BigEarthNet.txt.parquet"
)

SELECTED_PAIRS_FILE = Path(
    r"D:\SatQueryAI\data\splits\selected_pairs.csv"
)

OUTPUT_FILE = Path(
    r"D:\SatQueryAI\data\processed\training_annotations.parquet"
)


# ---------------------------------------------------------
# Load selected image-pair IDs
# ---------------------------------------------------------

print("Loading selected image pairs...")

selected_pairs = pd.read_csv(
    SELECTED_PAIRS_FILE
)

selected_ids = set(
    selected_pairs["patch_id"].astype(str)
)

print(f"Selected pairs: {len(selected_ids):,}")


# ---------------------------------------------------------
# Read the annotation dataset
# ---------------------------------------------------------

print("Reading BigEarthNet annotations...")

table = pq.read_table(
    PARQUET_FILE,
    columns=[
        "ID",
        "s1_name",
        "patch_id",
        "input",
        "output",
        "type",
        "category",
        "split",
        "latitude",
        "longitude",
        "country",
        "season",
        "climate_zone",
    ],
)

df = table.to_pandas()

print(f"Total annotations: {len(df):,}")


# ---------------------------------------------------------
# Keep only our selected pairs
# ---------------------------------------------------------

filtered = df[
    df["patch_id"].astype(str).isin(selected_ids)
].copy()

print(
    f"Selected annotations: {len(filtered):,}"
)


# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

unique_pairs = filtered["patch_id"].nunique()

print(
    f"Unique pairs represented: {unique_pairs:,}"
)

if unique_pairs != len(selected_ids):
    missing = selected_ids - set(
        filtered["patch_id"].astype(str)
    )

    raise RuntimeError(
        f"{len(missing)} selected pairs were not found "
        f"in the annotation dataset."
    )


# ---------------------------------------------------------
# Save filtered annotations
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

filtered.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print()
print("Extraction complete.")
print(f"Saved to: {OUTPUT_FILE}")


# ---------------------------------------------------------
# Show annotation distribution
# ---------------------------------------------------------

print("\nAnnotation types:")

print(
    filtered["type"].value_counts()
)

print("\nAnnotation categories:")

print(
    filtered["category"].value_counts()
)