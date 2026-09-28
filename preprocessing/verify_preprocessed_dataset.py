from pathlib import Path
import random

import pandas as pd
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

PT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

SELECTED_PAIRS = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "selected_pairs.csv"
)

ANNOTATIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_annotations.parquet"
)

EXPECTED_SHAPE = (14, 120, 120)
EXPECTED_DTYPE = torch.float32

SAMPLE_COUNT = 20
RANDOM_SEED = 42


print("=== SatQueryAI preprocessing verification ===")
print()

# --------------------------------------------------
# 1. Verify PT file count
# --------------------------------------------------

pt_files = list(PT_DIR.glob("*.pt"))

print(f"PT files found: {len(pt_files):,}")

if len(pt_files) != 10_000:
    raise RuntimeError(
        f"Expected 10,000 PT files, found {len(pt_files):,}"
    )

# --------------------------------------------------
# 2. Load selected pairs
# --------------------------------------------------

selected = pd.read_csv(SELECTED_PAIRS)

print(f"Selected pairs: {len(selected):,}")

if len(selected) != 10_000:
    raise RuntimeError(
        f"Expected 10,000 selected pairs, found {len(selected):,}"
    )

selected_ids = set(selected["patch_id"])

# --------------------------------------------------
# 3. Verify PT filenames match selected patches
# --------------------------------------------------

pt_ids = {
    path.stem
    for path in pt_files
}

missing_pt = selected_ids - pt_ids
extra_pt = pt_ids - selected_ids

print(f"Missing PT files: {len(missing_pt):,}")
print(f"Extra PT files:   {len(extra_pt):,}")

if missing_pt:
    print("Example missing:", next(iter(missing_pt)))
    raise RuntimeError("Some selected patches have no PT file.")

if extra_pt:
    print("Example extra:", next(iter(extra_pt)))
    raise RuntimeError("PT directory contains unexpected patch IDs.")

# --------------------------------------------------
# 4. Random tensor verification
# --------------------------------------------------

random.seed(RANDOM_SEED)

sample_ids = random.sample(
    sorted(selected_ids),
    SAMPLE_COUNT,
)

print()
print(f"Checking {SAMPLE_COUNT} random tensors...")
print()

for i, patch_id in enumerate(sample_ids, start=1):

    path = PT_DIR / f"{patch_id}.pt"

    data = torch.load(
        path,
        map_location="cpu",
    )

    # Check required fields.
    required = {
        "patch_id",
        "channel_names",
        "image",
    }

    missing_fields = required - set(data.keys())

    if missing_fields:
        raise RuntimeError(
            f"{patch_id}: missing fields {missing_fields}"
        )

    # Check patch ID.
    if data["patch_id"] != patch_id:
        raise RuntimeError(
            f"{patch_id}: stored patch_id mismatch."
        )

    tensor = data["image"]

    # Shape.
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{patch_id}: wrong shape {tuple(tensor.shape)}"
        )

    # Dtype.
    if tensor.dtype != EXPECTED_DTYPE:
        raise RuntimeError(
            f"{patch_id}: wrong dtype {tensor.dtype}"
        )

    # Finite values.
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{patch_id}: non-finite values found."
        )

    # Range.
    min_value = tensor.min().item()
    max_value = tensor.max().item()

    if min_value < 0.0 or max_value > 1.0:
        raise RuntimeError(
            f"{patch_id}: values outside [0,1]: "
            f"{min_value} to {max_value}"
        )

    # Channel count.
    if len(data["channel_names"]) != 14:
        raise RuntimeError(
            f"{patch_id}: expected 14 channel names."
        )

    print(
        f"[{i:02d}/{SAMPLE_COUNT}] "
        f"{patch_id} | "
        f"shape={tuple(tensor.shape)} | "
        f"range=[{min_value:.4f}, {max_value:.4f}]"
    )

# --------------------------------------------------
# 5. Verify annotations
# --------------------------------------------------

print()
print("Checking annotation alignment...")

annotations = pd.read_parquet(
    ANNOTATIONS,
    columns=["patch_id"],
)

annotation_ids = set(annotations["patch_id"])

missing_annotations = selected_ids - annotation_ids

print(
    f"Selected patches without annotations: "
    f"{len(missing_annotations):,}"
)

if missing_annotations:
    print(
        "Example missing annotation:",
        next(iter(missing_annotations)),
    )
    raise RuntimeError(
        "Some selected patches have no annotations."
    )

# --------------------------------------------------
# Success
# --------------------------------------------------

print()
print("=== VERIFICATION SUCCESSFUL ===")
print()
print("10,000 PT files verified.")
print("All PT filenames match selected patches.")
print(f"{SAMPLE_COUNT} random tensors passed validation.")
print("All selected patches have annotations.")
print("Tensor shape: (14, 120, 120)")
print("Tensor dtype: torch.float32")
print("Tensor range: [0, 1]")