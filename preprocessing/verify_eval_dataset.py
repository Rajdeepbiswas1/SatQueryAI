from pathlib import Path
import random

import pandas as pd
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

PT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_eval"
)

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "evaluation_manifest.csv"
)

SAMPLE_COUNT = 30
RANDOM_SEED = 42

EXPECTED_SHAPE = (14, 120, 120)
EXPECTED_DTYPE = torch.float32


print("=== SatQueryAI evaluation verification ===")
print()

# --------------------------------------------------
# 1. Load manifest
# --------------------------------------------------

manifest = pd.read_csv(
    MANIFEST,
    dtype={"patch_id": str},
)

print(
    f"Manifest entries: {len(manifest):,}"
)

expected_counts = {
    "validation": 1000,
    "test": 1000,
    "benchmark": 1082,
}

for split, expected in expected_counts.items():

    actual = (
        manifest["dataset_split"]
        == split
    ).sum()

    print(
        f"{split.capitalize():12s}: "
        f"{actual:,}"
    )

    if actual != expected:
        raise RuntimeError(
            f"Expected {expected:,} "
            f"{split} entries, found {actual:,}."
        )

# --------------------------------------------------
# 2. Verify PT count
# --------------------------------------------------

pt_files = list(
    PT_DIR.glob("*.pt")
)

print(
    f"\nPT files: {len(pt_files):,}"
)

if len(pt_files) != len(manifest):
    raise RuntimeError(
        "PT file count does not match manifest."
    )

# --------------------------------------------------
# 3. Verify filenames match manifest
# --------------------------------------------------

manifest_ids = set(
    manifest["patch_id"]
)

pt_ids = {
    path.stem
    for path in pt_files
}

missing = manifest_ids - pt_ids
extra = pt_ids - manifest_ids

print(
    f"Missing PT files: {len(missing):,}"
)

print(
    f"Extra PT files:   {len(extra):,}"
)

if missing:
    raise RuntimeError(
        f"Missing evaluation PT files. "
        f"Example: {next(iter(missing))}"
    )

if extra:
    raise RuntimeError(
        f"Unexpected evaluation PT files. "
        f"Example: {next(iter(extra))}"
    )

# --------------------------------------------------
# 4. Check duplicates in manifest
# --------------------------------------------------

duplicates = manifest[
    manifest["patch_id"].duplicated(
        keep=False
    )
]

if not duplicates.empty:
    raise RuntimeError(
        "Duplicate patch IDs found in evaluation manifest."
    )

# --------------------------------------------------
# 5. Check that each split is disjoint
# --------------------------------------------------

sets = {
    split: set(
        manifest.loc[
            manifest["dataset_split"] == split,
            "patch_id",
        ]
    )
    for split in expected_counts
}

for first, first_ids in sets.items():

    for second, second_ids in sets.items():

        if first >= second:
            continue

        overlap = first_ids & second_ids

        if overlap:
            raise RuntimeError(
                f"Overlap between {first} and {second}: "
                f"{len(overlap):,}"
            )

# --------------------------------------------------
# 6. Random tensor checks
# --------------------------------------------------

random.seed(RANDOM_SEED)

sample_ids = random.sample(
    sorted(manifest_ids),
    min(SAMPLE_COUNT, len(manifest_ids)),
)

print()
print(
    f"Checking {len(sample_ids)} random tensors..."
)

for i, patch_id in enumerate(
    sample_ids,
    start=1,
):

    path = PT_DIR / f"{patch_id}.pt"

    data = torch.load(
        path,
        map_location="cpu",
    )

    required = {
        "patch_id",
        "dataset_split",
        "channel_names",
        "image",
    }

    missing_fields = (
        required - set(data.keys())
    )

    if missing_fields:
        raise RuntimeError(
            f"{patch_id}: missing fields "
            f"{missing_fields}"
        )

    if data["patch_id"] != patch_id:
        raise RuntimeError(
            f"{patch_id}: patch_id mismatch."
        )

    manifest_split = manifest.loc[
        manifest["patch_id"] == patch_id,
        "dataset_split",
    ].iloc[0]

    if data["dataset_split"] != manifest_split:
        raise RuntimeError(
            f"{patch_id}: split mismatch."
        )

    tensor = data["image"]

    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{patch_id}: wrong shape "
            f"{tuple(tensor.shape)}"
        )

    if tensor.dtype != EXPECTED_DTYPE:
        raise RuntimeError(
            f"{patch_id}: wrong dtype "
            f"{tensor.dtype}"
        )

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"{patch_id}: non-finite values."
        )

    min_value = tensor.min().item()
    max_value = tensor.max().item()

    if min_value < 0.0 or max_value > 1.0:
        raise RuntimeError(
            f"{patch_id}: values outside [0,1]."
        )

    if len(data["channel_names"]) != 14:
        raise RuntimeError(
            f"{patch_id}: incorrect channel count."
        )

    print(
        f"[{i:02d}/{len(sample_ids)}] "
        f"{manifest_split:10s} | "
        f"{patch_id} | "
        f"shape={tuple(tensor.shape)} | "
        f"range=[{min_value:.4f}, {max_value:.4f}]"
    )

# --------------------------------------------------
# 7. Success
# --------------------------------------------------

print()
print("=== VERIFICATION SUCCESSFUL ===")
print()
print("3,082 evaluation tensors verified.")
print("Validation: 1,000")
print("Test:       1,000")
print("Benchmark:  1,082")
print("All filenames match the manifest.")
print("All sampled tensors have the correct shape.")
print("All sampled tensors are float32.")
print("All sampled tensors contain finite values.")
print("All sampled tensors are within [0,1].")
print("Evaluation splits are mutually disjoint.")