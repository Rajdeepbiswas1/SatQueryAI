import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

TRAIN_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
    / "clc_vocabulary.json"
)

REFERENCE_MAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

TRAIN_OUTPUT = OUTPUT_DIR / "train_semantic_targets.pt"
VAL_OUTPUT = OUTPUT_DIR / "val_semantic_targets.pt"
MANIFEST_OUTPUT = OUTPUT_DIR / "semantic_target_manifest.json"


# ============================================================
# FIXED CONFIGURATION
# ============================================================

SOURCE_SIZE = 120
TARGET_SIZE = 14

VERIFY_TOLERANCE = 1e-8


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GENERATE C3 SEMANTIC TARGETS")
print("=" * 70)
print()

print(f"Source grid : {SOURCE_SIZE} x {SOURCE_SIZE}")
print(f"Target grid : {TARGET_SIZE} x {TARGET_SIZE}")
print(f"Vocabulary  : {VOCAB_PATH}")
print(f"Reference maps: {REFERENCE_MAP_ROOT}")
print()


# ============================================================
# LOAD VOCABULARY
# ============================================================

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"CLC vocabulary not found:\n{VOCAB_PATH}"
    )

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)


# Support the expected vocabulary structure while remaining
# deliberately strict about the actual class list.
if not isinstance(vocab_data, dict):
    raise RuntimeError(
        "CLC vocabulary JSON must contain a dictionary."
    )

if "clc_vocabulary" not in vocab_data:
    raise RuntimeError(
        "Frozen vocabulary JSON does not contain "
        "'clc_vocabulary'."
    )

clc_values = vocab_data["clc_vocabulary"]

if not isinstance(clc_values, list):
    raise RuntimeError(
        "'clc_vocabulary' must be a list."
    )


clc_values = sorted(
    {int(x) for x in clc_values}
)

if len(clc_values) != 32:
    raise RuntimeError(
        f"Expected 32 frozen CLC classes, "
        f"found {len(clc_values)}:\n{clc_values}"
    )

CLC_TO_INDEX = {
    clc: idx
    for idx, clc in enumerate(clc_values)
}

NUM_CLASSES = len(clc_values)

print("Frozen CLC vocabulary:")
print(clc_values)
print(f"Number of classes: {NUM_CLASSES}")
print()


# ============================================================
# LOAD SPLITS
# ============================================================

def load_patch_ids(parquet_path: Path):
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found:\n{parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    if "patch_id" not in df.columns:
        raise RuntimeError(
            f"'patch_id' column not found in:\n{parquet_path}\n"
            f"Columns: {list(df.columns)}"
        )

    patch_ids = sorted(
        {
            str(x)
            for x in df["patch_id"].dropna().tolist()
        }
    )

    return patch_ids


train_patch_ids = load_patch_ids(TRAIN_PARQUET)
val_patch_ids = load_patch_ids(VAL_PARQUET)

train_set = set(train_patch_ids)
val_set = set(val_patch_ids)

overlap = train_set & val_set

print("SPLIT CHECK")
print("-" * 70)
print(f"Train patches:      {len(train_patch_ids)}")
print(f"Validation patches: {len(val_patch_ids)}")
print(f"Train/val overlap:  {len(overlap)}")

if overlap:
    raise RuntimeError(
        "Train/validation patch overlap detected."
    )

print()


# ============================================================
# INDEX ALL REFERENCE MAPS
# ============================================================

print("INDEXING REFERENCE MAPS...")
print("-" * 70)

if not REFERENCE_MAP_ROOT.exists():
    raise FileNotFoundError(
        f"Reference-map directory not found:\n"
        f"{REFERENCE_MAP_ROOT}"
    )

map_files = sorted(
    REFERENCE_MAP_ROOT.rglob("*_reference_map.tif")
)

print(f"Reference maps found: {len(map_files)}")

map_index = {}

for path in map_files:

    filename = path.name

    suffix = "_reference_map.tif"

    if not filename.endswith(suffix):
        continue

    patch_id = filename[:-len(suffix)]

    if patch_id in map_index:
        raise RuntimeError(
            "Duplicate reference map for patch:\n"
            f"{patch_id}\n"
            f"Existing: {map_index[patch_id]}\n"
            f"Duplicate: {path}"
        )

    map_index[patch_id] = path

print(f"Reference maps indexed: {len(map_index)}")
print()


# ============================================================
# SPLIT MAP CHECK
# ============================================================

missing_train = [
    pid for pid in train_patch_ids
    if pid not in map_index
]

missing_val = [
    pid for pid in val_patch_ids
    if pid not in map_index
]

print("MAP AVAILABILITY CHECK")
print("-" * 70)
print(f"Missing train maps: {len(missing_train)}")
print(f"Missing val maps:   {len(missing_val)}")

if missing_train:
    print("First missing train patch:", missing_train[0])

if missing_val:
    print("First missing validation patch:", missing_val[0])

if missing_train or missing_val:
    raise RuntimeError(
        "Some split patches do not have reference maps."
    )

print()


# ============================================================
# EXACT OVERLAP WEIGHTS
# ============================================================

def build_overlap_weights(
    source_size: int,
    target_size: int,
):
    """
    Build exact geometric overlap weights.

    Result:
        weights[target_cell, source_pixel]

    Each target row sums to 1.0.

    The source and target grids are represented in the
    normalized coordinate interval [0, 1].
    """

    weights = np.zeros(
        (target_size, source_size),
        dtype=np.float64,
    )

    for target_idx in range(target_size):

        target_left = target_idx / target_size
        target_right = (
            (target_idx + 1) / target_size
        )

        target_width = (
            target_right - target_left
        )

        for source_idx in range(source_size):

            source_left = source_idx / source_size
            source_right = (
                (source_idx + 1) / source_size
            )

            overlap = max(
                0.0,
                min(
                    target_right,
                    source_right,
                )
                - max(
                    target_left,
                    source_left,
                ),
            )

            weights[
                target_idx,
                source_idx,
            ] = overlap / target_width

    row_sums = weights.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Overlap-weight rows do not sum to 1."
        )

    return weights


weights_y = build_overlap_weights(
    SOURCE_SIZE,
    TARGET_SIZE,
)

weights_x = build_overlap_weights(
    SOURCE_SIZE,
    TARGET_SIZE,
)

print("EXACT AGGREGATION CHECK")
print("-" * 70)
print(
    "Y row-sum min/max:",
    weights_y.sum(axis=1).min(),
    weights_y.sum(axis=1).max(),
)
print(
    "X row-sum min/max:",
    weights_x.sum(axis=1).min(),
    weights_x.sum(axis=1).max(),
)
print()


# Torch versions for the tensor calculation.
weights_y_t = torch.from_numpy(weights_y)
weights_x_t = torch.from_numpy(weights_x)


# ============================================================
# SINGLE MAP CONVERSION
# ============================================================

def map_to_semantic_target(
    patch_id: str,
    map_path: Path,
):
    """
    Convert one 120x120 integer CLC map to:

        [32, 14, 14]

    using exact geometric area overlap.
    """

    with rasterio.open(map_path) as src:

        if src.count < 1:
            raise RuntimeError(
                f"No raster band found:\n{map_path}"
            )

        clc_map = src.read(1)

    if clc_map.shape != (
        SOURCE_SIZE,
        SOURCE_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected map shape for {patch_id}: "
            f"{clc_map.shape}"
        )

    clc_map = clc_map.astype(
        np.int64,
        copy=False,
    )

    unique_values = np.unique(
        clc_map
    ).tolist()

    unseen = [
        int(x)
        for x in unique_values
        if int(x) not in CLC_TO_INDEX
    ]

    if unseen:
        raise RuntimeError(
            f"Unseen CLC values in patch {patch_id}: "
            f"{unseen}"
        )

    # --------------------------------------------------------
    # Source CLC counts.
    # --------------------------------------------------------

    source_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int32,
    )

    for clc_value, class_index in CLC_TO_INDEX.items():
        source_counts[class_index] = np.count_nonzero(
            clc_map == clc_value
        )

    if source_counts.sum() != (
        SOURCE_SIZE * SOURCE_SIZE
    ):
        raise RuntimeError(
            f"Source class counts do not cover all "
            f"pixels for {patch_id}."
        )

    # --------------------------------------------------------
    # One-hot source map:
    #
    # [32, 120, 120]
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Convert raw CLC codes to frozen class indices.
    #
    # Example:
    #   111 -> 0
    #   112 -> 1
    #   ...
    #   523 -> 31
    # --------------------------------------------------------

    source_tensor = torch.from_numpy(
        clc_map
    ).long()

    indexed_map = torch.empty_like(
        source_tensor
    )

    for clc_value, class_index in CLC_TO_INDEX.items():
        indexed_map[
            source_tensor == clc_value
        ] = class_index

    # Final safety check: every pixel must now contain
    # an index in [0, NUM_CLASSES - 1].
    if (
        indexed_map.min().item() < 0
        or indexed_map.max().item() >= NUM_CLASSES
    ):
        raise RuntimeError(
            f"Invalid class index range in {patch_id}: "
            f"min={indexed_map.min().item()}, "
            f"max={indexed_map.max().item()}"
        )

    one_hot = torch.nn.functional.one_hot(
        indexed_map,
        num_classes=NUM_CLASSES,
    )

    one_hot = one_hot.permute(
        2, 0, 1
    ).to(
        dtype=torch.float64
    )

    # --------------------------------------------------------
    # Exact area aggregation:
    #
    # [14,120] × [32,120,120] × [14,120]
    #
    # -> [32,14,14]
    # --------------------------------------------------------

    target = torch.einsum(
        "ai,cij,bj->cab",
        weights_y_t,
        one_hot,
        weights_x_t,
    )

    # --------------------------------------------------------
    # Verify each target spatial cell sums to 1.
    # --------------------------------------------------------

    cell_sums = target.sum(
        dim=0
    )

    max_cell_sum_error = float(
        torch.abs(
            cell_sums - 1.0
        ).max().item()
    )

    # --------------------------------------------------------
    # Recover source-pixel-equivalent class areas.
    #
    # Every target cell represents:
    #
    # (120 / 14)^2 source pixels.
    # --------------------------------------------------------

    equivalent_source_area_per_cell = (
        SOURCE_SIZE / TARGET_SIZE
    ) ** 2

    recovered_class_area = (
        target.sum(
            dim=(1, 2)
        )
        * equivalent_source_area_per_cell
    )

    source_counts_t = torch.from_numpy(
        source_counts
    ).to(
        dtype=torch.float64
    )

    max_area_error = float(
        torch.abs(
            recovered_class_area
            - source_counts_t
        ).max().item()
    )

    # --------------------------------------------------------
    # Verify target values stay in [0,1].
    # --------------------------------------------------------

    target_min = float(
        target.min().item()
    )

    target_max = float(
        target.max().item()
    )

    if target_min < -VERIFY_TOLERANCE:
        raise RuntimeError(
            f"Negative semantic fraction in {patch_id}: "
            f"{target_min}"
        )

    if target_max > (
        1.0 + VERIFY_TOLERANCE
    ):
        raise RuntimeError(
            f"Semantic fraction > 1 in {patch_id}: "
            f"{target_max}"
        )

    # --------------------------------------------------------
    # Final integrity checks.
    # --------------------------------------------------------

    if max_cell_sum_error > VERIFY_TOLERANCE:
        raise RuntimeError(
            f"Cell-sum conservation failed for "
            f"{patch_id}: {max_cell_sum_error}"
        )

    if max_area_error > VERIFY_TOLERANCE:
        raise RuntimeError(
            f"Class-area conservation failed for "
            f"{patch_id}: {max_area_error}"
        )

    return (
        target.to(torch.float32),
        source_counts,
        max_cell_sum_error,
        max_area_error,
    )


# ============================================================
# GENERATE ONE SPLIT
# ============================================================

def generate_split(
    split_name: str,
    patch_ids,
):
    print()
    print("=" * 70)
    print(f"GENERATING {split_name.upper()} SEMANTIC TARGETS")
    print("=" * 70)

    targets = []
    source_counts_all = []

    max_cell_error = 0.0
    max_area_error = 0.0

    total = len(patch_ids)

    for index, patch_id in enumerate(
        patch_ids,
        start=1,
    ):

        map_path = map_index[patch_id]

        (
            target,
            source_counts,
            cell_error,
            area_error,
        ) = map_to_semantic_target(
            patch_id,
            map_path,
        )

        targets.append(target)
        source_counts_all.append(
            torch.from_numpy(source_counts)
        )

        max_cell_error = max(
            max_cell_error,
            cell_error,
        )

        max_area_error = max(
            max_area_error,
            area_error,
        )

        if (
            index <= 3
            or index == total
            or index % 500 == 0
        ):
            print(
                f"{index}/{total} "
                f"| patch={patch_id} "
                f"| max_cell_err={cell_error:.3e} "
                f"| max_area_err={area_error:.3e}"
            )

    target_tensor = torch.stack(
        targets,
        dim=0,
    )

    source_counts_tensor = torch.stack(
        source_counts_all,
        dim=0,
    )

    print()
    print(
        f"{split_name.upper()} tensor shape: "
        f"{tuple(target_tensor.shape)}"
    )

    print(
        f"{split_name.upper()} dtype: "
        f"{target_tensor.dtype}"
    )

    print(
        f"{split_name.upper()} max cell-sum error: "
        f"{max_cell_error:.3e}"
    )

    print(
        f"{split_name.upper()} max area error: "
        f"{max_area_error:.3e}"
    )

    return (
        target_tensor,
        source_counts_tensor,
        max_cell_error,
        max_area_error,
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for output_path in [
    TRAIN_OUTPUT,
    VAL_OUTPUT,
    MANIFEST_OUTPUT,
]:
    if output_path.exists():
        raise RuntimeError(
            "Output already exists:\n"
            f"{output_path}\n\n"
            "This script intentionally refuses to overwrite "
            "existing C3 semantic targets."
        )


# ============================================================
# GENERATE TRAIN + VALIDATION
# ============================================================

(
    train_targets,
    train_source_counts,
    train_max_cell_error,
    train_max_area_error,
) = generate_split(
    "train",
    train_patch_ids,
)

(
    val_targets,
    val_source_counts,
    val_max_cell_error,
    val_max_area_error,
) = generate_split(
    "validation",
    val_patch_ids,
)


# ============================================================
# FINAL SHAPE CHECK
# ============================================================

expected_train_shape = (
    len(train_patch_ids),
    NUM_CLASSES,
    TARGET_SIZE,
    TARGET_SIZE,
)

expected_val_shape = (
    len(val_patch_ids),
    NUM_CLASSES,
    TARGET_SIZE,
    TARGET_SIZE,
)

if tuple(train_targets.shape) != expected_train_shape:
    raise RuntimeError(
        f"Unexpected train target shape: "
        f"{tuple(train_targets.shape)}"
    )

if tuple(val_targets.shape) != expected_val_shape:
    raise RuntimeError(
        f"Unexpected validation target shape: "
        f"{tuple(val_targets.shape)}"
    )


# ============================================================
# GLOBAL MASS CHECK
# ============================================================

train_cell_mass_error = float(
    torch.abs(
        train_targets.double().sum(dim=1)
        - 1.0
    ).max().item()
)

val_cell_mass_error = float(
    torch.abs(
        val_targets.double().sum(dim=1)
        - 1.0
    ).max().item()
)


# ============================================================
# SAVE PAYLOADS
# ============================================================

train_payload = {
    "patch_ids": train_patch_ids,
    "targets": train_targets,
    "source_pixel_counts": train_source_counts,
    "clc_values": clc_values,
    "source_size": SOURCE_SIZE,
    "target_size": TARGET_SIZE,
    "num_classes": NUM_CLASSES,
    "dtype": str(train_targets.dtype),
}

val_payload = {
    "patch_ids": val_patch_ids,
    "targets": val_targets,
    "source_pixel_counts": val_source_counts,
    "clc_values": clc_values,
    "source_size": SOURCE_SIZE,
    "target_size": TARGET_SIZE,
    "num_classes": NUM_CLASSES,
    "dtype": str(val_targets.dtype),
}


print()
print("=" * 70)
print("SAVING TARGETS")
print("=" * 70)

torch.save(
    train_payload,
    TRAIN_OUTPUT,
)

print(
    f"Saved train targets:\n{TRAIN_OUTPUT}"
)

torch.save(
    val_payload,
    VAL_OUTPUT,
)

print(
    f"Saved validation targets:\n{VAL_OUTPUT}"
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "source_size": SOURCE_SIZE,
    "target_size": TARGET_SIZE,
    "num_classes": NUM_CLASSES,
    "clc_values": clc_values,

    "train_patches": len(train_patch_ids),
    "validation_patches": len(val_patch_ids),

    "train_target_shape": list(
        train_targets.shape
    ),

    "validation_target_shape": list(
        val_targets.shape
    ),

    "train_max_cell_sum_error": train_max_cell_error,
    "train_max_area_error": train_max_area_error,

    "validation_max_cell_sum_error": val_max_cell_error,
    "validation_max_area_error": val_max_area_error,

    "train_global_cell_mass_error": train_cell_mass_error,
    "validation_global_cell_mass_error": val_cell_mass_error,

    "aggregation": (
        "exact geometric overlap from "
        "120x120 source pixels to 14x14 target cells"
    ),

    "saved_target_dtype": "torch.float32",
}

with open(
    MANIFEST_OUTPUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("C3 SEMANTIC TARGET GENERATION COMPLETE")
print("=" * 70)

print()
print("TRAIN")
print(f"  patches        : {len(train_patch_ids)}")
print(f"  tensor shape   : {tuple(train_targets.shape)}")
print(f"  max cell error : {train_max_cell_error:.3e}")
print(f"  max area error : {train_max_area_error:.3e}")

print()
print("VALIDATION")
print(f"  patches        : {len(val_patch_ids)}")
print(f"  tensor shape   : {tuple(val_targets.shape)}")
print(f"  max cell error : {val_max_cell_error:.3e}")
print(f"  max area error : {val_max_area_error:.3e}")

print()
print("OUTPUTS")
print(f"  {TRAIN_OUTPUT}")
print(f"  {VAL_OUTPUT}")
print(f"  {MANIFEST_OUTPUT}")

print()
print("No original images or reference maps were modified.")
print()