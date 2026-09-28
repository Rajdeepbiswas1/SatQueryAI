import json
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

TRAIN_PATH = TARGET_DIR / "train_semantic_targets.pt"
VAL_PATH = TARGET_DIR / "val_semantic_targets.pt"
VOCAB_PATH = TARGET_DIR / "clc_vocabulary.json"
MANIFEST_PATH = TARGET_DIR / "semantic_target_manifest.json"


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 32
GRID_SIZE = 14

CELL_SUM_TOL = 1e-6
PATCH_MASS_TOL = 1e-4
CLASS_FRACTION_TOL = 1e-6


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VERIFY C3 SEMANTIC TARGET FILES")
print("=" * 70)
print()


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    TRAIN_PATH,
    VAL_PATH,
    VOCAB_PATH,
    MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("Required files: PASS")
print()


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:
    vocab = json.load(f)

clc_values = [
    int(x)
    for x in vocab["clc_vocabulary"]
]

if len(clc_values) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected {NUM_CLASSES} CLC classes, "
        f"found {len(clc_values)}"
    )

print("VOCABULARY")
print("-" * 70)
print(f"Classes: {len(clc_values)}")
print(f"CLC values: {clc_values}")
print()


# ============================================================
# VERIFY ONE SPLIT
# ============================================================

def verify_split(
    name: str,
    path: Path,
    expected_patch_count: int,
):
    print("=" * 70)
    print(f"VERIFYING {name.upper()}")
    print("=" * 70)

    payload = torch.load(
        path,
        map_location="cpu",
    )

    required_keys = [
        "patch_ids",
        "targets",
        "source_pixel_counts",
        "clc_values",
        "source_size",
        "target_size",
        "num_classes",
    ]

    for key in required_keys:
        if key not in payload:
            raise RuntimeError(
                f"{name}: missing key '{key}'"
            )

    patch_ids = payload["patch_ids"]
    targets = payload["targets"]
    source_counts = payload["source_pixel_counts"]

    print(f"Patch count: {len(patch_ids)}")
    print(f"Target shape: {tuple(targets.shape)}")
    print(f"Target dtype: {targets.dtype}")
    print(
        f"Source-count shape: "
        f"{tuple(source_counts.shape)}"
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    expected_shape = (
        expected_patch_count,
        NUM_CLASSES,
        GRID_SIZE,
        GRID_SIZE,
    )

    if tuple(targets.shape) != expected_shape:
        raise RuntimeError(
            f"{name}: unexpected target shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {tuple(targets.shape)}"
        )

    if tuple(source_counts.shape) != (
        expected_patch_count,
        NUM_CLASSES,
    ):
        raise RuntimeError(
            f"{name}: unexpected source-count shape."
        )

    if len(patch_ids) != expected_patch_count:
        raise RuntimeError(
            f"{name}: patch count mismatch."
        )

    # --------------------------------------------------------
    # Metadata checks
    # --------------------------------------------------------

    if payload["source_size"] != 120:
        raise RuntimeError(
            f"{name}: source_size is not 120."
        )

    if payload["target_size"] != 14:
        raise RuntimeError(
            f"{name}: target_size is not 14."
        )

    if payload["num_classes"] != NUM_CLASSES:
        raise RuntimeError(
            f"{name}: num_classes is not 32."
        )

    stored_clc = [
        int(x)
        for x in payload["clc_values"]
    ]

    if stored_clc != clc_values:
        raise RuntimeError(
            f"{name}: stored CLC vocabulary does not "
            f"match frozen vocabulary."
        )

    # --------------------------------------------------------
    # Numeric checks
    # --------------------------------------------------------

    if not torch.is_floating_point(targets):
        raise RuntimeError(
            f"{name}: targets are not floating point."
        )

    if not torch.isfinite(targets).all():
        raise RuntimeError(
            f"{name}: targets contain NaN or Inf."
        )

    if not torch.isfinite(source_counts.float()).all():
        raise RuntimeError(
            f"{name}: source counts contain NaN or Inf."
        )

    target_min = float(
        targets.min().item()
    )

    target_max = float(
        targets.max().item()
    )

    print(
        f"Target value range: "
        f"[{target_min:.9f}, {target_max:.9f}]"
    )

    if target_min < -CLASS_FRACTION_TOL:
        raise RuntimeError(
            f"{name}: negative target value detected."
        )

    if target_max > 1.0 + CLASS_FRACTION_TOL:
        raise RuntimeError(
            f"{name}: target value above 1 detected."
        )

    # --------------------------------------------------------
    # Every spatial cell should sum to 1 across CLC classes.
    # --------------------------------------------------------

    cell_sums = targets.double().sum(
        dim=1
    )

    cell_sum_error = torch.abs(
        cell_sums - 1.0
    )

    max_cell_sum_error = float(
        cell_sum_error.max().item()
    )

    mean_cell_sum_error = float(
        cell_sum_error.mean().item()
    )

    print(
        f"Mean cell-sum error: "
        f"{mean_cell_sum_error:.9e}"
    )

    print(
        f"Max cell-sum error:  "
        f"{max_cell_sum_error:.9e}"
    )

    if max_cell_sum_error > CELL_SUM_TOL:
        raise RuntimeError(
            f"{name}: spatial cell sums failed."
        )

    # --------------------------------------------------------
    # Every patch contains 14*14 = 196 total cell mass.
    # --------------------------------------------------------

    patch_mass = targets.double().sum(
        dim=(1, 2, 3)
    )

    expected_patch_mass = float(
        GRID_SIZE * GRID_SIZE
    )

    patch_mass_error = torch.abs(
        patch_mass - expected_patch_mass
    )

    max_patch_mass_error = float(
        patch_mass_error.max().item()
    )

    print(
        f"Max patch-mass error: "
        f"{max_patch_mass_error:.9e}"
    )

    if max_patch_mass_error > PATCH_MASS_TOL:
        raise RuntimeError(
            f"{name}: patch total mass failed."
        )

    # --------------------------------------------------------
    # Class fractions:
    #
    # sum over the 14x14 cells gives the class's
    # source-pixel-equivalent area / (120/14)^2.
    # --------------------------------------------------------

    class_area_fraction = targets.double().sum(
        dim=(2, 3)
    )

    reconstructed_source_area = (
        class_area_fraction
        * ((120.0 / 14.0) ** 2)
    )

    source_counts_float = (
        source_counts.to(
            dtype=torch.float64
        )
    )

    area_error = torch.abs(
        reconstructed_source_area
        - source_counts_float
    )

    max_area_error = float(
        area_error.max().item()
    )

    mean_area_error = float(
        area_error.mean().item()
    )

    print(
        f"Mean reconstructed area error: "
        f"{mean_area_error:.9e}"
    )

    print(
        f"Max reconstructed area error:  "
        f"{max_area_error:.9e}"
    )

    # Float32 serialization introduces a small rounding error,
    # so use a realistic tolerance here.
    if max_area_error > 1e-3:
        raise RuntimeError(
            f"{name}: reconstructed source areas "
            f"deviate unexpectedly."
        )

    # --------------------------------------------------------
    # Patch IDs
    # --------------------------------------------------------

    if len(
        set(patch_ids)
    ) != len(patch_ids):
        raise RuntimeError(
            f"{name}: duplicate patch IDs detected."
        )

    print(
        f"Unique patch IDs: "
        f"{len(set(patch_ids))}"
    )

    # --------------------------------------------------------
    # Class presence sanity
    # --------------------------------------------------------

    nonzero_class_count = int(
        (
            targets.sum(
                dim=(0, 2, 3)
            )
            > 0
        ).sum().item()
    )

    print(
        f"CLC classes represented: "
        f"{nonzero_class_count}/{NUM_CLASSES}"
    )

    # A split does not need to contain every frozen class.
    # The important invariant is that every class present in
    # the split belongs to the frozen training vocabulary.
    #
    # The source-count tensor has exactly one column per frozen
    # vocabulary class, so its shape already enforces the class
    # indexing contract.

    if nonzero_class_count > NUM_CLASSES:
        raise RuntimeError(
            f"{name}: represented CLC class count exceeds "
            f"the frozen vocabulary."
        )

    print(
        f"Represented classes: "
        f"{nonzero_class_count}/{NUM_CLASSES}"
    )

    if name.lower() == "train":
        if nonzero_class_count != NUM_CLASSES:
            raise RuntimeError(
                "TRAIN must represent all frozen CLC classes, "
                f"but only {nonzero_class_count}/{NUM_CLASSES} "
                "are present."
            )

    elif name.lower() == "validation":
        print(
            "Validation is allowed to contain fewer classes "
            "than the frozen training vocabulary."
        )

    print()
    print(f"{name.upper()} STATUS: PASS")
    print()

    return {
        "patch_count": len(patch_ids),
        "shape": list(targets.shape),
        "dtype": str(targets.dtype),
        "target_min": target_min,
        "target_max": target_max,
        "max_cell_sum_error": max_cell_sum_error,
        "max_patch_mass_error": max_patch_mass_error,
        "mean_area_error": mean_area_error,
        "max_area_error": max_area_error,
        "nonzero_class_count": nonzero_class_count,
    }


# ============================================================
# VERIFY TRAIN + VALIDATION
# ============================================================

train_result = verify_split(
    "train",
    TRAIN_PATH,
    6081,
)

val_result = verify_split(
    "validation",
    VAL_PATH,
    1521,
)


# ============================================================
# PATCH-DISJOINT CHECK
# ============================================================

train_payload = torch.load(
    TRAIN_PATH,
    map_location="cpu",
)

val_payload = torch.load(
    VAL_PATH,
    map_location="cpu",
)

train_ids = set(
    train_payload["patch_ids"]
)

val_ids = set(
    val_payload["patch_ids"]
)

overlap = train_ids & val_ids

print("=" * 70)
print("TRAIN / VALIDATION PATCH CHECK")
print("=" * 70)

print(
    f"Train patches:      {len(train_ids)}"
)

print(
    f"Validation patches: {len(val_ids)}"
)

print(
    f"Overlap:             {len(overlap)}"
)

if overlap:
    raise RuntimeError(
        "Train/validation patch overlap detected."
    )

print("Patch disjointness: PASS")
print()


# ============================================================
# FINAL REPORT
# ============================================================

print("=" * 70)
print("C3 SEMANTIC TARGET VERIFICATION COMPLETE")
print("=" * 70)

print()
print("All saved semantic-target checks passed.")
print()
print("Training:")
for key, value in train_result.items():
    print(f"  {key}: {value}")

print()
print("Validation:")
for key, value in val_result.items():
    print(f"  {key}: {value}")

print()
print("Files verified:")
print(f"  {TRAIN_PATH}")
print(f"  {VAL_PATH}")
print(f"  {VOCAB_PATH}")
print(f"  {MANIFEST_PATH}")

print()
print("No source imagery or reference maps were modified.")
print()