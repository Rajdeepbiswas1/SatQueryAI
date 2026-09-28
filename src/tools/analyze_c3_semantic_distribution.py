import json
from pathlib import Path

import pandas as pd
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

CSV_OUTPUT = TARGET_DIR / "semantic_class_distribution.csv"
JSON_OUTPUT = TARGET_DIR / "semantic_class_distribution.json"


# ============================================================
# CONFIG
# ============================================================

SOURCE_SIZE = 120
TARGET_SIZE = 14

# A class-fraction value of 1/(120*120) corresponds to
# approximately one source pixel.
ONE_SOURCE_PIXEL_FRACTION = (
    1.0 / (SOURCE_SIZE * SOURCE_SIZE)
)

# These thresholds are useful for distinguishing
# numerical nonzero values from meaningful source-map presence.
FRACTION_THRESHOLDS = [
    0.0,
    1.0 / (SOURCE_SIZE * SOURCE_SIZE),
    10.0 / (SOURCE_SIZE * SOURCE_SIZE),
    100.0 / (SOURCE_SIZE * SOURCE_SIZE),
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("C3 SEMANTIC CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)
print()

print(f"Train targets: {TRAIN_PATH}")
print(f"Validation targets: {VAL_PATH}")
print(f"Vocabulary: {VOCAB_PATH}")
print()


# ============================================================
# FILE CHECK
# ============================================================

for path in [
    TRAIN_PATH,
    VAL_PATH,
    VOCAB_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


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

num_classes = len(clc_values)

if num_classes != 32:
    raise RuntimeError(
        f"Expected 32 classes, found {num_classes}."
    )

print("Frozen CLC vocabulary:")
print(clc_values)
print()


# ============================================================
# LOAD TARGET FILE
# ============================================================

def load_target_file(path: Path):
    payload = torch.load(
        path,
        map_location="cpu",
    )

    required_keys = [
        "patch_ids",
        "targets",
        "source_pixel_counts",
        "clc_values",
    ]

    for key in required_keys:
        if key not in payload:
            raise RuntimeError(
                f"{path}: missing key '{key}'"
            )

    targets = payload["targets"].float()
    source_counts = payload["source_pixel_counts"].float()
    patch_ids = payload["patch_ids"]

    if targets.ndim != 4:
        raise RuntimeError(
            f"{path}: unexpected target dimensions "
            f"{targets.shape}"
        )

    if targets.shape[1:] != (
        num_classes,
        TARGET_SIZE,
        TARGET_SIZE,
    ):
        raise RuntimeError(
            f"{path}: unexpected target shape "
            f"{tuple(targets.shape)}"
        )

    if source_counts.shape != (
        len(patch_ids),
        num_classes,
    ):
        raise RuntimeError(
            f"{path}: unexpected source-count shape "
            f"{tuple(source_counts.shape)}"
        )

    return (
        patch_ids,
        targets,
        source_counts,
    )


train_ids, train_targets, train_source_counts = (
    load_target_file(TRAIN_PATH)
)

val_ids, val_targets, val_source_counts = (
    load_target_file(VAL_PATH)
)

print("Loaded target files successfully.")
print(
    f"Train tensor: {tuple(train_targets.shape)}"
)
print(
    f"Validation tensor: {tuple(val_targets.shape)}"
)
print()


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_split(
    split_name,
    patch_ids,
    targets,
    source_counts,
):
    print()
    print("=" * 70)
    print(f"{split_name.upper()} CLASS DISTRIBUTION")
    print("=" * 70)

    n = len(patch_ids)

    # --------------------------------------------------------
    # Total target mass per class.
    #
    # Each 14x14 cell contributes its CLC fraction.
    # There are 196 total spatial cells per patch.
    # --------------------------------------------------------

    total_target_mass = targets.sum(
        dim=(0, 2, 3)
    )

    mean_target_mass = (
        total_target_mass / n
    )

    # --------------------------------------------------------
    # Convert to source-pixel-equivalent area.
    # --------------------------------------------------------

    source_pixels_per_target_cell = (
        SOURCE_SIZE / TARGET_SIZE
    ) ** 2

    total_source_equivalent_area = (
        total_target_mass
        * source_pixels_per_target_cell
    )

    mean_source_equivalent_area = (
        total_source_equivalent_area / n
    )

    # Mean fraction of a 120x120 patch.
    mean_patch_fraction = (
        mean_source_equivalent_area
        / (SOURCE_SIZE * SOURCE_SIZE)
    )

    # --------------------------------------------------------
    # Direct source counts from stored verification data.
    # --------------------------------------------------------

    direct_source_total = source_counts.sum(
        dim=0
    )

    direct_source_mean = (
        direct_source_total / n
    )

    direct_mean_fraction = (
        direct_source_mean
        / (SOURCE_SIZE * SOURCE_SIZE)
    )

    # --------------------------------------------------------
    # Patch presence.
    #
    # Presence based on direct integer source counts is
    # the most interpretable criterion.
    # --------------------------------------------------------

    presence_pixels = []

    for threshold in [
        0,
        1,
        10,
        100,
    ]:
        count = (
            (source_counts > threshold)
            .sum(dim=0)
        )

        presence_pixels.append(count)

    presence_0 = presence_pixels[0]
    presence_1 = presence_pixels[1]
    presence_10 = presence_pixels[2]
    presence_100 = presence_pixels[3]

    # --------------------------------------------------------
    # Target-value statistics.
    # --------------------------------------------------------

    max_target_value = targets.amax(
        dim=(0, 2, 3)
    )

    mean_nonzero_value = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    for class_idx in range(num_classes):
        values = targets[
            :,
            class_idx,
            :,
            :
        ]

        nonzero = values[
            values > 0
        ]

        if nonzero.numel() > 0:
            mean_nonzero_value[
                class_idx
            ] = nonzero.double().mean()

    # --------------------------------------------------------
    # Per-class entropy contribution is not used yet.
    # We only record distribution statistics here.
    # --------------------------------------------------------

    rows = []

    for i, clc in enumerate(clc_values):

        rows.append(
            {
                "split": split_name,
                "class_index": i,
                "clc": clc,

                "total_source_pixels":
                    float(
                        direct_source_total[i].item()
                    ),

                "mean_source_pixels_per_patch":
                    float(
                        direct_source_mean[i].item()
                    ),

                "mean_patch_fraction":
                    float(
                        direct_mean_fraction[i].item()
                    ),

                "total_target_mass":
                    float(
                        total_target_mass[i].item()
                    ),

                "mean_target_mass_per_patch":
                    float(
                        mean_target_mass[i].item()
                    ),

                "patches_with_at_least_1_pixel":
                    int(
                        presence_1[i].item()
                    ),

                "patch_presence_pct_at_least_1_pixel":
                    float(
                        presence_1[i].item()
                        / n
                        * 100.0
                    ),

                "patches_with_at_least_10_pixels":
                    int(
                        presence_10[i].item()
                    ),

                "patch_presence_pct_at_least_10_pixels":
                    float(
                        presence_10[i].item()
                        / n
                        * 100.0
                    ),

                "patches_with_at_least_100_pixels":
                    int(
                        presence_100[i].item()
                    ),

                "patch_presence_pct_at_least_100_pixels":
                    float(
                        presence_100[i].item()
                        / n
                        * 100.0
                    ),

                "max_target_value":
                    float(
                        max_target_value[i].item()
                    ),

                "mean_nonzero_target_value":
                    float(
                        mean_nonzero_value[i].item()
                    ),
            }
        )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print(
        f"{'IDX':>3} "
        f"{'CLC':>4} "
        f"{'MEAN PX':>12} "
        f"{'MEAN %':>10} "
        f"{'PRES. %':>10} "
        f"{'PRES10 %':>10} "
        f"{'PRES100 %':>10}"
    )

    print("-" * 70)

    ordered = df.sort_values(
        "mean_patch_fraction",
        ascending=False,
    )

    for _, row in ordered.iterrows():

        print(
            f"{int(row['class_index']):>3} "
            f"{int(row['clc']):>4} "
            f"{row['mean_source_pixels_per_patch']:>12.2f} "
            f"{row['mean_patch_fraction'] * 100:>9.3f}% "
            f"{row['patch_presence_pct_at_least_1_pixel']:>9.2f}% "
            f"{row['patch_presence_pct_at_least_10_pixels']:>9.2f}% "
            f"{row['patch_presence_pct_at_least_100_pixels']:>9.2f}%"
        )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    nonzero_classes = int(
        (
            direct_source_total > 0
        ).sum().item()
    )

    absent_classes = [
        int(clc_values[i])
        for i in range(num_classes)
        if direct_source_total[i].item() == 0
    ]

    sorted_fraction = torch.sort(
        direct_mean_fraction
    ).values

    smallest_fraction = float(
        sorted_fraction[0].item()
    )

    largest_fraction = float(
        sorted_fraction[-1].item()
    )

    median_fraction = float(
        torch.median(
            direct_mean_fraction
        ).item()
    )

    ratio = None

    positive = direct_mean_fraction[
        direct_mean_fraction > 0
    ]

    if positive.numel() > 0:
        ratio = float(
            positive.max().item()
            / positive.min().item()
        )

    summary = {
        "split": split_name,
        "patch_count": n,
        "represented_classes": nonzero_classes,
        "total_classes": num_classes,
        "absent_classes": absent_classes,
        "smallest_positive_mean_fraction":
            smallest_fraction,
        "median_mean_fraction":
            median_fraction,
        "largest_mean_fraction":
            largest_fraction,
        "largest_to_smallest_positive_ratio":
            ratio,
    }

    print()
    print("SUMMARY")
    print("-" * 70)
    print(
        f"Represented classes: "
        f"{nonzero_classes}/{num_classes}"
    )

    print(
        f"Smallest positive mean fraction: "
        f"{smallest_fraction:.8f}"
    )

    print(
        f"Median mean fraction: "
        f"{median_fraction:.8f}"
    )

    print(
        f"Largest mean fraction: "
        f"{largest_fraction:.8f}"
    )

    if ratio is not None:
        print(
            f"Largest/smallest positive ratio: "
            f"{ratio:.2f}"
        )

    if absent_classes:
        print(
            f"Absent classes: {absent_classes}"
        )
    else:
        print(
            "Absent classes: none"
        )

    return df, summary


# ============================================================
# RUN ANALYSIS
# ============================================================

train_df, train_summary = analyze_split(
    "train",
    train_ids,
    train_targets,
    train_source_counts,
)

val_df, val_summary = analyze_split(
    "validation",
    val_ids,
    val_targets,
    val_source_counts,
)


# ============================================================
# COMBINE + SAVE
# ============================================================

combined_df = pd.concat(
    [
        train_df,
        val_df,
    ],
    ignore_index=True,
)

combined_df.to_csv(
    CSV_OUTPUT,
    index=False,
)

summary = {
    "config": {
        "source_size": SOURCE_SIZE,
        "target_size": TARGET_SIZE,
        "num_classes": num_classes,
        "clc_values": clc_values,
    },
    "train": train_summary,
    "validation": val_summary,
}

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("C3 SEMANTIC DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)

print()
print("CSV:")
print(CSV_OUTPUT)

print()
print("JSON:")
print(JSON_OUTPUT)

print()
print("No target tensors or reference maps were modified.")
print()