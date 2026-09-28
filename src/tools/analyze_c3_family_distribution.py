from pathlib import Path
import csv
import json

import torch


# ============================================================
# C3 19-CLASS SEMANTIC FAMILY DISTRIBUTION ANALYSIS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

TRAIN_PATH = (
    TARGET_DIR
    / "train_family_targets.pt"
)

VAL_PATH = (
    TARGET_DIR
    / "val_family_targets.pt"
)

OUTPUT_CSV = (
    TARGET_DIR
    / "family_distribution_analysis.csv"
)

OUTPUT_JSON = (
    TARGET_DIR
    / "family_distribution_analysis.json"
)


# ============================================================
# LOADING
# ============================================================

def load_targets(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected file format: {path}"
        )

    if "targets" not in data:
        raise RuntimeError(
            f"'targets' missing from: {path}"
        )

    targets = data["targets"].float()

    family_names = data["family_names"]

    if targets.ndim != 4:
        raise RuntimeError(
            f"Expected [N,C,H,W], got "
            f"{tuple(targets.shape)}"
        )

    if targets.shape[1] != 19:
        raise RuntimeError(
            f"Expected 19 families, got "
            f"{targets.shape[1]}"
        )

    if len(family_names) != 19:
        raise RuntimeError(
            "Expected exactly 19 family names."
        )

    return targets, family_names


# ============================================================
# ANALYSIS
# ============================================================

def analyze_split(
    targets,
    family_names,
    split_name,
):

    n = targets.shape[0]

    # Total spatial mass per family.
    total_area = (
        targets
        .double()
        .sum(
            dim=(0, 2, 3)
        )
    )

    # Mean area per patch.
    mean_area = (
        total_area
        / n
    )

    # Fraction of all semantic mass.
    grand_total = float(
        total_area.sum()
    )

    area_fraction = (
        total_area
        / grand_total
    )

    # Patch presence.
    patch_area = (
        targets
        .double()
        .sum(
            dim=(2, 3)
        )
    )

    presence = (
        patch_area > 1e-10
    ).float().mean(
        dim=0
    )

    rows = []

    for i, family in enumerate(
        family_names
    ):

        rows.append(
            {
                "split":
                    split_name,

                "family_index":
                    i,

                "semantic_family":
                    family,

                "total_area":
                    float(
                        total_area[i]
                    ),

                "mean_area_per_patch":
                    float(
                        mean_area[i]
                    ),

                "area_fraction":
                    float(
                        area_fraction[i]
                    ),

                "patch_presence_fraction":
                    float(
                        presence[i]
                    ),
            }
        )

    return rows


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize(rows):

    fractions = [
        row["area_fraction"]
        for row in rows
        if row["area_fraction"] > 0
    ]

    positive_presence = [
        row["patch_presence_fraction"]
        for row in rows
        if row["patch_presence_fraction"] > 0
    ]

    largest_fraction = max(
        fractions
    )

    smallest_fraction = min(
        fractions
    )

    ratio = (
        largest_fraction
        / smallest_fraction
        if smallest_fraction > 0
        else float("inf")
    )

    sorted_rows = sorted(
        rows,
        key=lambda x:
            x["area_fraction"],
        reverse=True,
    )

    return {
        "num_families":
            len(rows),

        "active_families":
            sum(
                row["area_fraction"] > 0
                for row in rows
            ),

        "largest_area_fraction":
            largest_fraction,

        "smallest_positive_area_fraction":
            smallest_fraction,

        "largest_to_smallest_positive_ratio":
            ratio,

        "median_area_fraction":
            sorted(
                fractions
            )[len(fractions) // 2],

        "mean_patch_presence":
            (
                sum(
                    positive_presence
                )
                / len(
                    positive_presence
                )
                if positive_presence
                else 0.0
            ),

        "ranked_families":
            [
                {
                    "family":
                        row[
                            "semantic_family"
                        ],

                    "area_fraction":
                        row[
                            "area_fraction"
                        ],

                    "presence_fraction":
                        row[
                            "patch_presence_fraction"
                        ],
                }
                for row in sorted_rows
            ],
    }


# ============================================================
# MAIN
# ============================================================

print(
    "=" * 70
)

print(
    "C3 19-CLASS SEMANTIC FAMILY DISTRIBUTION ANALYSIS"
)

print(
    "=" * 70
)

print()

train_targets, family_names = load_targets(
    TRAIN_PATH
)

val_targets, val_family_names = load_targets(
    VAL_PATH
)

if family_names != val_family_names:
    raise RuntimeError(
        "Train and validation family orders differ."
    )


print(
    f"Train shape:      {tuple(train_targets.shape)}"
)

print(
    f"Validation shape: {tuple(val_targets.shape)}"
)

print()


# ============================================================
# ANALYZE
# ============================================================

train_rows = analyze_split(
    train_targets,
    family_names,
    "train",
)

val_rows = analyze_split(
    val_targets,
    family_names,
    "validation",
)

all_rows = (
    train_rows
    + val_rows
)


# ============================================================
# PRINT TRAIN
# ============================================================

print(
    "=" * 70
)

print(
    "TRAIN DISTRIBUTION"
)

print(
    "=" * 70
)

print()

train_sorted = sorted(
    train_rows,
    key=lambda x:
        x["area_fraction"],
    reverse=True,
)

for rank, row in enumerate(
    train_sorted,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"{row['semantic_family']:<72}"
        f"area={row['area_fraction'] * 100:7.3f}%  "
        f"mean={row['mean_area_per_patch']:9.3f}  "
        f"presence={row['patch_presence_fraction'] * 100:7.3f}%"
    )


# ============================================================
# PRINT VALIDATION
# ============================================================

print()
print(
    "=" * 70
)

print(
    "VALIDATION DISTRIBUTION"
)

print(
    "=" * 70
)

print()

val_sorted = sorted(
    val_rows,
    key=lambda x:
        x["area_fraction"],
    reverse=True,
)

for rank, row in enumerate(
    val_sorted,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"{row['semantic_family']:<72}"
        f"area={row['area_fraction'] * 100:7.3f}%  "
        f"mean={row['mean_area_per_patch']:9.3f}  "
        f"presence={row['patch_presence_fraction'] * 100:7.3f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

train_summary = summarize(
    train_rows
)

val_summary = summarize(
    val_rows
)

print()
print(
    "=" * 70
)

print(
    "TRAIN IMBALANCE SUMMARY"
)

print(
    "=" * 70
)

print(
    f"Active families: "
    f"{train_summary['active_families']}/19"
)

print(
    f"Smallest positive fraction: "
    f"{train_summary['smallest_positive_area_fraction']:.8f}"
)

print(
    f"Largest fraction: "
    f"{train_summary['largest_area_fraction']:.8f}"
)

print(
    f"Largest/smallest ratio: "
    f"{train_summary['largest_to_smallest_positive_ratio']:.3f}"
)

print(
    f"Median fraction: "
    f"{train_summary['median_area_fraction']:.8f}"
)


print()
print(
    "=" * 70
)

print(
    "VALIDATION IMBALANCE SUMMARY"
)

print(
    "=" * 70
)

print(
    f"Active families: "
    f"{val_summary['active_families']}/19"
)

print(
    f"Smallest positive fraction: "
    f"{val_summary['smallest_positive_area_fraction']:.8f}"
)

print(
    f"Largest fraction: "
    f"{val_summary['largest_area_fraction']:.8f}"
)

print(
    f"Largest/smallest ratio: "
    f"{val_summary['largest_to_smallest_positive_ratio']:.3f}"
)

print(
    f"Median fraction: "
    f"{val_summary['median_area_fraction']:.8f}"
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "family_index",
            "semantic_family",
            "total_area",
            "mean_area_per_patch",
            "area_fraction",
            "patch_presence_fraction",
        ],
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


# ============================================================
# SAVE JSON
# ============================================================

report = {
    "train":
        train_summary,

    "validation":
        val_summary,

    "family_names":
        family_names,

    "train_shape":
        list(
            train_targets.shape
        ),

    "validation_shape":
        list(
            val_targets.shape
        ),
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print()
print(
    "=" * 70
)

print(
    "DISTRIBUTION ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

print()

print(
    "Saved:"
)

print(
    f"  {OUTPUT_CSV}"
)

print(
    f"  {OUTPUT_JSON}"
)

print()