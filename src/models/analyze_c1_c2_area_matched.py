from __future__ import annotations

import os
import bisect
import numpy as np
import pandas as pd


INPUT_PATH = r"D:\SatQueryAI\data\processed\c1_c2_comparison\paired_per_sample.csv"
OUTPUT_DIR = r"D:\SatQueryAI\data\processed\c1_c2_area_matched_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "area_matched_summary.csv"
)

PAIRED_PATH = os.path.join(
    OUTPUT_DIR,
    "area_matched_pairs.csv"
)


def classify_reference(phrase: str) -> str:
    p = str(phrase).strip().lower()

    if p.startswith("largest "):
        return "largest"

    if p.startswith("smallest "):
        return "smallest"

    return "ordinary"


def nearest_match_without_replacement(
    reference_areas: np.ndarray,
    ordinary_areas: np.ndarray,
):
    """
    For every reference sample, find one ordinary-reference sample
    with the closest target area, without reusing ordinary samples.

    The ordinary samples are kept sorted by target area, and a
    local nearest-neighbor search is performed with bisect.
    """
    order = np.argsort(ordinary_areas)
    sorted_areas = ordinary_areas[order]

    available = list(range(len(sorted_areas)))

    matches = []

    for ref_idx, ref_area in sorted(
        enumerate(reference_areas),
        key=lambda x: x[1]
    ):
        if not available:
            break

        values = [sorted_areas[i] for i in available]
        pos = bisect.bisect_left(values, ref_area)

        candidates = []

        if pos < len(available):
            candidates.append(pos)

        if pos > 0:
            candidates.append(pos - 1)

        best_pos = min(
            candidates,
            key=lambda p: abs(
                sorted_areas[available[p]] - ref_area
            )
        )

        selected_sorted_position = available.pop(best_pos)
        selected_original_index = int(
            order[selected_sorted_position]
        )

        matches.append(
            (
                int(ref_idx),
                selected_original_index
            )
        )

    return matches


print("=" * 80)
print("C1 vs C2 — NEAREST TARGET-AREA MATCHED ANALYSIS")
print("=" * 80)

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(INPUT_PATH)

df = pd.read_csv(INPUT_PATH)

required = [
    "row_index",
    "patch_id",
    "phrase",
    "target_area",
    "c1_base_iou",
    "c2_iou",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing required columns:\n" +
        "\n".join(missing)
    )

df["reference_type"] = df["phrase"].apply(
    classify_reference
)

df["iou_gain"] = (
    df["c2_iou"] -
    df["c1_base_iou"]
)

ordinary = df[
    df["reference_type"] == "ordinary"
].reset_index(drop=True)

largest = df[
    df["reference_type"] == "largest"
].reset_index(drop=True)

smallest = df[
    df["reference_type"] == "smallest"
].reset_index(drop=True)


print()
print(f"Ordinary samples : {len(ordinary):,}")
print(f"Largest samples  : {len(largest):,}")
print(f"Smallest samples : {len(smallest):,}")


all_pairs = []


def build_pairs(
    reference_df: pd.DataFrame,
    reference_type: str,
):
    ref_areas = reference_df["target_area"].to_numpy(
        dtype=float
    )

    ordinary_areas = ordinary["target_area"].to_numpy(
        dtype=float
    )

    matches = nearest_match_without_replacement(
        ref_areas,
        ordinary_areas
    )

    rows = []

    for ref_idx, ordinary_idx in matches:

        ref = reference_df.iloc[ref_idx]
        ord_row = ordinary.iloc[ordinary_idx]

        area_diff = (
            ref["target_area"] -
            ord_row["target_area"]
        )

        abs_area_diff = abs(area_diff)

        rows.append({
            "reference_type": reference_type,

            "reference_row_index":
                ref["row_index"],

            "ordinary_row_index":
                ord_row["row_index"],

            "reference_patch_id":
                ref["patch_id"],

            "ordinary_patch_id":
                ord_row["patch_id"],

            "reference_phrase":
                ref["phrase"],

            "ordinary_phrase":
                ord_row["phrase"],

            "reference_target_area":
                ref["target_area"],

            "ordinary_target_area":
                ord_row["target_area"],

            "target_area_difference":
                area_diff,

            "absolute_target_area_difference":
                abs_area_diff,

            "reference_c1_base_iou":
                ref["c1_base_iou"],

            "ordinary_c1_base_iou":
                ord_row["c1_base_iou"],

            "reference_c2_iou":
                ref["c2_iou"],

            "ordinary_c2_iou":
                ord_row["c2_iou"],

            "reference_c1_to_c2_gain":
                ref["iou_gain"],

            "ordinary_c1_to_c2_gain":
                ord_row["iou_gain"],

            "c1_base_iou_gap_vs_ordinary":
                ref["c1_base_iou"] -
                ord_row["c1_base_iou"],

            "c2_iou_gap_vs_ordinary":
                ref["c2_iou"] -
                ord_row["c2_iou"],

            "c2_gain_gap_vs_ordinary":
                ref["iou_gain"] -
                ord_row["iou_gain"],
        })

    return pd.DataFrame(rows)


largest_pairs = build_pairs(
    largest,
    "largest"
)

smallest_pairs = build_pairs(
    smallest,
    "smallest"
)

paired = pd.concat(
    [
        largest_pairs,
        smallest_pairs
    ],
    ignore_index=True
)

paired.to_csv(
    PAIRED_PATH,
    index=False
)


def summarize(group: pd.DataFrame):

    return pd.Series({
        "matched_samples":
            len(group),

        "mean_reference_target_area":
            group["reference_target_area"].mean(),

        "mean_ordinary_target_area":
            group["ordinary_target_area"].mean(),

        "mean_absolute_area_difference":
            group[
                "absolute_target_area_difference"
            ].mean(),

        "median_absolute_area_difference":
            group[
                "absolute_target_area_difference"
            ].median(),

        "maximum_absolute_area_difference":
            group[
                "absolute_target_area_difference"
            ].max(),

        "reference_c1_base_iou":
            group[
                "reference_c1_base_iou"
            ].mean(),

        "ordinary_c1_base_iou":
            group[
                "ordinary_c1_base_iou"
            ].mean(),

        "c1_base_iou_gap":
            group[
                "c1_base_iou_gap_vs_ordinary"
            ].mean(),

        "reference_c2_iou":
            group[
                "reference_c2_iou"
            ].mean(),

        "ordinary_c2_iou":
            group[
                "ordinary_c2_iou"
            ].mean(),

        "c2_iou_gap":
            group[
                "c2_iou_gap_vs_ordinary"
            ].mean(),

        "reference_c2_gain":
            group[
                "reference_c1_to_c2_gain"
            ].mean(),

        "ordinary_c2_gain":
            group[
                "ordinary_c1_to_c2_gain"
            ].mean(),

        "c2_gain_gap":
            group[
                "c2_gain_gap_vs_ordinary"
            ].mean(),

        "reference_improved_pct":
            100.0 * (
                group[
                    "reference_c1_to_c2_gain"
                ] > 1e-9
            ).mean(),

        "ordinary_improved_pct":
            100.0 * (
                group[
                    "ordinary_c1_to_c2_gain"
                ] > 1e-9
            ).mean(),
    })


summary = (
    paired
    .groupby(
        "reference_type",
        sort=True
    )
    .apply(
        summarize,
        include_groups=False
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


print()
print("=" * 80)
print("NEAREST-AREA MATCHED RESULTS")
print("=" * 80)

print(
    summary.to_string(index=False)
)


print()
print("=" * 80)
print("AREA MATCH QUALITY")
print("=" * 80)

for ref_type in ["largest", "smallest"]:

    part = paired[
        paired["reference_type"] == ref_type
    ]

    print()
    print(ref_type.upper())

    print(
        f"Matched samples: {len(part):,}"
    )

    print(
        f"Mean absolute area difference: "
        f"{part['absolute_target_area_difference'].mean():.9f}"
    )

    print(
        f"Median absolute area difference: "
        f"{part['absolute_target_area_difference'].median():.9f}"
    )

    print(
        f"95th percentile area difference: "
        f"{part['absolute_target_area_difference'].quantile(0.95):.9f}"
    )

    print(
        f"Maximum area difference: "
        f"{part['absolute_target_area_difference'].max():.9f}"
    )


print()
print("=" * 80)
print("WITHIN-MATCH C2 EFFECT")
print("=" * 80)

for ref_type in ["largest", "smallest"]:

    part = paired[
        paired["reference_type"] == ref_type
    ]

    print()
    print(ref_type.upper())

    print(
        "Reference C1 base IoU: "
        f"{part['reference_c1_base_iou'].mean():.9f}"
    )

    print(
        "Matched ordinary C1 base IoU: "
        f"{part['ordinary_c1_base_iou'].mean():.9f}"
    )

    print(
        "Reference C2 IoU: "
        f"{part['reference_c2_iou'].mean():.9f}"
    )

    print(
        "Matched ordinary C2 IoU: "
        f"{part['ordinary_c2_iou'].mean():.9f}"
    )

    print(
        "Reference C2 gain: "
        f"{part['reference_c1_to_c2_gain'].mean():+.9f}"
    )

    print(
        "Matched ordinary C2 gain: "
        f"{part['ordinary_c1_to_c2_gain'].mean():+.9f}"
    )

    print(
        "Reference-vs-ordinary C1 gap: "
        f"{part['c1_base_iou_gap_vs_ordinary'].mean():+.9f}"
    )

    print(
        "Reference-vs-ordinary C2 gap: "
        f"{part['c2_iou_gap_vs_ordinary'].mean():+.9f}"
    )


print()
print("=" * 80)
print("OPTIONAL TIGHT-MATCH SUBSET: AREA DIFFERENCE <= 0.01")
print("=" * 80)

tight = paired[
    paired["absolute_target_area_difference"] <= 0.01
].copy()

print(
    f"Total tight matches: {len(tight):,}"
)

for ref_type in ["largest", "smallest"]:

    part = tight[
        tight["reference_type"] == ref_type
    ]

    print()
    print(ref_type.upper())

    print(
        f"Samples: {len(part):,}"
    )

    if len(part) == 0:
        continue

    print(
        f"Reference C1 base IoU: "
        f"{part['reference_c1_base_iou'].mean():.9f}"
    )

    print(
        f"Ordinary C1 base IoU: "
        f"{part['ordinary_c1_base_iou'].mean():.9f}"
    )

    print(
        f"C1 gap: "
        f"{part['c1_base_iou_gap_vs_ordinary'].mean():+.9f}"
    )

    print(
        f"Reference C2 IoU: "
        f"{part['reference_c2_iou'].mean():.9f}"
    )

    print(
        f"Ordinary C2 IoU: "
        f"{part['ordinary_c2_iou'].mean():.9f}"
    )

    print(
        f"C2 gap: "
        f"{part['c2_iou_gap_vs_ordinary'].mean():+.9f}"
    )

    print(
        f"Reference C2 gain: "
        f"{part['reference_c1_to_c2_gain'].mean():+.9f}"
    )

    print(
        f"Ordinary C2 gain: "
        f"{part['ordinary_c1_to_c2_gain'].mean():+.9f}"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Summary saved to:\n{SUMMARY_PATH}"
)

print(
    f"Paired rows saved to:\n{PAIRED_PATH}"
)