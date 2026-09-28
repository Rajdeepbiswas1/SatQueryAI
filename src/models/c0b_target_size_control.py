"""
SATQUERYAI — C0-B TARGET-SIZE CONTROL

Purpose
-------
Determine whether the lower performance of superlative queries
(largest/smallest/etc.) remains after controlling for target size.

Uses the already-generated C0 CSV:

    data/processed/reference_grounding_c0/
        validation_predictions_c0.csv

No model loading.
No GPU.
No training.

Outputs:
    1. Overall largest vs smallest comparison
    2. Target-box area statistics
    3. IoU by target-area bins
    4. IoU by target-area bins for:
         - largest
         - smallest
         - non-superlative
    5. CSV + JSON results
"""

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_c0"
    / "validation_predictions_c0.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_c0"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BIN_RESULTS_CSV = (
    OUTPUT_DIR
    / "c0b_target_size_bins.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "c0b_target_size_control_summary.json"
)


# ======================================================================
# SUPERLATIVE DETECTION
# ======================================================================

LARGEST_PATTERN = re.compile(
    r"\b(largest|biggest|greatest|maximum|max)\b",
    re.IGNORECASE,
)

SMALLEST_PATTERN = re.compile(
    r"\b(smallest|tiniest|least|minimum|min)\b",
    re.IGNORECASE,
)


def classify_query(phrase):
    """
    Classify a query into:
        largest
        smallest
        other_superlative
        non_superlative
    """

    text = str(phrase).lower()

    has_largest = bool(
        LARGEST_PATTERN.search(text)
    )

    has_smallest = bool(
        SMALLEST_PATTERN.search(text)
    )

    if has_largest and not has_smallest:
        return "largest"

    if has_smallest and not has_largest:
        return "smallest"

    # Covers terms such as most/highest/lowest/least that may have
    # appeared in the earlier C0 superlative detector.
    generic_superlatives = [
        r"\bhighest\b",
        r"\blowest\b",
        r"\bmost\b",
        r"\bleast\b",
    ]

    if any(
        re.search(p, text)
        for p in generic_superlatives
    ):
        return "other_superlative"

    return "non_superlative"


# ======================================================================
# SUMMARY FUNCTION
# ======================================================================

def summarize(df, name):

    if len(df) == 0:
        return {
            "group": name,
            "count": 0,
            "mean_iou": None,
            "median_iou": None,
            "mean_target_area": None,
            "median_target_area": None,
            "iou_ge_025_pct": None,
            "iou_ge_050_pct": None,
            "iou_ge_075_pct": None,
        }

    return {
        "group": name,
        "count": int(len(df)),
        "mean_iou": float(
            df["iou"].mean()
        ),
        "median_iou": float(
            df["iou"].median()
        ),
        "mean_target_area": float(
            df["target_area"].mean()
        ),
        "median_target_area": float(
            df["target_area"].median()
        ),
        "iou_ge_025_pct": float(
            100.0 * (df["iou"] >= 0.25).mean()
        ),
        "iou_ge_050_pct": float(
            100.0 * (df["iou"] >= 0.50).mean()
        ),
        "iou_ge_075_pct": float(
            100.0 * (df["iou"] >= 0.75).mean()
        ),
    }


# ======================================================================
# LOAD
# ======================================================================

print("=" * 70)
print("SATQUERYAI — C0-B TARGET-SIZE CONTROL")
print("=" * 70)

print()
print(
    f"Input CSV: {INPUT_CSV}"
)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

print()
print(
    f"Rows loaded: {len(df):,}"
)


# ======================================================================
# VALIDATE REQUIRED COLUMNS
# ======================================================================

required = [
    "phrase",
    "target_x1",
    "target_y1",
    "target_x2",
    "target_y2",
    "iou",
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ======================================================================
# TARGET AREA
# ======================================================================

df["target_width"] = (
    df["target_x2"]
    - df["target_x1"]
)

df["target_height"] = (
    df["target_y2"]
    - df["target_y1"]
)

df["target_area"] = (
    df["target_width"]
    * df["target_height"]
)

# Safety check.
df["target_area"] = (
    df["target_area"]
    .clip(lower=0.0)
)


# ======================================================================
# QUERY CLASS
# ======================================================================

df["query_type"] = (
    df["phrase"]
    .apply(classify_query)
)


# ======================================================================
# OVERALL GROUP SUMMARIES
# ======================================================================

print()
print("=" * 70)
print("OVERALL QUERY-TYPE RESULTS")
print("=" * 70)

groups = {}

for group_name in [
    "largest",
    "smallest",
    "other_superlative",
    "non_superlative",
]:

    subset = df[
        df["query_type"] == group_name
    ]

    groups[group_name] = summarize(
        subset,
        group_name,
    )

    print()
    print(
        json.dumps(
            groups[group_name],
            indent=2,
        )
    )


# ======================================================================
# SUPERLATIVE TOTALS
# ======================================================================

super_df = df[
    df["query_type"].isin(
        [
            "largest",
            "smallest",
            "other_superlative",
        ]
    )
]

non_super_df = df[
    df["query_type"] == "non_superlative"
]

groups["all_superlative"] = summarize(
    super_df,
    "all_superlative",
)

groups["non_superlative"] = summarize(
    non_super_df,
    "non_superlative",
)


# ======================================================================
# IMPORTANT COMPARISON
# ======================================================================

print()
print("=" * 70)
print("LARGEST VS SMALLEST")
print("=" * 70)

largest_df = df[
    df["query_type"] == "largest"
]

smallest_df = df[
    df["query_type"] == "smallest"
]

print()
print(
    "Largest:"
)

print(
    json.dumps(
        summarize(
            largest_df,
            "largest",
        ),
        indent=2,
    )
)

print()
print(
    "Smallest:"
)

print(
    json.dumps(
        summarize(
            smallest_df,
            "smallest",
        ),
        indent=2,
    )
)


# ======================================================================
# TARGET AREA DISTRIBUTION
# ======================================================================

print()
print("=" * 70)
print("TARGET AREA DISTRIBUTION")
print("=" * 70)

area_stats = {}

for group_name, subset in [
    ("largest", largest_df),
    ("smallest", smallest_df),
    ("non_superlative", non_super_df),
]:

    if len(subset) == 0:
        continue

    areas = subset["target_area"]

    area_stats[group_name] = {
        "count": int(len(areas)),
        "min": float(areas.min()),
        "p05": float(areas.quantile(0.05)),
        "p10": float(areas.quantile(0.10)),
        "p25": float(areas.quantile(0.25)),
        "median": float(areas.median()),
        "p75": float(areas.quantile(0.75)),
        "p90": float(areas.quantile(0.90)),
        "p95": float(areas.quantile(0.95)),
        "max": float(areas.max()),
        "mean": float(areas.mean()),
    }

    print()
    print(
        group_name
    )

    print(
        json.dumps(
            area_stats[group_name],
            indent=2,
        )
    )


# ======================================================================
# COMMON TARGET-AREA BINS
# ======================================================================
#
# These bins are defined over the entire validation set so that
# comparisons across query groups use the same intervals.
#
# Important:
# This is a descriptive control, not a causal statistical model.
# ======================================================================

quantiles = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.00,
]

edges = (
    df["target_area"]
    .quantile(quantiles)
    .values
)

# Remove duplicate edges.
edges = np.unique(edges)

if len(edges) < 3:

    raise RuntimeError(
        "Not enough unique target-area values to build bins."
    )


# Make sure final edge includes maximum.
edges[0] = min(
    edges[0],
    float(df["target_area"].min()),
)

edges[-1] = max(
    edges[-1],
    float(df["target_area"].max()),
)


labels = []

for i in range(
    len(edges) - 1
):

    labels.append(
        f"{edges[i]:.6f}"
        f"–"
        f"{edges[i + 1]:.6f}"
    )


df["target_area_bin"] = pd.cut(
    df["target_area"],
    bins=edges,
    labels=labels,
    include_lowest=True,
    duplicates="drop",
)


# ======================================================================
# BIN ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("IOU BY TARGET AREA")
print("=" * 70)

bin_rows = []

for bin_value, bin_df in df.groupby(
    "target_area_bin",
    observed=True,
):

    print()
    print(
        f"Target area bin: {bin_value}"
    )

    for group_name in [
        "largest",
        "smallest",
        "non_superlative",
    ]:

        subset = bin_df[
            bin_df["query_type"]
            == group_name
        ]

        if len(subset) == 0:
            row = {
                "target_area_bin":
                    str(bin_value),
                "query_type":
                    group_name,
                "count":
                    0,
                "mean_iou":
                    None,
                "median_iou":
                    None,
                "mean_target_area":
                    None,
            }

        else:

            row = {
                "target_area_bin":
                    str(bin_value),
                "query_type":
                    group_name,
                "count":
                    int(len(subset)),
                "mean_iou":
                    float(
                        subset["iou"].mean()
                    ),
                "median_iou":
                    float(
                        subset["iou"].median()
                    ),
                "mean_target_area":
                    float(
                        subset["target_area"].mean()
                    ),
            }

        bin_rows.append(row)

        print(
            f"  {group_name:18s} "
            f"n={row['count']:5d} "
            f"mean IoU="
            f"{row['mean_iou'] if row['mean_iou'] is not None else 'N/A'}"
        )


bin_results = pd.DataFrame(
    bin_rows
)

bin_results.to_csv(
    BIN_RESULTS_CSV,
    index=False,
)


# ======================================================================
# SIZE-MATCHED COMPARISON
# ======================================================================
#
# Calculate the superlative vs non-superlative difference in each
# shared target-area bin.
# ======================================================================

print()
print("=" * 70)
print("SIZE-MATCHED SUPERLATIVE GAP")
print("=" * 70)

matched_rows = []

for bin_value, bin_df in df.groupby(
    "target_area_bin",
    observed=True,
):

    super_subset = bin_df[
        bin_df["query_type"].isin(
            [
                "largest",
                "smallest",
                "other_superlative",
            ]
        )
    ]

    non_subset = bin_df[
        bin_df["query_type"]
        == "non_superlative"
    ]

    if (
        len(super_subset) == 0
        or len(non_subset) == 0
    ):
        continue

    super_iou = float(
        super_subset["iou"].mean()
    )

    non_iou = float(
        non_subset["iou"].mean()
    )

    difference = (
        super_iou
        - non_iou
    )

    row = {
        "target_area_bin":
            str(bin_value),
        "superlative_count":
            int(len(super_subset)),
        "non_superlative_count":
            int(len(non_subset)),
        "superlative_mean_iou":
            super_iou,
        "non_superlative_mean_iou":
            non_iou,
        "superlative_minus_non_superlative":
            difference,
    }

    matched_rows.append(row)

    print()
    print(
        f"{bin_value}"
    )

    print(
        f"  Superlative:     "
        f"n={len(super_subset):4d} "
        f"IoU={super_iou:.6f}"
    )

    print(
        f"  Non-superlative: "
        f"n={len(non_subset):4d} "
        f"IoU={non_iou:.6f}"
    )

    print(
        f"  Difference:      "
        f"{difference:+.6f}"
    )


matched_df = pd.DataFrame(
    matched_rows
)


# ======================================================================
# SIZE-MATCHED AGGREGATE
# ======================================================================
#
# Also calculate a simple pooled comparison across all rows where
# both classes exist in a target-area bin.
#
# This is descriptive only; it prevents us from claiming a causal
# effect from the query wording.
# ======================================================================

if len(matched_df) > 0:

    valid_bins = set(
        matched_df[
            "target_area_bin"
        ]
    )

    pooled = df[
        df["target_area_bin"]
        .astype(str)
        .isin(
            set(
                str(x)
                for x in valid_bins
            )
        )
    ].copy()

    pooled_super = pooled[
        pooled["query_type"].isin(
            [
                "largest",
                "smallest",
                "other_superlative",
            ]
        )
    ]

    pooled_non = pooled[
        pooled["query_type"]
        == "non_superlative"
    ]

    pooled_super_iou = float(
        pooled_super["iou"].mean()
    )

    pooled_non_iou = float(
        pooled_non["iou"].mean()
    )

    pooled_difference = (
        pooled_super_iou
        - pooled_non_iou
    )

else:

    pooled_super_iou = None
    pooled_non_iou = None
    pooled_difference = None


# ======================================================================
# SAVE SUMMARY
# ======================================================================

summary = {
    "input_csv":
        str(INPUT_CSV),

    "rows":
        int(len(df)),

    "query_groups":
        groups,

    "target_area_distribution":
        area_stats,

    "size_matched_bins":
        matched_rows,

    "size_matched_pooled": {
        "superlative_count":
            int(len(pooled_super))
            if len(matched_df) > 0
            else 0,

        "non_superlative_count":
            int(len(pooled_non))
            if len(matched_df) > 0
            else 0,

        "superlative_mean_iou":
            pooled_super_iou,

        "non_superlative_mean_iou":
            pooled_non_iou,

        "difference":
            pooled_difference,
    },

    "interpretation_note":
        (
            "This is a descriptive target-size control. "
            "It tests whether the superlative/non-superlative "
            "IoU gap persists within common target-area bins. "
            "It does not establish causality."
        ),
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ======================================================================
# FINAL MESSAGE
# ======================================================================

print()
print("=" * 70)
print("C0-B COMPLETE")
print("=" * 70)

print()
print(
    f"Detailed bin results:"
)

print(
    BIN_RESULTS_CSV
)

print()
print(
    f"Summary:"
)

print(
    SUMMARY_JSON
)

print()
print(
    "No model was loaded."
)

print(
    "No GPU was used."
)

print(
    "No training was performed."
)