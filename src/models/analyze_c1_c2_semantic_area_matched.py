from __future__ import annotations

import os
import re
import bisect
import pandas as pd
import numpy as np


INPUT_PATH = r"D:\SatQueryAI\data\processed\c1_c2_comparison\paired_per_sample.csv"

OUTPUT_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\c1_c2_semantic_area_matched_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIR_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_area_matched_pairs.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "semantic_area_matched_summary.csv"
)

UNMATCHED_PATH = os.path.join(
    OUTPUT_DIR,
    "unmatched_superlative_phrases.csv"
)


# -------------------------------------------------------------------------
# Convert superlative phrases to their underlying semantic class.
#
# Examples:
#   largest patch of arable land
#       -> arable land
#
#   smallest continuous region of mixed forest
#       -> mixed forest
#
#   largest connected patch of coniferous forest
#       -> coniferous forest
#
# Ordinary phrases are left unchanged.
# -------------------------------------------------------------------------

SUPERLATIVE_PATTERN = re.compile(
    r"^(largest|smallest)\s+"
    r"(?:"
    r"patch|"
    r"contiguous\s+area|"
    r"continuous\s+area|"
    r"connected\s+area|"
    r"contiguous\s+region|"
    r"continuous\s+region|"
    r"connected\s+region|"
    r"continuous\s+patch|"
    r"connected\s+patch"
    r")\s+of\s+(.+)$",
    flags=re.IGNORECASE,
)


def parse_phrase(phrase: str):
    text = str(phrase).strip().lower()

    match = SUPERLATIVE_PATTERN.match(text)

    if match:
        relation = match.group(1).lower()
        semantic_class = match.group(2).strip()

        return relation, semantic_class

    return "ordinary", text


print("=" * 80)
print("C1 vs C2 — SAME-CLASS + AREA-MATCHED ANALYSIS")
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
    "c2_minus_c1_base_iou",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing required columns:\n" +
        "\n".join(missing)
    )

df["iou_gain"] = df["c2_minus_c1_base_iou"]

print(f"Rows loaded: {len(df):,}")


# -------------------------------------------------------------------------
# Parse phrases
# -------------------------------------------------------------------------

parsed = df["phrase"].apply(parse_phrase)

df["query_relation"] = parsed.apply(
    lambda x: x[0]
)

df["semantic_class"] = parsed.apply(
    lambda x: x[1]
)


# -------------------------------------------------------------------------
# Show parsing statistics
# -------------------------------------------------------------------------

print()
print("=" * 80)
print("QUERY-TYPE DISTRIBUTION")
print("=" * 80)

print(
    df["query_relation"]
    .value_counts()
    .sort_index()
    .to_string()
)


# -------------------------------------------------------------------------
# Find superlative phrases that did not parse.
# -------------------------------------------------------------------------

superlative_mask = df["phrase"].astype(str).str.lower().str.startswith(
    ("largest ", "smallest ")
)

failed_superlative = df[
    superlative_mask &
    ~df["query_relation"].isin(["largest", "smallest"])
].copy()

if len(failed_superlative):
    unmatched = (
        failed_superlative["phrase"]
        .value_counts()
        .reset_index()
    )

    unmatched.columns = [
        "phrase",
        "samples",
    ]

    unmatched.to_csv(
        UNMATCHED_PATH,
        index=False
    )

    print()
    print("=" * 80)
    print("WARNING: UNPARSED SUPERLATIVE PHRASES")
    print("=" * 80)

    print(
        unmatched.to_string(index=False)
    )

else:
    print()
    print("All largest/smallest phrases parsed successfully.")


# -------------------------------------------------------------------------
# Only use successfully parsed relations.
# -------------------------------------------------------------------------

ordinary = df[
    df["query_relation"] == "ordinary"
].copy()

largest = df[
    df["query_relation"] == "largest"
].copy()

smallest = df[
    df["query_relation"] == "smallest"
].copy()


print()
print("=" * 80)
print("USABLE DATA")
print("=" * 80)

print(f"Ordinary : {len(ordinary):,}")
print(f"Largest  : {len(largest):,}")
print(f"Smallest : {len(smallest):,}")


# -------------------------------------------------------------------------
# Match within semantic class.
#
# For each superlative example:
#   1. require same semantic class
#   2. choose ordinary sample with nearest target area
#   3. do not reuse an ordinary sample
#
# This is performed separately for largest and smallest.
# -------------------------------------------------------------------------

def match_within_class(
    reference_df: pd.DataFrame,
    ordinary_df: pd.DataFrame,
    relation: str,
):
    records = []
    unmatched = []

    classes = sorted(
        reference_df["semantic_class"]
        .dropna()
        .unique()
    )

    for semantic_class in classes:

        ref_group = reference_df[
            reference_df["semantic_class"] ==
            semantic_class
        ].copy()

        ord_group = ordinary_df[
            ordinary_df["semantic_class"] ==
            semantic_class
        ].copy()

        if len(ord_group) == 0:
            for _, ref in ref_group.iterrows():
                unmatched.append({
                    "relation": relation,
                    "semantic_class": semantic_class,
                    "reference_row_index": ref["row_index"],
                    "reference_phrase": ref["phrase"],
                    "reason": "no ordinary examples in same semantic class",
                })

            continue

        # Sort ordinary examples by target area.
        ord_group = ord_group.sort_values(
            "target_area"
        ).reset_index(drop=True)

        available = set(
            range(len(ord_group))
        )

        # Process reference examples from smallest area
        # to largest area for deterministic matching.
        ref_group = ref_group.sort_values(
            "target_area"
        )

        for _, ref in ref_group.iterrows():

            if not available:
                unmatched.append({
                    "relation": relation,
                    "semantic_class": semantic_class,
                    "reference_row_index": ref["row_index"],
                    "reference_phrase": ref["phrase"],
                    "reason": "ordinary samples exhausted",
                })
                continue

            available_positions = sorted(
                available
            )

            areas = [
                float(ord_group.iloc[i]["target_area"])
                for i in available_positions
            ]

            target_area = float(
                ref["target_area"]
            )

            pos = bisect.bisect_left(
                areas,
                target_area
            )

            candidate_positions = []

            if pos < len(available_positions):
                candidate_positions.append(pos)

            if pos > 0:
                candidate_positions.append(pos - 1)

            best_position = min(
                candidate_positions,
                key=lambda p: abs(
                    areas[p] - target_area
                )
            )

            chosen_index = available_positions[
                best_position
            ]

            available.remove(
                chosen_index
            )

            ordinary_row = ord_group.iloc[
                chosen_index
            ]

            ref_area = float(
                ref["target_area"]
            )

            ord_area = float(
                ordinary_row["target_area"]
            )

            records.append({
                "relation": relation,

                "semantic_class":
                    semantic_class,

                "reference_row_index":
                    ref["row_index"],

                "ordinary_row_index":
                    ordinary_row["row_index"],

                "reference_patch_id":
                    ref["patch_id"],

                "ordinary_patch_id":
                    ordinary_row["patch_id"],

                "reference_phrase":
                    ref["phrase"],

                "ordinary_phrase":
                    ordinary_row["phrase"],

                "reference_target_area":
                    ref_area,

                "ordinary_target_area":
                    ord_area,

                "absolute_area_difference":
                    abs(
                        ref_area -
                        ord_area
                    ),

                "reference_c1_base_iou":
                    ref["c1_base_iou"],

                "ordinary_c1_base_iou":
                    ordinary_row["c1_base_iou"],

                "reference_c2_iou":
                    ref["c2_iou"],

                "ordinary_c2_iou":
                    ordinary_row["c2_iou"],

                "reference_c2_gain":
                    ref["iou_gain"],

                "ordinary_c2_gain":
                    ordinary_row["iou_gain"],

                "c1_gap_reference_minus_ordinary":
                    (
                        ref["c1_base_iou"] -
                        ordinary_row["c1_base_iou"]
                    ),

                "c2_gap_reference_minus_ordinary":
                    (
                        ref["c2_iou"] -
                        ordinary_row["c2_iou"]
                    ),

                "c2_gain_gap_reference_minus_ordinary":
                    (
                        ref["iou_gain"] -
                        ordinary_row["iou_gain"]
                    ),
            })

    return (
        pd.DataFrame(records),
        pd.DataFrame(unmatched),
    )


largest_pairs, largest_unmatched = match_within_class(
    largest,
    ordinary,
    "largest"
)

smallest_pairs, smallest_unmatched = match_within_class(
    smallest,
    ordinary,
    "smallest"
)

pairs = pd.concat(
    [
        largest_pairs,
        smallest_pairs,
    ],
    ignore_index=True
)

unmatched = pd.concat(
    [
        largest_unmatched,
        smallest_unmatched,
    ],
    ignore_index=True
)


pairs.to_csv(
    PAIR_PATH,
    index=False
)


if len(unmatched):
    unmatched.to_csv(
        UNMATCHED_PATH,
        index=False
    )


# -------------------------------------------------------------------------
# Aggregate summary
# -------------------------------------------------------------------------

def summarize(group):

    return pd.Series({
        "matched_samples":
            len(group),

        "semantic_classes":
            1,

        "mean_area_difference":
            group["absolute_area_difference"].mean(),

        "median_area_difference":
            group["absolute_area_difference"].median(),

        "95pct_area_difference":
            group[
                "absolute_area_difference"
            ].quantile(0.95),

        "reference_c1_base_iou":
            group[
                "reference_c1_base_iou"
            ].mean(),

        "ordinary_c1_base_iou":
            group[
                "ordinary_c1_base_iou"
            ].mean(),

        "c1_gap":
            group[
                "c1_gap_reference_minus_ordinary"
            ].mean(),

        "reference_c2_iou":
            group[
                "reference_c2_iou"
            ].mean(),

        "ordinary_c2_iou":
            group[
                "ordinary_c2_iou"
            ].mean(),

        "c2_gap":
            group[
                "c2_gap_reference_minus_ordinary"
            ].mean(),

        "reference_c2_gain":
            group[
                "reference_c2_gain"
            ].mean(),

        "ordinary_c2_gain":
            group[
                "ordinary_c2_gain"
            ].mean(),

        "c2_gain_gap":
            group[
                "c2_gain_gap_reference_minus_ordinary"
            ].mean(),

        "reference_improved_pct":
            100.0 * (
                group[
                    "reference_c2_gain"
                ] > 1e-9
            ).mean(),

        "ordinary_improved_pct":
            100.0 * (
                group[
                    "ordinary_c2_gain"
                ] > 1e-9
            ).mean(),
    })


summary = (
    pairs
    .groupby(
        "relation",
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


# -------------------------------------------------------------------------
# Per semantic-class summary
# -------------------------------------------------------------------------

class_summary = (
    pairs
    .groupby(
        [
            "relation",
            "semantic_class",
        ],
        sort=True
    )
    .apply(
        summarize,
        include_groups=False
    )
    .reset_index()
)

class_summary_path = os.path.join(
    OUTPUT_DIR,
    "semantic_class_summary.csv"
)

class_summary.to_csv(
    class_summary_path,
    index=False
)


# -------------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------------

print()
print("=" * 80)
print("SAME-CLASS + AREA-MATCHED SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("MATCH COVERAGE")
print("=" * 80)

print(
    f"Largest matched : "
    f"{len(largest_pairs):,} / {len(largest):,}"
)

print(
    f"Smallest matched: "
    f"{len(smallest_pairs):,} / {len(smallest):,}"
)

print(
    f"Unmatched       : "
    f"{len(unmatched):,}"
)


# -------------------------------------------------------------------------
# Tight area match: <= 0.01
# -------------------------------------------------------------------------

print()
print("=" * 80)
print("TIGHT MATCH: ABSOLUTE TARGET-AREA DIFFERENCE <= 0.01")
print("=" * 80)

tight = pairs[
    pairs["absolute_area_difference"] <= 0.01
].copy()

print(
    f"Total tight matches: {len(tight):,}"
)

for relation in ["largest", "smallest"]:

    part = tight[
        tight["relation"] ==
        relation
    ]

    print()
    print(relation.upper())
    print("-" * 80)

    if len(part) == 0:
        print("No tight matches.")
        continue

    print(
        f"Samples: {len(part):,}"
    )

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
        f"{part['c1_gap_reference_minus_ordinary'].mean():+.9f}"
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
        f"{part['c2_gap_reference_minus_ordinary'].mean():+.9f}"
    )

    print(
        f"Reference C2 gain: "
        f"{part['reference_c2_gain'].mean():+.9f}"
    )

    print(
        f"Ordinary C2 gain: "
        f"{part['ordinary_c2_gain'].mean():+.9f}"
    )


# -------------------------------------------------------------------------
# Semantic-class examples
# -------------------------------------------------------------------------

print()
print("=" * 80)
print("LARGEST SEMANTIC-CLASS GAPS")
print("=" * 80)

largest_class = class_summary[
    class_summary["relation"] ==
    "largest"
].copy()

if len(largest_class):
    print(
        largest_class
        .sort_values("c2_gap")
        .head(15)
        .to_string(index=False)
    )


print()
print("=" * 80)
print("SMALLEST SEMANTIC-CLASS GAPS")
print("=" * 80)

smallest_class = class_summary[
    class_summary["relation"] ==
    "smallest"
].copy()

if len(smallest_class):
    print(
        smallest_class
        .sort_values("c2_gap")
        .head(15)
        .to_string(index=False)
    )


print()
print("=" * 80)
print("FILES")
print("=" * 80)

print(
    f"Pairs:\n{PAIR_PATH}"
)

print(
    f"Overall summary:\n{SUMMARY_PATH}"
)

print(
    f"Class summary:\n{class_summary_path}"
)

if len(unmatched):
    print(
        f"Unmatched:\n{UNMATCHED_PATH}"
    )

print()
print("=" * 80)
print("DONE")
print("=" * 80)