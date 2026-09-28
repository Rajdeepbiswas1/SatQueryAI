from __future__ import annotations

import os
import pandas as pd

INPUT_PATH = r"D:\SatQueryAI\data\processed\c1_c2_comparison\paired_per_sample.csv"
OUTPUT_DIR = r"D:\SatQueryAI\data\processed\c1_c2_size_matched_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUTPUT_DIR,
    "size_matched_reference_analysis.csv"
)


def classify_reference(phrase: str) -> str:
    p = str(phrase).strip().lower()

    if p.startswith("largest "):
        return "largest"

    if p.startswith("smallest "):
        return "smallest"

    return "ordinary"


df = pd.read_csv(INPUT_PATH)

required = [
    "row_index",
    "phrase",
    "target_area",
    "c1_base_iou",
    "c2_iou",
    "c2_minus_c1_base_iou",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing required columns:\n" + "\n".join(missing)
    )

print("=" * 80)
print("C1 vs C2 — SIZE-MATCHED REFERENCE ANALYSIS")
print("=" * 80)

print(f"Rows: {len(df):,}")

df["reference_type"] = df["phrase"].apply(classify_reference)

df["iou_gain"] = df["c2_minus_c1_base_iou"]

# -------------------------------------------------------------------------
# Create global equal-frequency target-size bins.
# These bins are shared by all reference types.
# -------------------------------------------------------------------------

df["size_bin"] = pd.qcut(
    df["target_area"],
    q=6,
    labels=[
        "tiny",
        "small",
        "medium",
        "large",
        "very_large",
        "huge",
    ],
    duplicates="drop",
)

# -------------------------------------------------------------------------
# Aggregate by reference type AND target size.
# -------------------------------------------------------------------------

summary = (
    df
    .groupby(
        ["reference_type", "size_bin"],
        observed=False
    )
    .agg(
        samples=("row_index", "count"),
        mean_target_area=("target_area", "mean"),
        c1_base_iou=("c1_base_iou", "mean"),
        c2_iou=("c2_iou", "mean"),
        mean_iou_gain=("iou_gain", "mean"),
        median_iou_gain=("iou_gain", "median"),
        improved_pct=(
            "iou_gain",
            lambda x: 100.0 * (x > 1e-9).mean()
        ),
        worse_pct=(
            "iou_gain",
            lambda x: 100.0 * (x < -1e-9).mean()
        ),
    )
    .reset_index()
)

summary.to_csv(OUT_PATH, index=False)

print()
print("=" * 80)
print("SIZE-MATCHED RESULTS")
print("=" * 80)

for size in [
    "tiny",
    "small",
    "medium",
    "large",
    "very_large",
    "huge",
]:
    part = summary[summary["size_bin"].astype(str) == size]

    if len(part) == 0:
        continue

    print()
    print(f"{size.upper()}")
    print("-" * 80)

    print(
        part[
            [
                "reference_type",
                "samples",
                "mean_target_area",
                "c1_base_iou",
                "c2_iou",
                "mean_iou_gain",
                "improved_pct",
                "worse_pct",
            ]
        ].to_string(index=False)
    )


print()
print("=" * 80)
print("DIRECT SIZE-MATCHED COMPARISONS")
print("=" * 80)

for size in [
    "tiny",
    "small",
    "medium",
    "large",
    "very_large",
    "huge",
]:

    part = summary[
        summary["size_bin"].astype(str) == size
    ].set_index("reference_type")

    if not all(
        x in part.index
        for x in ["ordinary", "largest", "smallest"]
    ):
        continue

    print()
    print(f"{size.upper()}:")

    ordinary = part.loc["ordinary"]
    largest = part.loc["largest"]
    smallest = part.loc["smallest"]

    print(
        f"  ordinary : "
        f"C1={ordinary.c1_base_iou:.6f} "
        f"C2={ordinary.c2_iou:.6f} "
        f"gain={ordinary.mean_iou_gain:+.6f} "
        f"N={int(ordinary.samples)}"
    )

    print(
        f"  largest  : "
        f"C1={largest.c1_base_iou:.6f} "
        f"C2={largest.c2_iou:.6f} "
        f"gain={largest.mean_iou_gain:+.6f} "
        f"N={int(largest.samples)}"
    )

    print(
        f"  smallest : "
        f"C1={smallest.c1_base_iou:.6f} "
        f"C2={smallest.c2_iou:.6f} "
        f"gain={smallest.mean_iou_gain:+.6f} "
        f"N={int(smallest.samples)}"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(f"Saved: {OUT_PATH}")