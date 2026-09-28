from __future__ import annotations

import os
import pandas as pd
import numpy as np


INPUT_PATH = r"D:\SatQueryAI\data\processed\c1_c2_comparison\paired_per_sample.csv"
OUTPUT_DIR = r"D:\SatQueryAI\data\processed\c1_c2_reference_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "reference_type_summary.csv"
)

PHRASE_PATH = os.path.join(
    OUTPUT_DIR,
    "reference_phrase_summary.csv"
)


def classify_reference_phrase(phrase: str) -> str:
    p = str(phrase).strip().lower()

    if p.startswith("largest "):
        return "superlative-largest"

    if p.startswith("smallest "):
        return "superlative-smallest"

    return "ordinary-reference"


print("=" * 80)
print("C1 vs C2 — REFERENCE TYPE ANALYSIS")
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
        "Missing required columns:\n" + "\n".join(missing)
    )

print(f"Rows loaded: {len(df):,}")

df["reference_type"] = df["phrase"].apply(
    classify_reference_phrase
)

df["iou_gain"] = (
    df["c2_iou"] -
    df["c1_base_iou"]
)

df["c2_improved"] = (
    df["iou_gain"] > 1e-9
)

df["c2_worse"] = (
    df["iou_gain"] < -1e-9
)

df["c2_unchanged"] = (
    df["iou_gain"].abs() <= 1e-9
)


def summarize(group):
    return pd.Series({
        "samples": len(group),

        "mean_target_area":
            group["target_area"].mean(),

        "c1_base_iou":
            group["c1_base_iou"].mean(),

        "c2_iou":
            group["c2_iou"].mean(),

        "mean_iou_gain":
            group["iou_gain"].mean(),

        "median_iou_gain":
            group["iou_gain"].median(),

        "improved_pct":
            100.0 * group["c2_improved"].mean(),

        "worse_pct":
            100.0 * group["c2_worse"].mean(),

        "unchanged_pct":
            100.0 * group["c2_unchanged"].mean(),
    })


print()
print("=" * 80)
print("REFERENCE TYPE DISTRIBUTION")
print("=" * 80)

type_summary = (
    df
    .groupby("reference_type", sort=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)

print(type_summary.to_string(index=False))

type_summary.to_csv(
    SUMMARY_PATH,
    index=False
)


print()
print("=" * 80)
print("PER-PHRASE ANALYSIS")
print("=" * 80)

phrase_summary = (
    df
    .groupby(
        ["reference_type", "phrase"],
        sort=True
    )
    .apply(summarize, include_groups=False)
    .reset_index()
)

phrase_summary.to_csv(
    PHRASE_PATH,
    index=False
)

print(
    f"Unique phrases: "
    f"{df['phrase'].nunique():,}"
)

print()
print("Top 20 phrases by sample count:")
print(
    phrase_summary
    .sort_values("samples", ascending=False)
    .head(20)
    .to_string(index=False)
)


print()
print("=" * 80)
print("BEST / WORST REFERENCE PHRASES")
print("=" * 80)

eligible = phrase_summary[
    phrase_summary["samples"] >= 5
].copy()

print()
print("Largest positive C2 gains:")
print(
    eligible
    .sort_values("mean_iou_gain", ascending=False)
    .head(15)
    .to_string(index=False)
)

print()
print("Largest negative C2 changes:")
print(
    eligible
    .sort_values("mean_iou_gain", ascending=True)
    .head(15)
    .to_string(index=False)
)


print()
print("=" * 80)
print("SUPERLATIVE VS ORDINARY")
print("=" * 80)

superlative = df[
    df["reference_type"].isin(
        ["superlative-largest", "superlative-smallest"]
    )
]

ordinary = df[
    df["reference_type"] == "ordinary-reference"
]

print(
    f"Superlative samples: {len(superlative):,}"
)

print(
    f"Ordinary reference samples: {len(ordinary):,}"
)

if len(superlative):
    print(
        f"Superlative C1 base IoU: "
        f"{superlative['c1_base_iou'].mean():.9f}"
    )
    print(
        f"Superlative C2 IoU:      "
        f"{superlative['c2_iou'].mean():.9f}"
    )
    print(
        f"Superlative gain:        "
        f"{superlative['iou_gain'].mean():+.9f}"
    )

if len(ordinary):
    print(
        f"Ordinary C1 base IoU:    "
        f"{ordinary['c1_base_iou'].mean():.9f}"
    )
    print(
        f"Ordinary C2 IoU:         "
        f"{ordinary['c2_iou'].mean():.9f}"
    )
    print(
        f"Ordinary gain:           "
        f"{ordinary['iou_gain'].mean():+.9f}"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Type summary:\n{SUMMARY_PATH}"
)

print(
    f"Phrase summary:\n{PHRASE_PATH}"
)