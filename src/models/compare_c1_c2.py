from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd


C1_PATH = r"D:\SatQueryAI\data\processed\c1_diagnostic\exact_c1_per_sample.csv"
C2_PATH = r"D:\SatQueryAI\data\processed\c2_diagnostic\exact_c2_per_sample.csv"

OUT_DIR = r"D:\SatQueryAI\data\processed\c1_c2_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

PAIRED_PATH = os.path.join(OUT_DIR, "paired_per_sample.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "comparison_summary.txt")


def require_columns(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} diagnostic is missing required columns:\n"
            + "\n".join(missing)
        )


def safe_mean(s):
    return float(pd.to_numeric(s, errors="coerce").mean())


def safe_median(s):
    return float(pd.to_numeric(s, errors="coerce").median())


def pct(x):
    return 100.0 * float(x)


def iou_from_boxes(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = a_area + b_area - inter

    if union <= 0:
        return 0.0

    return inter / union


def box_centroid_error(
    pred_x1,
    pred_y1,
    pred_x2,
    pred_y2,
    target_x1,
    target_y1,
    target_x2,
    target_y2,
):
    pred_cx = (pred_x1 + pred_x2) / 2.0
    pred_cy = (pred_y1 + pred_y2) / 2.0

    tgt_cx = (target_x1 + target_x2) / 2.0
    tgt_cy = (target_y1 + target_y2) / 2.0

    return math.sqrt(
        (pred_cx - tgt_cx) ** 2 +
        (pred_cy - tgt_cy) ** 2
    )


print("=" * 80)
print("C1 vs C2 — EXACT PAIRED GROUNDING COMPARISON")
print("=" * 80)
print("GPU: OFF")
print()

print("C1 diagnostic:")
print(C1_PATH)
print()

print("C2 diagnostic:")
print(C2_PATH)
print()

print("=" * 80)
print("VERIFYING INPUT FILES")
print("=" * 80)

if not os.path.isfile(C1_PATH):
    raise FileNotFoundError(C1_PATH)

if not os.path.isfile(C2_PATH):
    raise FileNotFoundError(C2_PATH)

print("C1 diagnostic: FOUND")
print("C2 diagnostic: FOUND")
print()

print("=" * 80)
print("LOADING DIAGNOSTICS")
print("=" * 80)

c1 = pd.read_csv(C1_PATH)
c2 = pd.read_csv(C2_PATH)

print(f"C1 rows: {len(c1):,}")
print(f"C2 rows: {len(c2):,}")
print()

c1_required = [
    "dataset_index",
    "row_index",
    "patch_id",
    "phrase",
    "target_x1",
    "target_y1",
    "target_x2",
    "target_y2",
    "target_area",
    "base_x1",
    "base_y1",
    "base_x2",
    "base_y2",
    "final_x1",
    "final_y1",
    "final_x2",
    "final_y2",
    "base_iou",
    "final_iou",
    "heatmap_peak_x",
    "heatmap_peak_y",
    "heatmap_peak_inside_gt",
    "heatmap_mass_inside_gt",
    "heatmap_entropy",
]

c2_required = [
    "dataset_index",
    "row_index",
    "patch_id",
    "phrase",
    "target_x1",
    "target_y1",
    "target_x2",
    "target_y2",
    "target_area",
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
    "iou",
    "heatmap_peak_x",
    "heatmap_peak_y",
    "heatmap_peak_inside_gt",
    "gt_mass",
    "heatmap_entropy",
]

require_columns(c1, "C1", c1_required)
require_columns(c2, "C2", c2_required)

print("Required C1 columns: OK")
print("Required C2 columns: OK")
print()

print("=" * 80)
print("VERIFYING PAIRED ALIGNMENT")
print("=" * 80)

if len(c1) != len(c2):
    raise RuntimeError(
        f"Row count mismatch: C1={len(c1)}, C2={len(c2)}"
    )

key_cols = ["row_index", "patch_id", "phrase"]

c1_keys = c1[key_cols].copy()
c2_keys = c2[key_cols].copy()

for col in key_cols:
    c1_keys[col] = c1_keys[col].astype(str)
    c2_keys[col] = c2_keys[col].astype(str)

if not c1_keys.reset_index(drop=True).equals(
    c2_keys.reset_index(drop=True)
):
    mismatch = c1_keys.ne(c2_keys).any(axis=1)
    print("First mismatched rows:")
    print(
        pd.concat(
            [
                c1_keys[mismatch].head(10).add_prefix("C1_"),
                c2_keys[mismatch].head(10).add_prefix("C2_"),
            ],
            axis=1,
        )
    )
    raise RuntimeError("C1/C2 row ordering or pairing does not match.")

print("row_index: IDENTICAL")
print("patch_id: IDENTICAL")
print("phrase: IDENTICAL")
print()

target_cols = [
    "target_x1",
    "target_y1",
    "target_x2",
    "target_y2",
    "target_area",
]

target_match = True

for col in target_cols:
    a = pd.to_numeric(c1[col], errors="coerce").to_numpy()
    b = pd.to_numeric(c2[col], errors="coerce").to_numpy()

    if not np.allclose(a, b, atol=1e-7, rtol=0.0, equal_nan=True):
        target_match = False
        diff = np.nanmax(np.abs(a - b))
        print(f"Target mismatch in {col}, max abs diff = {diff}")

if not target_match:
    raise RuntimeError("C1/C2 target geometry does not match.")

print("Target geometry: IDENTICAL")
print()

# -------------------------------------------------------------------------
# Normalize numeric fields
# -------------------------------------------------------------------------

numeric_c1 = [
    "target_x1", "target_y1", "target_x2", "target_y2",
    "target_area",
    "base_x1", "base_y1", "base_x2", "base_y2",
    "final_x1", "final_y1", "final_x2", "final_y2",
    "base_iou", "final_iou",
    "heatmap_peak_x", "heatmap_peak_y",
    "heatmap_peak_inside_gt",
    "heatmap_mass_inside_gt",
    "heatmap_entropy",
]

numeric_c2 = [
    "target_x1", "target_y1", "target_x2", "target_y2",
    "target_area",
    "pred_x1", "pred_y1", "pred_x2", "pred_y2",
    "iou",
    "heatmap_peak_x", "heatmap_peak_y",
    "heatmap_peak_inside_gt",
    "gt_mass",
    "heatmap_entropy",
]

for col in numeric_c1:
    c1[col] = pd.to_numeric(c1[col], errors="coerce")

for col in numeric_c2:
    c2[col] = pd.to_numeric(c2[col], errors="coerce")

# -------------------------------------------------------------------------
# Build paired dataframe
# -------------------------------------------------------------------------

paired = pd.DataFrame({
    "dataset_index": c1["dataset_index"].values,
    "row_index": c1["row_index"].values,
    "patch_id": c1["patch_id"].values,
    "phrase": c1["phrase"].values,

    "target_x1": c1["target_x1"].values,
    "target_y1": c1["target_y1"].values,
    "target_x2": c1["target_x2"].values,
    "target_y2": c1["target_y2"].values,
    "target_area": c1["target_area"].values,

    # C1
    "c1_base_x1": c1["base_x1"].values,
    "c1_base_y1": c1["base_y1"].values,
    "c1_base_x2": c1["base_x2"].values,
    "c1_base_y2": c1["base_y2"].values,
    "c1_final_x1": c1["final_x1"].values,
    "c1_final_y1": c1["final_y1"].values,
    "c1_final_x2": c1["final_x2"].values,
    "c1_final_y2": c1["final_y2"].values,

    "c1_base_iou": c1["base_iou"].values,
    "c1_final_iou": c1["final_iou"].values,

    "c1_peak_x": c1["heatmap_peak_x"].values,
    "c1_peak_y": c1["heatmap_peak_y"].values,
    "c1_peak_inside_gt": c1["heatmap_peak_inside_gt"].values,
    "c1_mass_inside_gt": c1["heatmap_mass_inside_gt"].values,
    "c1_entropy": c1["heatmap_entropy"].values,

    # C2
    "c2_pred_x1": c2["pred_x1"].values,
    "c2_pred_y1": c2["pred_y1"].values,
    "c2_pred_x2": c2["pred_x2"].values,
    "c2_pred_y2": c2["pred_y2"].values,
    "c2_iou": c2["iou"].values,

    "c2_peak_x": c2["heatmap_peak_x"].values,
    "c2_peak_y": c2["heatmap_peak_y"].values,
    "c2_peak_inside_gt": c2["heatmap_peak_inside_gt"].values,
    "c2_mass_inside_gt": c2["gt_mass"].values,
    "c2_entropy": c2["heatmap_entropy"].values,
})

# -------------------------------------------------------------------------
# Derived metrics
# -------------------------------------------------------------------------

paired["c1_correction_iou_change"] = (
    paired["c1_final_iou"] - paired["c1_base_iou"]
)

paired["c2_minus_c1_base_iou"] = (
    paired["c2_iou"] - paired["c1_base_iou"]
)

paired["c2_minus_c1_final_iou"] = (
    paired["c2_iou"] - paired["c1_final_iou"]
)

paired["c1_base_center_error"] = paired.apply(
    lambda r: box_centroid_error(
        r.c1_base_x1, r.c1_base_y1,
        r.c1_base_x2, r.c1_base_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c1_final_center_error"] = paired.apply(
    lambda r: box_centroid_error(
        r.c1_final_x1, r.c1_final_y1,
        r.c1_final_x2, r.c1_final_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c2_center_error"] = paired.apply(
    lambda r: box_centroid_error(
        r.c2_pred_x1, r.c2_pred_y1,
        r.c2_pred_x2, r.c2_pred_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c2_minus_c1_base_center_error"] = (
    paired["c2_center_error"] -
    paired["c1_base_center_error"]
)

paired["c2_minus_c1_final_center_error"] = (
    paired["c2_center_error"] -
    paired["c1_final_center_error"]
)

def area_ratio(px1, py1, px2, py2, tx1, ty1, tx2, ty2):
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    tgt_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)

    if tgt_area <= 0:
        return np.nan

    return pred_area / tgt_area


paired["c1_base_area_ratio"] = paired.apply(
    lambda r: area_ratio(
        r.c1_base_x1, r.c1_base_y1,
        r.c1_base_x2, r.c1_base_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c1_final_area_ratio"] = paired.apply(
    lambda r: area_ratio(
        r.c1_final_x1, r.c1_final_y1,
        r.c1_final_x2, r.c1_final_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c2_area_ratio"] = paired.apply(
    lambda r: area_ratio(
        r.c2_pred_x1, r.c2_pred_y1,
        r.c2_pred_x2, r.c2_pred_y2,
        r.target_x1, r.target_y1,
        r.target_x2, r.target_y2,
    ),
    axis=1,
)

paired["c2_minus_c1_base_mass_inside"] = (
    paired["c2_mass_inside_gt"] -
    paired["c1_mass_inside_gt"]
)

paired["c2_minus_c1_base_entropy"] = (
    paired["c2_entropy"] -
    paired["c1_entropy"]
)

paired["c2_minus_c1_base_peak_inside"] = (
    paired["c2_peak_inside_gt"] -
    paired["c1_peak_inside_gt"]
)

# -------------------------------------------------------------------------
# Target-size bins
# -------------------------------------------------------------------------

try:
    paired["size_bin"] = pd.qcut(
        paired["target_area"],
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
except ValueError:
    paired["size_bin"] = "unavailable"

# -------------------------------------------------------------------------
# Improved / worse / unchanged
# -------------------------------------------------------------------------

EPS = 1e-9

paired["outcome_vs_base"] = np.where(
    paired["c2_minus_c1_base_iou"] > EPS,
    "improved",
    np.where(
        paired["c2_minus_c1_base_iou"] < -EPS,
        "worse",
        "unchanged",
    ),
)

paired["outcome_vs_final"] = np.where(
    paired["c2_minus_c1_final_iou"] > EPS,
    "improved",
    np.where(
        paired["c2_minus_c1_final_iou"] < -EPS,
        "worse",
        "unchanged",
    ),
)

# -------------------------------------------------------------------------
# Save paired data
# -------------------------------------------------------------------------

paired.to_csv(PAIRED_PATH, index=False)

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

lines = []

lines.append("=" * 80)
lines.append("C1 vs C2 — EXACT PAIRED GROUNDING COMPARISON")
lines.append("=" * 80)
lines.append("")

lines.append("DATA")
lines.append(f"Rows: {len(paired):,}")
lines.append("Pairing verified: YES")
lines.append("Target geometry verified: YES")
lines.append("")

lines.append("PRIMARY COMPARISON: C1 BASE vs C2")
lines.append("-" * 80)

c1_base_iou = paired["c1_base_iou"]
c1_final_iou = paired["c1_final_iou"]
c2_iou = paired["c2_iou"]

lines.append(f"C1 base mean IoU : {safe_mean(c1_base_iou):.9f}")
lines.append(f"C1 base median IoU: {safe_median(c1_base_iou):.9f}")
lines.append(f"C1 final mean IoU: {safe_mean(c1_final_iou):.9f}")
lines.append(f"C1 final median IoU: {safe_median(c1_final_iou):.9f}")
lines.append(f"C2 mean IoU      : {safe_mean(c2_iou):.9f}")
lines.append(f"C2 median IoU    : {safe_median(c2_iou):.9f}")
lines.append("")

gain_base = paired["c2_minus_c1_base_iou"]

lines.append(
    f"Mean IoU gain C2 - C1 base : {safe_mean(gain_base):+.9f}"
)
lines.append(
    f"Median IoU gain             : {safe_median(gain_base):+.9f}"
)
lines.append(
    f"Improved vs C1 base         : {pct((gain_base > EPS).mean()):.2f}%"
)
lines.append(
    f"Worse vs C1 base            : {pct((gain_base < -EPS).mean()):.2f}%"
)
lines.append(
    f"Unchanged                   : {pct((gain_base.abs() <= EPS).mean()):.2f}%"
)
lines.append("")

lines.append("SECONDARY COMPARISON: C1 FINAL vs C2")
lines.append("-" * 80)

gain_final = paired["c2_minus_c1_final_iou"]

lines.append(
    f"Mean IoU gain C2 - C1 final: {safe_mean(gain_final):+.9f}"
)
lines.append(
    f"Median IoU gain            : {safe_median(gain_final):+.9f}"
)
lines.append(
    f"Improved                   : {pct((gain_final > EPS).mean()):.2f}%"
)
lines.append(
    f"Worse                      : {pct((gain_final < -EPS).mean()):.2f}%"
)
lines.append("")

lines.append("C1 CORRECTION HEAD EFFECT")
lines.append("-" * 80)

corr = paired["c1_correction_iou_change"]

lines.append(
    f"Mean correction IoU change : {safe_mean(corr):+.9f}"
)
lines.append(
    f"Median correction change   : {safe_median(corr):+.9f}"
)
lines.append(
    f"Improved after correction  : {pct((corr > EPS).mean()):.2f}%"
)
lines.append(
    f"Hurt after correction      : {pct((corr < -EPS).mean()):.2f}%"
)
lines.append("")

lines.append("DERIVED BOX-CENTER ERROR")
lines.append("-" * 80)

lines.append(
    f"C1 base mean center error : "
    f"{safe_mean(paired['c1_base_center_error']):.9f}"
)
lines.append(
    f"C1 final mean center error: "
    f"{safe_mean(paired['c1_final_center_error']):.9f}"
)
lines.append(
    f"C2 mean center error      : "
    f"{safe_mean(paired['c2_center_error']):.9f}"
)
lines.append(
    f"C2 - C1 base center error : "
    f"{safe_mean(paired['c2_minus_c1_base_center_error']):+.9f}"
)
lines.append(
    f"C2 - C1 final center error: "
    f"{safe_mean(paired['c2_minus_c1_final_center_error']):+.9f}"
)
lines.append(
    "NOTE: box-center error is derived from stored bounding boxes; "
    "it is not C2's heatmap moment centroid error."
)
lines.append("")

lines.append("HEATMAP METRICS")
lines.append("-" * 80)

lines.append(
    f"C1 mass inside GT mean: "
    f"{safe_mean(paired['c1_mass_inside_gt']):.9f}"
)
lines.append(
    f"C2 mass inside GT mean: "
    f"{safe_mean(paired['c2_mass_inside_gt']):.9f}"
)
lines.append(
    f"C2 - C1 mass change  : "
    f"{safe_mean(paired['c2_minus_c1_base_mass_inside']):+.9f}"
)
lines.append("")

lines.append(
    f"C1 peak-inside rate: "
    f"{pct(paired['c1_peak_inside_gt'].mean()):.2f}%"
)
lines.append(
    f"C2 peak-inside rate: "
    f"{pct(paired['c2_peak_inside_gt'].mean()):.2f}%"
)
lines.append(
    f"Peak-inside change : "
    f"{pct((paired['c2_peak_inside_gt'] > paired['c1_peak_inside_gt']).mean()):.2f}% "
    "of rows have C2=inside when C1 was not, "
    "before considering reverse changes."
)
lines.append("")

lines.append(
    f"C1 entropy mean: {safe_mean(paired['c1_entropy']):.9f}"
)
lines.append(
    f"C2 entropy mean: {safe_mean(paired['c2_entropy']):.9f}"
)
lines.append(
    f"C2 - C1 entropy change: "
    f"{safe_mean(paired['c2_minus_c1_base_entropy']):+.9f}"
)
lines.append("")

lines.append("TARGET-SIZE ANALYSIS")
lines.append("-" * 80)
lines.append(
    "Six equal-frequency target-area bins: "
    "tiny / small / medium / large / very_large / huge."
)
lines.append(
    "These are diagnostic quantile bins, not the historical C0 bin boundaries."
)
lines.append("")

size_summary = (
    paired
    .groupby("size_bin", observed=False)
    .agg(
        samples=("row_index", "count"),
        target_area_mean=("target_area", "mean"),
        c1_base_iou=("c1_base_iou", "mean"),
        c2_iou=("c2_iou", "mean"),
        iou_gain=("c2_minus_c1_base_iou", "mean"),
        c1_center_error=("c1_base_center_error", "mean"),
        c2_center_error=("c2_center_error", "mean"),
        c1_mass_inside=("c1_mass_inside_gt", "mean"),
        c2_mass_inside=("c2_mass_inside_gt", "mean"),
        c1_entropy=("c1_entropy", "mean"),
        c2_entropy=("c2_entropy", "mean"),
    )
    .reset_index()
)

for _, r in size_summary.iterrows():
    lines.append(
        f"{str(r['size_bin']):>10} | "
        f"N={int(r['samples']):4d} | "
        f"C1={r['c1_base_iou']:.6f} | "
        f"C2={r['c2_iou']:.6f} | "
        f"gain={r['iou_gain']:+.6f} | "
        f"center C1={r['c1_center_error']:.6f} | "
        f"C2={r['c2_center_error']:.6f}"
    )

lines.append("")
lines.append("DIFFICULTY QUARTILES")
lines.append("-" * 80)

# Define difficulty from C1 base IoU:
paired["c1_difficulty_quartile"] = pd.qcut(
    paired["c1_base_iou"],
    q=4,
    labels=["hardest", "hard", "easy", "easiest"],
    duplicates="drop",
)

difficulty_summary = (
    paired
    .groupby("c1_difficulty_quartile", observed=False)
    .agg(
        samples=("row_index", "count"),
        c1_base_iou=("c1_base_iou", "mean"),
        c2_iou=("c2_iou", "mean"),
        gain=("c2_minus_c1_base_iou", "mean"),
    )
    .reset_index()
)

for _, r in difficulty_summary.iterrows():
    lines.append(
        f"{str(r['c1_difficulty_quartile']):>8} | "
        f"N={int(r['samples']):4d} | "
        f"C1={r['c1_base_iou']:.6f} | "
        f"C2={r['c2_iou']:.6f} | "
        f"gain={r['gain']:+.6f}"
    )

lines.append("")

lines.append("HARD-SAMPLE OVERLAP")
lines.append("-" * 80)

threshold = paired["c1_base_iou"].quantile(0.25)

hard_c1 = set(
    paired.loc[
        paired["c1_base_iou"] <= threshold,
        "row_index"
    ].tolist()
)

hard_c2_threshold = paired["c2_iou"].quantile(0.25)

hard_c2 = set(
    paired.loc[
        paired["c2_iou"] <= hard_c2_threshold,
        "row_index"
    ].tolist()
)

intersection = hard_c1 & hard_c2
union = hard_c1 | hard_c2

lines.append(f"C1 bottom 25% threshold: {threshold:.9f}")
lines.append(f"C2 bottom 25% threshold: {hard_c2_threshold:.9f}")
lines.append(f"C1 hard count: {len(hard_c1):,}")
lines.append(f"C2 hard count: {len(hard_c2):,}")
lines.append(f"Overlap count: {len(intersection):,}")

if union:
    jaccard = len(intersection) / len(union)
else:
    jaccard = 0.0

lines.append(f"Hard-set Jaccard: {jaccard:.6f}")
lines.append("")

lines.append("TOP 20 C2 GAINS OVER C1 BASE")
lines.append("-" * 80)

top_gain = paired.nlargest(20, "c2_minus_c1_base_iou")

for _, r in top_gain.iterrows():
    lines.append(
        f"row={r['row_index']} | "
        f"patch={r['patch_id']} | "
        f"C1={r['c1_base_iou']:.6f} | "
        f"C2={r['c2_iou']:.6f} | "
        f"gain={r['c2_minus_c1_base_iou']:+.6f} | "
        f"area={r['target_area']:.6f}"
    )

lines.append("")
lines.append("BOTTOM 20 C2 LOSSES VS C1 BASE")
lines.append("-" * 80)

top_loss = paired.nsmallest(20, "c2_minus_c1_base_iou")

for _, r in top_loss.iterrows():
    lines.append(
        f"row={r['row_index']} | "
        f"patch={r['patch_id']} | "
        f"C1={r['c1_base_iou']:.6f} | "
        f"C2={r['c2_iou']:.6f} | "
        f"change={r['c2_minus_c1_base_iou']:+.6f} | "
        f"area={r['target_area']:.6f}"
    )

lines.append("")

# -------------------------------------------------------------------------
# Correlations
# -------------------------------------------------------------------------

lines.append("CORRELATIONS WITH C2 IoU GAIN")
lines.append("-" * 80)

corr_candidates = [
    "target_area",
    "c1_base_iou",
    "c1_base_center_error",
    "c1_base_area_ratio",
    "c1_base_center_error",
    "c1_mass_inside_gt",
    "c1_entropy",
    "c1_peak_inside_gt",
]

for col in corr_candidates:
    x = paired[col].astype(float)
    y = paired["c2_minus_c1_base_iou"].astype(float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() >= 3:
        value = np.corrcoef(x[valid], y[valid])[0, 1]
        lines.append(f"{col:35s}: {value:+.6f}")

lines.append("")
lines.append("=" * 80)
lines.append("FILES")
lines.append("=" * 80)
lines.append(f"Per-sample paired CSV: {PAIRED_PATH}")
lines.append(f"Summary report        : {SUMMARY_PATH}")
lines.append("")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))

print()
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved paired CSV: {PAIRED_PATH}")
print(f"Saved summary   : {SUMMARY_PATH}")