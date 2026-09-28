"""
SATQUERYAI — C0 SUPERLATIVE VS NON-SUPERLATIVE DIAGNOSTIC

Purpose
-------
Evaluate the existing patch-disjoint reference-grounding checkpoint
and determine whether queries containing superlative/comparative terms
such as "largest" and "smallest" are substantially harder than
non-superlative category queries.

NO TRAINING IS PERFORMED.

Uses the exact project components:
    - DofaShardCache
    - GroundingTextCache
    - ReferenceGroundingDataset
    - collate_reference_batch
    - GroundingHeatmapModel

Uses the exact local checkpoint:
    models/checkpoints/reference_grounding_large/best.pt

Uses the exact local validation split:
    data/processed/reference_grounding_split/reference_val.parquet

Uses the same 10%-90% quantile box extraction used by the trainer.
"""

import sys
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ======================================================================
# PROJECT SETUP
# ======================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import the exact classes/functions from the existing training code.
from src.models.train_reference_grounding import (
    DofaShardCache,
    GroundingTextCache,
    GroundingHeatmapModel,
    ReferenceGroundingDataset,
    collate_reference_batch,
)


# ======================================================================
# PATHS
# ======================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "reference_grounding_large"
    / "best.pt"
)

VAL_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

DOFA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

TEXT_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grounding_text_embeddings"
    / "grounding_text_embedding_shards"
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

CSV_OUT = (
    OUTPUT_DIR
    / "validation_predictions_c0.csv"
)

SUMMARY_OUT = (
    OUTPUT_DIR
    / "superlative_summary.json"
)

QUERY_SUMMARY_OUT = (
    OUTPUT_DIR
    / "query_level_summary.csv"
)


# ======================================================================
# CONFIG
# ======================================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 32

QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90


# Terms used to identify comparative/superlative queries.
#
# We deliberately keep this conservative. The main terms of interest
# from the current reference-grounding problem are:
#     largest / smallest / biggest / tiniest
#
# "most"/"least"/"highest"/"lowest" are included because they can also
# encode comparison, but their exact semantic use will be visible in
# the saved CSV.
SUPERLATIVE_PATTERNS = [
    r"\blargest\b",
    r"\bsmallest\b",
    r"\bbiggest\b",
    r"\btiniest\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bmost\b",
    r"\bleast\b",
]


# ======================================================================
# HELPERS
# ======================================================================

def is_superlative_query(text):
    """
    Determine whether a query contains a superlative/comparative term.
    """
    if text is None:
        return False

    text = str(text).lower()

    for pattern in SUPERLATIVE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def box_iou(box_a, box_b):
    """
    IoU for [x1, y1, x2, y2] normalized boxes.
    """

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def weighted_quantile(values, weights, q):
    """
    Weighted quantile.

    values  : 1-D numerical array
    weights : non-negative weights corresponding to values
    q       : quantile in [0, 1]

    This reproduces the distribution-based quantile idea used by the
    grounding trainer without requiring an additional GPU operation.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            "values and weights must both be 1-D"
        )

    if len(values) != len(weights):
        raise ValueError(
            "values and weights must have equal length"
        )

    if len(values) == 0:
        raise ValueError(
            "Cannot compute quantile of empty distribution"
        )

    q = float(q)

    if not 0.0 <= q <= 1.0:
        raise ValueError(
            f"Quantile must be in [0,1], got {q}"
        )

    # Remove invalid values safely.
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )

    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        raise ValueError(
            "No valid weighted values"
        )

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    total_weight = weights.sum()

    if total_weight <= 0.0:
        # Degenerate case: use ordinary quantile.
        return float(
            np.quantile(values, q)
        )

    cumulative = np.cumsum(weights)
    target = q * total_weight

    idx = np.searchsorted(
        cumulative,
        target,
        side="left",
    )

    idx = min(
        max(idx, 0),
        len(values) - 1,
    )

    return float(values[idx])


def heatmap_to_probability(heatmap):
    """
    Convert a model heatmap-like output into a normalized spatial
    probability distribution.

    Handles either:
        - probability heatmaps
        - logits

    Returns:
        [B, H, W]
    """

    if heatmap.ndim != 3:
        raise ValueError(
            f"Expected [B,H,W], got {tuple(heatmap.shape)}"
        )

    # Determine whether this already looks like a probability map.
    flat = heatmap.reshape(
        heatmap.shape[0],
        -1,
    )

    sums = flat.sum(dim=1)

    nonnegative = bool(
        torch.all(
            heatmap >= -1e-6
        ).item()
    )

    approximately_one = bool(
        torch.all(
            torch.abs(sums - 1.0) < 1e-3
        ).item()
    )

    if nonnegative and approximately_one:
        probs = heatmap

    else:
        probs = torch.softmax(
            flat,
            dim=1,
        ).reshape_as(heatmap)

    return probs


def extract_quantile_box(probability_map):
    """
    Convert a [H,W] probability map into a normalized bounding box
    using the project's 10%-90% distribution quantile method.
    """

    h, w = probability_map.shape

    probability_map = (
        probability_map
        .detach()
        .cpu()
        .numpy()
    )

    # Cell-center coordinates normalized to [0,1].
    xs = (
        np.arange(w, dtype=np.float64)
        + 0.5
    ) / w

    ys = (
        np.arange(h, dtype=np.float64)
        + 0.5
    ) / h

    grid_x, grid_y = np.meshgrid(
        xs,
        ys,
        indexing="xy",
    )

    x_values = grid_x.reshape(-1)
    y_values = grid_y.reshape(-1)
    weights = probability_map.reshape(-1)

    x1 = weighted_quantile(
        x_values,
        weights,
        QUANTILE_LOW,
    )

    x2 = weighted_quantile(
        x_values,
        weights,
        QUANTILE_HIGH,
    )

    y1 = weighted_quantile(
        y_values,
        weights,
        QUANTILE_LOW,
    )

    y2 = weighted_quantile(
        y_values,
        weights,
        QUANTILE_HIGH,
    )

    # Clamp to valid image coordinates.
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))

    # Guarantee ordering.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def summarize_dataframe(df, group_name):
    """
    Produce the main C0 statistics.
    """

    if len(df) == 0:
        return {
            "group": group_name,
            "count": 0,
            "mean_iou": None,
            "median_iou": None,
            "iou_ge_025_pct": None,
            "iou_ge_050_pct": None,
            "iou_ge_075_pct": None,
        }

    ious = df["iou"].astype(float)

    return {
        "group": group_name,
        "count": int(len(df)),
        "mean_iou": float(
            ious.mean()
        ),
        "median_iou": float(
            ious.median()
        ),
        "iou_ge_025_pct": float(
            100.0 * (ious >= 0.25).mean()
        ),
        "iou_ge_050_pct": float(
            100.0 * (ious >= 0.50).mean()
        ),
        "iou_ge_075_pct": float(
            100.0 * (ious >= 0.75).mean()
        ),
    }


# ======================================================================
# START
# ======================================================================

print("=" * 70)
print("SATQUERYAI — C0 SUPERLATIVE DIAGNOSTIC")
print("=" * 70)

print(
    f"Device              : {DEVICE}"
)

print(
    f"Checkpoint          : {CHECKPOINT}"
)

print(
    f"Validation parquet  : {VAL_PARQUET}"
)

print(
    f"DOFA root            : {DOFA_ROOT}"
)

print(
    f"Text cache           : {TEXT_CACHE_DIR}"
)


# ======================================================================
# CHECK PATHS
# ======================================================================

print()
print("=" * 70)
print("CHECKING PATHS")
print("=" * 70)

required_paths = [
    CHECKPOINT,
    VAL_PARQUET,
    DOFA_ROOT,
    TEXT_CACHE_DIR,
]

for path in required_paths:

    exists = path.exists()

    print(
        f"{'OK' if exists else 'MISSING'} : {path}"
    )

    if not exists:
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )


# ======================================================================
# LOAD CHECKPOINT
# ======================================================================

print()
print("=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

print(
    f"Checkpoint epoch      : "
    f"{checkpoint.get('epoch')}"
)

print(
    f"Checkpoint val IoU    : "
    f"{checkpoint.get('val_mean_iou')}"
)

print(
    "Checkpoint config:"
)

print(
    json.dumps(
        checkpoint.get("config", {}),
        indent=2,
    )
)


# ======================================================================
# LOAD DOFA CACHE
# ======================================================================

print()
print("=" * 70)
print("LOADING DOFA CACHE")
print("=" * 70)

dofa_cache = DofaShardCache(
    str(DOFA_ROOT)
)

print(
    "DOFA cache loaded."
)


# ======================================================================
# LOAD TEXT CACHE
# ======================================================================

print()
print("=" * 70)
print("LOADING TEXT CACHE")
print("=" * 70)

text_cache = GroundingTextCache(
    str(TEXT_CACHE_DIR)
)

print(
    f"Indexed text embeddings: "
    f"{len(text_cache.index):,}"
)


# ======================================================================
# LOAD DATASET
# ======================================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATASET")
print("=" * 70)

dataset = ReferenceGroundingDataset(
    str(VAL_PARQUET),
    dofa_cache,
    text_cache,
)

print(
    f"Validation rows: "
    f"{len(dataset):,}"
)


# ======================================================================
# DATA LOADER
# ======================================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_reference_batch,
)

print(
    f"Validation batches: "
    f"{len(loader):,}"
)


# ======================================================================
# MODEL
# ======================================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = GroundingHeatmapModel()

missing_keys, unexpected_keys = (
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
)

if missing_keys:

    print(
        "WARNING — missing model keys:"
    )

    for key in missing_keys:
        print(
            f"  {key}"
        )

if unexpected_keys:

    print(
        "WARNING — unexpected model keys:"
    )

    for key in unexpected_keys:
        print(
            f"  {key}"
        )

model.to(DEVICE)
model.eval()

print(
    "Model loaded successfully."
)


# ======================================================================
# VALIDATION
# ======================================================================

print()
print("=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

results = []

with torch.no_grad():

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        # --------------------------------------------------------------
        # First-batch diagnostic
        # --------------------------------------------------------------

        if batch_idx == 1:

            print()
            print("Batch fields:")

            for key, value in batch.items():

                if torch.is_tensor(value):

                    print(
                        f"  {key}: "
                        f"Tensor "
                        f"{tuple(value.shape)}"
                    )

                else:

                    print(
                        f"  {key}: "
                        f"{type(value).__name__}"
                    )

        # --------------------------------------------------------------
        # Exact model interface confirmed from the project:
        #
        #     model(s1, s2, text_embedding)
        # --------------------------------------------------------------

        s1 = batch["s1"].to(DEVICE)

        s2 = batch["s2"].to(DEVICE)

        text_embedding = (
            batch["text_embedding"]
            .to(DEVICE)
        )

        outputs = model(
            s1,
            s2,
            text_embedding,
        )

        # --------------------------------------------------------------
        # First-batch output diagnostic
        # --------------------------------------------------------------

        if batch_idx == 1:

            print()
            print("Model output:")

            if isinstance(outputs, dict):

                for key, value in outputs.items():

                    if torch.is_tensor(value):

                        print(
                            f"  {key}: "
                            f"Tensor "
                            f"{tuple(value.shape)}"
                        )

                    else:

                        print(
                            f"  {key}: "
                            f"{type(value).__name__}"
                        )

            else:

                print(
                    f"Type: "
                    f"{type(outputs)}"
                )

        # --------------------------------------------------------------
        # Extract heatmap
        # --------------------------------------------------------------

        if not isinstance(outputs, dict):

            raise RuntimeError(
                "GroundingHeatmapModel did not return "
                "a dictionary."
            )

        if "heatmap" in outputs:

            heatmap = outputs["heatmap"]

        elif "heatmap_logits" in outputs:

            heatmap = outputs["heatmap_logits"]

        else:

            raise KeyError(
                "Could not find 'heatmap' or "
                "'heatmap_logits' in model outputs. "
                f"Available keys: {list(outputs.keys())}"
            )

        # --------------------------------------------------------------
        # Convert heatmap into spatial probability map
        # --------------------------------------------------------------

        probabilities = (
            heatmap_to_probability(
                heatmap
            )
        )

        # --------------------------------------------------------------
        # Ground-truth boxes
        # --------------------------------------------------------------

        target_boxes = (
            batch["target_box"]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------------
        # Exact known metadata fields
        # --------------------------------------------------------------

        patch_ids = [
            str(x)
            for x in batch["patch_id"]
        ]

        row_indices = [
            int(x)
            for x in batch["row_index"]
        ]

        categories = [
            str(x)
            for x in batch["category"]
        ]

        phrases = [
            str(x)
            for x in batch["phrase"]
        ]

        # --------------------------------------------------------------
        # Process examples in batch
        # --------------------------------------------------------------

        batch_n = probabilities.shape[0]

        if batch_n != len(phrases):

            raise RuntimeError(
                "Batch size mismatch:\n"
                f"heatmap batch = {batch_n}\n"
                f"phrases = {len(phrases)}"
            )

        for i in range(batch_n):

            probability_map = probabilities[i]

            predicted_box = (
                extract_quantile_box(
                    probability_map
                )
            )

            target_box = [
                float(target_boxes[i][0]),
                float(target_boxes[i][1]),
                float(target_boxes[i][2]),
                float(target_boxes[i][3]),
            ]

            iou = box_iou(
                predicted_box,
                target_box,
            )

            phrase = phrases[i]

            superlative = (
                is_superlative_query(
                    phrase
                )
            )

            results.append(
                {
                    "validation_position":
                        len(results),

                    "original_row_index":
                        row_indices[i],

                    "patch_id":
                        patch_ids[i],

                    "category":
                        categories[i],

                    "phrase":
                        phrase,

                    "superlative":
                        bool(superlative),

                    "pred_x1":
                        predicted_box[0],

                    "pred_y1":
                        predicted_box[1],

                    "pred_x2":
                        predicted_box[2],

                    "pred_y2":
                        predicted_box[3],

                    "target_x1":
                        target_box[0],

                    "target_y1":
                        target_box[1],

                    "target_x2":
                        target_box[2],

                    "target_y2":
                        target_box[3],

                    "iou":
                        float(iou),
                }
            )

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        if (
            batch_idx % 10 == 0
            or batch_idx == len(loader)
        ):

            print(
                f"Batch "
                f"{batch_idx:,}/"
                f"{len(loader):,}"
            )


# ======================================================================
# RESULTS DATAFRAME
# ======================================================================

results_df = pd.DataFrame(
    results
)

print()
print(
    f"Rows evaluated: "
    f"{len(results_df):,}"
)

if len(results_df) != len(dataset):

    raise RuntimeError(
        "Validation row count mismatch:\n"
        f"Dataset = {len(dataset):,}\n"
        f"Results = {len(results_df):,}"
    )


# ======================================================================
# SAVE ALL PER-EXAMPLE RESULTS
# ======================================================================

results_df.to_csv(
    CSV_OUT,
    index=False,
)

print()
print(
    "Saved detailed validation results:"
)

print(
    CSV_OUT
)


# ======================================================================
# GROUPS
# ======================================================================

superlative_df = results_df[
    results_df["superlative"] == True
].copy()

non_superlative_df = results_df[
    results_df["superlative"] == False
].copy()


# ======================================================================
# MAIN SUMMARIES
# ======================================================================

overall_summary = summarize_dataframe(
    results_df,
    "overall",
)

superlative_summary = summarize_dataframe(
    superlative_df,
    "superlative",
)

non_superlative_summary = summarize_dataframe(
    non_superlative_df,
    "non_superlative",
)


# ======================================================================
# DIFFERENCE
# ======================================================================

if (
    superlative_summary["mean_iou"] is not None
    and non_superlative_summary["mean_iou"] is not None
):

    mean_iou_difference = (
        superlative_summary["mean_iou"]
        -
        non_superlative_summary["mean_iou"]
    )

else:

    mean_iou_difference = None


# ======================================================================
# QUERY-LEVEL ANALYSIS
# ======================================================================

query_groups = (
    results_df
    .groupby(
        ["phrase", "superlative"],
        dropna=False,
    )
    .agg(
        count=("iou", "size"),
        mean_iou=("iou", "mean"),
        median_iou=("iou", "median"),
    )
    .reset_index()
    .sort_values(
        ["superlative", "mean_iou"]
    )
)

query_groups.to_csv(
    QUERY_SUMMARY_OUT,
    index=False,
)


# ======================================================================
# FULL SUMMARY
# ======================================================================

summary = {
    "checkpoint":
        str(CHECKPOINT),

    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "checkpoint_recorded_val_iou":
        checkpoint.get("val_mean_iou"),

    "validation_rows":
        int(len(results_df)),

    "validation_unique_patches":
        int(results_df["patch_id"].nunique()),

    "quantile_low":
        QUANTILE_LOW,

    "quantile_high":
        QUANTILE_HIGH,

    "overall":
        overall_summary,

    "superlative":
        superlative_summary,

    "non_superlative":
        non_superlative_summary,

    "superlative_minus_non_superlative_mean_iou":
        mean_iou_difference,

    "superlative_terms_used":
        SUPERLATIVE_PATTERNS,
}


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ======================================================================
# PRINT MAIN RESULTS
# ======================================================================

print()
print("=" * 70)
print("C0 RESULTS")
print("=" * 70)

print()
print("OVERALL")
print(
    json.dumps(
        overall_summary,
        indent=2,
    )
)

print()
print("SUPERLATIVE")
print(
    json.dumps(
        superlative_summary,
        indent=2,
    )
)

print()
print("NON-SUPERLATIVE")
print(
    json.dumps(
        non_superlative_summary,
        indent=2,
    )
)

print()
print(
    "SUPERLATIVE - NON-SUPERLATIVE "
    "MEAN IoU DIFFERENCE:"
)

if mean_iou_difference is None:

    print("N/A")

else:

    print(
        f"{mean_iou_difference:.6f}"
    )


# ======================================================================
# SUPERLATIVE QUERY COUNTS
# ======================================================================

print()
print("=" * 70)
print("SUPERLATIVE QUERY COUNTS")
print("=" * 70)

if len(superlative_df) == 0:

    print(
        "No superlative queries detected."
    )

else:

    print(
        superlative_df["phrase"]
        .value_counts()
        .head(50)
        .to_string()
    )


# ======================================================================
# LOWEST-PERFORMING SUPERLATIVE QUERIES
# ======================================================================

print()
print("=" * 70)
print("LOWEST-PERFORMING SUPERLATIVE QUERIES")
print("=" * 70)

if len(superlative_df) > 0:

    print(
        query_groups[
            query_groups["superlative"] == True
        ]
        .sort_values("mean_iou")
        .head(20)
        .to_string(index=False)
    )

else:

    print("None.")


# ======================================================================
# HIGHEST-PERFORMING SUPERLATIVE QUERIES
# ======================================================================

print()
print("=" * 70)
print("HIGHEST-PERFORMING SUPERLATIVE QUERIES")
print("=" * 70)

if len(superlative_df) > 0:

    print(
        query_groups[
            query_groups["superlative"] == True
        ]
        .sort_values(
            "mean_iou",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

else:

    print("None.")


# ======================================================================
# FILES
# ======================================================================

print()
print("=" * 70)
print("C0 COMPLETE")
print("=" * 70)

print()
print(
    f"Detailed predictions:"
)

print(
    CSV_OUT
)

print()
print(
    f"Summary:"
)

print(
    SUMMARY_OUT
)

print()
print(
    f"Query-level summary:"
)

print(
    QUERY_SUMMARY_OUT
)

print()
print(
    "No training was performed."
)