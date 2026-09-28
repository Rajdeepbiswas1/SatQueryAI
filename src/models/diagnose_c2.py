import os
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from src.models.train_reference_grounding import (
    DofaShardCache,
    GroundingTextCache,
    ReferenceGroundingDataset,
    collate_reference_batch,
)


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    r"D:\SatQueryAI\models\checkpoints"
    r"\c2_heatmap_only"
    r"\best.pt"
)

DOFA_ROOT = (
    r"D:\SatQueryAI\data\processed"
    r"\dofa_features_download"
    r"\dofa_features"
)

TEXT_CACHE_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\grounding_text_embeddings"
    r"\grounding_text_embedding_shards"
)

VAL_PARQUET = (
    r"D:\SatQueryAI\data\processed"
    r"\reference_grounding_split"
    r"\reference_val.parquet"
)

OUTPUT_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\c2_diagnostic"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "exact_c2_per_sample.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "c2_summary.json"
)

BATCH_SIZE = 32
NUM_WORKERS = 0
GRID_SIZE = 56

DEVICE = torch.device("cpu")

TOP_MASS_LEVELS = [0.10, 0.25, 0.50, 0.75]


# ============================================================
# EXACT C2 MODEL
# Copied from verify_c2.py without changing the architecture.
# ============================================================

class C2HeatmapModel(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        self.highres_grid_size = 56

        self.visual_fusion = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.spatial_pos = nn.Parameter(
            torch.zeros(
                1,
                grid_size * grid_size,
                hidden_dim
            )
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02
        )

        self.highres1 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.GroupNorm(
                32,
                hidden_dim
            )
        )

        self.highres2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.GroupNorm(
                32,
                hidden_dim
            )
        )

        self.query_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.query_match = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.match_norm = nn.LayerNorm(
            hidden_dim
        )

        self.match_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        s1,
        s2,
        text_embedding,
    ):

        B = s1.size(0)

        visual = torch.cat(
            [
                s1,
                s2,
            ],
            dim=-1
        )

        visual = self.visual_fusion(
            visual
        )

        visual = (
            visual
            +
            self.spatial_pos
        )

        feature_map = (
            visual
            .transpose(1, 2)
            .reshape(
                B,
                self.hidden_dim,
                self.grid_size,
                self.grid_size
            )
        )

        feature_map = F.interpolate(
            feature_map,
            size=(28, 28),
            mode="bilinear",
            align_corners=False
        )

        feature_map = self.highres1(
            feature_map
        )

        feature_map = F.interpolate(
            feature_map,
            size=(56, 56),
            mode="bilinear",
            align_corners=False
        )

        feature_map = self.highres2(
            feature_map
        )

        visual_tokens = (
            feature_map
            .flatten(2)
            .transpose(1, 2)
        )

        query = self.query_proj(
            text_embedding
        )

        v = self.visual_match(
            visual_tokens
        )

        q = self.query_match(
            query
        ).unsqueeze(1)

        matched = self.match_norm(
            v + q
        )

        logits = self.match_head(
            matched
        ).squeeze(-1)

        heatmap_logits = logits.reshape(
            B,
            56,
            56
        )

        heatmap = F.softmax(
            heatmap_logits.reshape(
                B,
                -1
            ),
            dim=-1
        ).reshape(
            B,
            56,
            56
        )

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
        }


# ============================================================
# EXACT C2 MOMENTS
# ============================================================

def heatmap_moments(heatmap):

    B, H, W = heatmap.shape

    xs = (
        torch.arange(
            W,
            device=heatmap.device,
            dtype=heatmap.dtype
        ) + 0.5
    ) / W

    ys = (
        torch.arange(
            H,
            device=heatmap.device,
            dtype=heatmap.dtype
        ) + 0.5
    ) / H

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.reshape(1, -1)
    yy = yy.reshape(1, -1)

    p = heatmap.reshape(B, -1)

    cx = (
        p * xx
    ).sum(dim=1)

    cy = (
        p * yy
    ).sum(dim=1)

    var_x = (
        p
        *
        (
            xx
            -
            cx.unsqueeze(1)
        ).pow(2)
    ).sum(dim=1)

    var_y = (
        p
        *
        (
            yy
            -
            cy.unsqueeze(1)
        ).pow(2)
    ).sum(dim=1)

    std_x = torch.sqrt(
        var_x + 1e-8
    )

    std_y = torch.sqrt(
        var_y + 1e-8
    )

    return (
        cx,
        cy,
        std_x,
        std_y
    )


# ============================================================
# EXACT C2 HEATMAP -> BOX
# ============================================================

def heatmap_to_box(heatmap):

    cx, cy, std_x, std_y = (
        heatmap_moments(heatmap)
    )

    width = (
        std_x * math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )

    height = (
        std_y * math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )

    x1 = (
        cx - width / 2.0
    ).clamp(
        0.0,
        1.0
    )

    y1 = (
        cy - height / 2.0
    ).clamp(
        0.0,
        1.0
    )

    x2 = (
        cx + width / 2.0
    ).clamp(
        0.0,
        1.0
    )

    y2 = (
        cy + height / 2.0
    ).clamp(
        0.0,
        1.0
    )

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1
    )


# ============================================================
# IoU
# ============================================================

def box_iou(
    boxes1,
    boxes2,
    eps=1e-6,
):

    x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0]
    )

    y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1]
    )

    x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2]
    )

    y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3]
    )

    inter_w = (
        x2 - x1
    ).clamp_min(0.0)

    inter_h = (
        y2 - y1
    ).clamp_min(0.0)

    intersection = (
        inter_w * inter_h
    )

    area1 = (
        (
            boxes1[:, 2]
            -
            boxes1[:, 0]
        ).clamp_min(0.0)
        *
        (
            boxes1[:, 3]
            -
            boxes1[:, 1]
        ).clamp_min(0.0)
    )

    area2 = (
        (
            boxes2[:, 2]
            -
            boxes2[:, 0]
        ).clamp_min(0.0)
        *
        (
            boxes2[:, 3]
            -
            boxes2[:, 1]
        ).clamp_min(0.0)
    )

    union = (
        area1
        +
        area2
        -
        intersection
    ).clamp_min(eps)

    return intersection / union


# ============================================================
# TARGET HEATMAP
# Exact box-cell intersection construction.
# ============================================================

def create_target_heatmap(
    target_box,
    grid_size=56,
):

    device = target_box.device
    dtype = target_box.dtype

    edges = torch.linspace(
        0.0,
        1.0,
        grid_size + 1,
        device=device,
        dtype=dtype
    )

    x_left = edges[:-1].view(
        1,
        grid_size
    )

    x_right = edges[1:].view(
        1,
        grid_size
    )

    y_top = edges[:-1].view(
        grid_size,
        1
    )

    y_bottom = edges[1:].view(
        grid_size,
        1
    )

    x1 = target_box[0]
    y1 = target_box[1]
    x2 = target_box[2]
    y2 = target_box[3]

    inter_w = (
        torch.minimum(
            x_right,
            x2
        )
        -
        torch.maximum(
            x_left,
            x1
        )
    ).clamp_min(0.0)

    inter_h = (
        torch.minimum(
            y_bottom,
            y2
        )
        -
        torch.maximum(
            y_top,
            y1
        )
    ).clamp_min(0.0)

    area = (
        inter_h
        *
        inter_w
    )

    area_sum = (
        area.sum()
        .clamp_min(1e-12)
    )

    return (
        area
        /
        area_sum
    )


# ============================================================
# BOX MASK
# ============================================================

def box_mask(
    box,
    H=56,
    W=56,
):

    x1, y1, x2, y2 = box

    xs = (
        torch.arange(
            W,
            device=box.device,
            dtype=box.dtype
        ) + 0.5
    ) / W

    ys = (
        torch.arange(
            H,
            device=box.device,
            dtype=box.dtype
        ) + 0.5
    ) / H

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    return (
        (xx >= x1)
        &
        (xx <= x2)
        &
        (yy >= y1)
        &
        (yy <= y2)
    )


# ============================================================
# TOP-MASS MASK
# ============================================================

def top_mass_mask(
    heatmap,
    mass_fraction,
):

    flat = heatmap.reshape(-1)

    values, indices = torch.sort(
        flat,
        descending=True
    )

    cumulative = torch.cumsum(
        values,
        dim=0
    )

    selected = (
        cumulative <= mass_fraction
    )

    # Always retain maximum cell.
    selected[0] = True

    # Also retain the first cell crossing
    # the requested mass fraction.
    crossing = torch.nonzero(
        cumulative >= mass_fraction,
        as_tuple=False
    )

    if crossing.numel() > 0:

        first_cross = (
            crossing[0, 0]
            .item()
        )

        selected[first_cross] = True

    selected_indices = (
        indices[selected]
    )

    mask = torch.zeros(
        flat.shape,
        dtype=torch.bool,
        device=heatmap.device
    )

    mask[selected_indices] = True

    return mask.reshape(
        heatmap.shape
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("C2 — EXACT PER-SAMPLE DEEP DIAGNOSTIC")
print("=" * 80)

print(f"Device:              {DEVICE}")
print(f"Checkpoint:          {CHECKPOINT_PATH}")
print(f"Validation parquet:  {VAL_PARQUET}")
print(f"DOFA cache:          {DOFA_ROOT}")
print(f"Text cache:          {TEXT_CACHE_DIR}")
print(f"Output directory:    {OUTPUT_DIR}")
print("GPU:                 OFF")


# ============================================================
# VERIFY PATHS
# ============================================================

for name, path in [
    ("Checkpoint", CHECKPOINT_PATH),
    ("Validation parquet", VAL_PARQUET),
    ("DOFA cache", DOFA_ROOT),
    ("Text cache", TEXT_CACHE_DIR),
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name}: FOUND")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 80)
print("LOADING C2 CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Stored validation IoU:",
    checkpoint.get("val_iou")
)

print(
    "Stored config:",
    checkpoint.get("config")
)


# ============================================================
# MODEL
# ============================================================

model = C2HeatmapModel()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.to(DEVICE)
model.eval()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "C2 model loaded successfully."
)

print(
    "Parameter count:",
    parameter_count
)


# ============================================================
# CACHES
# ============================================================

print()
print("=" * 80)
print("LOADING CACHES")
print("=" * 80)

dofa_cache = DofaShardCache(
    DOFA_ROOT
)

text_cache = GroundingTextCache(
    TEXT_CACHE_DIR
)


# ============================================================
# VALIDATION DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING VALIDATION DATASET")
print("=" * 80)

val_dataset = ReferenceGroundingDataset(
    VAL_PARQUET,
    dofa_cache,
    text_cache
)

print(
    f"Validation rows: {len(val_dataset):,}"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    collate_fn=collate_reference_batch
)

print(
    f"Validation batches: {len(val_loader):,}"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING EXACT C2 DEEP DIAGNOSTIC")
print("=" * 80)

records = []

processed = 0

with torch.no_grad():

    for batch_idx, batch in enumerate(
        val_loader,
        start=1
    ):

        s1 = batch["s1"].to(DEVICE)
        s2 = batch["s2"].to(DEVICE)
        text_embedding = (
            batch["text_embedding"]
            .to(DEVICE)
        )

        target_boxes = (
            batch["target_box"]
            .to(DEVICE)
        )

        outputs = model(
            s1,
            s2,
            text_embedding
        )

        heatmaps = outputs["heatmap"]

        predicted_boxes = (
            heatmap_to_box(
                heatmaps
            )
        )

        batch_ious = box_iou(
            predicted_boxes,
            target_boxes
        )

        for i in range(
            target_boxes.size(0)
        ):

            heatmap = heatmaps[i]
            target_box = target_boxes[i]

            pred_box = predicted_boxes[i]

            # ------------------------------------------------
            # TARGET HEATMAP
            # ------------------------------------------------

            target_heatmap = (
                create_target_heatmap(
                    target_box,
                    grid_size=GRID_SIZE
                )
            )

            # ------------------------------------------------
            # TARGET MASK
            # ------------------------------------------------

            target_mask = box_mask(
                target_box,
                H=GRID_SIZE,
                W=GRID_SIZE
            )

            # ------------------------------------------------
            # PREDICTED MOMENTS
            # ------------------------------------------------

            (
                pred_cx,
                pred_cy,
                pred_std_x,
                pred_std_y
            ) = heatmap_moments(
                heatmap.unsqueeze(0)
            )

            pred_cx = pred_cx[0]
            pred_cy = pred_cy[0]
            pred_std_x = pred_std_x[0]
            pred_std_y = pred_std_y[0]

            # ------------------------------------------------
            # TARGET MOMENTS
            # ------------------------------------------------

            (
                target_cx,
                target_cy,
                target_std_x,
                target_std_y
            ) = heatmap_moments(
                target_heatmap.unsqueeze(0)
            )

            target_cx = target_cx[0]
            target_cy = target_cy[0]
            target_std_x = target_std_x[0]
            target_std_y = target_std_y[0]

            # ------------------------------------------------
            # ERRORS
            # ------------------------------------------------

            centroid_error = torch.sqrt(
                (
                    pred_cx
                    -
                    target_cx
                ).pow(2)
                +
                (
                    pred_cy
                    -
                    target_cy
                ).pow(2)
            )

            spread_error = torch.sqrt(
                (
                    pred_std_x
                    -
                    target_std_x
                ).pow(2)
                +
                (
                    pred_std_y
                    -
                    target_std_y
                ).pow(2)
            )

            signed_spread_x = (
                pred_std_x
                -
                target_std_x
            )

            signed_spread_y = (
                pred_std_y
                -
                target_std_y
            )

            spread_ratio_x = (
                pred_std_x
                /
                target_std_x.clamp_min(
                    1e-12
                )
            )

            spread_ratio_y = (
                pred_std_y
                /
                target_std_y.clamp_min(
                    1e-12
                )
            )

            # ------------------------------------------------
            # GT MASS / LEAKAGE
            # ------------------------------------------------

            gt_mass = (
                heatmap
                *
                target_mask.float()
            ).sum()

            background_leakage = (
                1.0
                -
                gt_mass
            )

            # ------------------------------------------------
            # ENTROPY
            # ------------------------------------------------

            entropy = -torch.sum(
                heatmap
                *
                torch.log(
                    heatmap.clamp_min(
                        1e-12
                    )
                )
            )

            # ------------------------------------------------
            # TARGET HEATMAP CE
            # ------------------------------------------------

            heatmap_ce = -torch.sum(
                target_heatmap
                *
                torch.log(
                    heatmap.clamp_min(
                        1e-12
                    )
                )
            )

            # ------------------------------------------------
            # TARGET HEATMAP KL
            # ------------------------------------------------

            kl_div = torch.sum(
                target_heatmap.clamp_min(
                    1e-12
                )
                *
                (
                    torch.log(
                        target_heatmap.clamp_min(
                            1e-12
                        )
                    )
                    -
                    torch.log(
                        heatmap.clamp_min(
                            1e-12
                        )
                    )
                )
            )

            # ------------------------------------------------
            # PEAK
            # ------------------------------------------------

            peak_value = (
                heatmap.max()
                .item()
            )

            peak_index = (
                heatmap.reshape(-1)
                .argmax()
                .item()
            )

            peak_y = (
                peak_index // GRID_SIZE
            )

            peak_x = (
                peak_index % GRID_SIZE
            )

            peak_x_norm = (
                peak_x + 0.5
            ) / GRID_SIZE

            peak_y_norm = (
                peak_y + 0.5
            ) / GRID_SIZE

            peak_inside_gt = (
                peak_x_norm
                >= target_box[0].item()
                and
                peak_x_norm
                <= target_box[2].item()
                and
                peak_y_norm
                >= target_box[1].item()
                and
                peak_y_norm
                <= target_box[3].item()
            )

            # ------------------------------------------------
            # TARGET AREA
            # ------------------------------------------------

            target_area = (
                (
                    target_box[2]
                    -
                    target_box[0]
                )
                *
                (
                    target_box[3]
                    -
                    target_box[1]
                )
            ).item()

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            record = {

                "dataset_index":
                    processed + i,

                "row_index":
                    int(
                        batch["row_index"][i]
                    ),

                "patch_id":
                    str(
                        batch["patch_id"][i]
                    ),

                "phrase":
                    str(
                        batch["phrase"][i]
                    ),

                "target_x1":
                    target_box[0].item(),

                "target_y1":
                    target_box[1].item(),

                "target_x2":
                    target_box[2].item(),

                "target_y2":
                    target_box[3].item(),

                "target_area":
                    float(target_area),

                "pred_x1":
                    pred_box[0].item(),

                "pred_y1":
                    pred_box[1].item(),

                "pred_x2":
                    pred_box[2].item(),

                "pred_y2":
                    pred_box[3].item(),

                "iou":
                    batch_ious[i].item(),

                "pred_centroid_x":
                    pred_cx.item(),

                "pred_centroid_y":
                    pred_cy.item(),

                "target_centroid_x":
                    target_cx.item(),

                "target_centroid_y":
                    target_cy.item(),

                "centroid_error":
                    centroid_error.item(),

                "pred_std_x":
                    pred_std_x.item(),

                "pred_std_y":
                    pred_std_y.item(),

                "target_std_x":
                    target_std_x.item(),

                "target_std_y":
                    target_std_y.item(),

                "spread_error":
                    spread_error.item(),

                "signed_spread_x":
                    signed_spread_x.item(),

                "signed_spread_y":
                    signed_spread_y.item(),

                "spread_ratio_x":
                    spread_ratio_x.item(),

                "spread_ratio_y":
                    spread_ratio_y.item(),

                "gt_mass":
                    gt_mass.item(),

                "background_leakage":
                    background_leakage.item(),

                "heatmap_entropy":
                    entropy.item(),

                "heatmap_cross_entropy":
                    heatmap_ce.item(),

                "heatmap_kl_divergence":
                    kl_div.item(),

                "peak_value":
                    peak_value,

                "heatmap_peak_x":
                    peak_x_norm,

                "heatmap_peak_y":
                    peak_y_norm,

                "heatmap_peak_inside_gt":
                    int(peak_inside_gt),
            }

            # ------------------------------------------------
            # TOP MASS ANALYSIS
            # ------------------------------------------------

            for fraction in TOP_MASS_LEVELS:

                pct = int(
                    fraction * 100
                )

                mask = top_mass_mask(
                    heatmap,
                    fraction
                )

                selected_count = (
                    mask.sum().item()
                )

                overlap_count = (
                    (
                        mask
                        &
                        target_mask
                    )
                    .sum()
                    .item()
                )

                gt_cell_count = (
                    target_mask
                    .sum()
                    .item()
                )

                if selected_count > 0:
                    precision = (
                        overlap_count
                        /
                        selected_count
                    )
                else:
                    precision = 0.0

                if gt_cell_count > 0:
                    recall = (
                        overlap_count
                        /
                        gt_cell_count
                    )
                else:
                    recall = 0.0

                selected_gt_mass = (
                    heatmap
                    *
                    (
                        mask
                        &
                        target_mask
                    ).float()
                ).sum().item()

                selected_mass = (
                    heatmap[mask]
                    .sum()
                    .item()
                )

                record[
                    f"top_{pct}_selected_mass"
                ] = selected_mass

                record[
                    f"top_{pct}_precision"
                ] = precision

                record[
                    f"top_{pct}_recall"
                ] = recall

                record[
                    f"top_{pct}_gt_mass"
                ] = selected_gt_mass

            records.append(record)

        processed += target_boxes.size(0)

        if (
            batch_idx == 1
            or batch_idx % 20 == 0
            or batch_idx == len(val_loader)
        ):
            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(val_dataset):,}"
            )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if len(df) != len(val_dataset):
    raise RuntimeError(
        f"Expected {len(val_dataset)} records "
        f"but got {len(df)}."
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "stored_validation_iou":
        checkpoint.get("val_iou"),

    "reproduction_mean_iou":
        float(df["iou"].mean()),

    "median_iou":
        float(df["iou"].median()),

    "iou_ge_025_percent":
        float(
            (df["iou"] >= 0.25).mean()
            * 100.0
        ),

    "iou_ge_050_percent":
        float(
            (df["iou"] >= 0.50).mean()
            * 100.0
        ),

    "iou_ge_075_percent":
        float(
            (df["iou"] >= 0.75).mean()
            * 100.0
        ),

    "target_area_mean":
        float(df["target_area"].mean()),

    "target_area_median":
        float(df["target_area"].median()),

    "centroid_error_mean":
        float(df["centroid_error"].mean()),

    "centroid_error_median":
        float(df["centroid_error"].median()),

    "spread_error_mean":
        float(df["spread_error"].mean()),

    "spread_error_median":
        float(df["spread_error"].median()),

    "signed_spread_x_mean":
        float(df["signed_spread_x"].mean()),

    "signed_spread_y_mean":
        float(df["signed_spread_y"].mean()),

    "spread_ratio_x_mean":
        float(df["spread_ratio_x"].mean()),

    "spread_ratio_x_median":
        float(df["spread_ratio_x"].median()),

    "spread_ratio_y_mean":
        float(df["spread_ratio_y"].mean()),

    "spread_ratio_y_median":
        float(df["spread_ratio_y"].median()),

    "predicted_spread_x_wider_percent":
        float(
            (
                df["signed_spread_x"] > 0
            ).mean()
            * 100.0
        ),

    "predicted_spread_y_wider_percent":
        float(
            (
                df["signed_spread_y"] > 0
            ).mean()
            * 100.0
        ),

    "gt_mass_mean":
        float(df["gt_mass"].mean()),

    "gt_mass_median":
        float(df["gt_mass"].median()),

    "background_leakage_mean":
        float(
            df["background_leakage"].mean()
        ),

    "background_leakage_median":
        float(
            df["background_leakage"].median()
        ),

    "heatmap_entropy_mean":
        float(
            df["heatmap_entropy"].mean()
        ),

    "heatmap_entropy_median":
        float(
            df["heatmap_entropy"].median()
        ),

    "heatmap_ce_mean":
        float(
            df["heatmap_cross_entropy"].mean()
        ),

    "heatmap_kl_mean":
        float(
            df["heatmap_kl_divergence"].mean()
        ),

    "peak_inside_gt_percent":
        float(
            df["heatmap_peak_inside_gt"].mean()
            * 100.0
        ),

    "top_10_precision_mean":
        float(
            df["top_10_precision"].mean()
        ),

    "top_10_recall_mean":
        float(
            df["top_10_recall"].mean()
        ),

    "top_25_precision_mean":
        float(
            df["top_25_precision"].mean()
        ),

    "top_25_recall_mean":
        float(
            df["top_25_recall"].mean()
        ),

    "top_50_precision_mean":
        float(
            df["top_50_precision"].mean()
        ),

    "top_50_recall_mean":
        float(
            df["top_50_recall"].mean()
        ),

    "top_75_precision_mean":
        float(
            df["top_75_precision"].mean()
        ),

    "top_75_recall_mean":
        float(
            df["top_75_recall"].mean()
        ),

    "rows":
        int(len(df)),
}


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("C2 DEEP DIAGNOSTIC RESULTS")
print("=" * 80)

print(
    f"Rows                  : {len(df):,}"
)

print(
    f"Mean IoU              : "
    f"{df['iou'].mean():.9f}"
)

print(
    f"Median IoU            : "
    f"{df['iou'].median():.9f}"
)

print(
    f"IoU >= 0.25           : "
    f"{(df['iou'] >= 0.25).mean() * 100:.2f}%"
)

print(
    f"IoU >= 0.50           : "
    f"{(df['iou'] >= 0.50).mean() * 100:.2f}%"
)

print(
    f"IoU >= 0.75           : "
    f"{(df['iou'] >= 0.75).mean() * 100:.2f}%"
)

print()

print(
    f"Centroid error        : "
    f"{df['centroid_error'].mean():.9f}"
)

print(
    f"Spread error          : "
    f"{df['spread_error'].mean():.9f}"
)

print(
    f"Spread ratio X mean   : "
    f"{df['spread_ratio_x'].mean():.6f}"
)

print(
    f"Spread ratio X median : "
    f"{df['spread_ratio_x'].median():.6f}"
)

print(
    f"Spread ratio Y mean   : "
    f"{df['spread_ratio_y'].mean():.6f}"
)

print(
    f"Spread ratio Y median : "
    f"{df['spread_ratio_y'].median():.6f}"
)

print(
    f"X wider than target   : "
    f"{(df['signed_spread_x'] > 0).mean() * 100:.2f}%"
)

print(
    f"Y wider than target   : "
    f"{(df['signed_spread_y'] > 0).mean() * 100:.2f}%"
)

print()

print(
    f"GT mass               : "
    f"{df['gt_mass'].mean():.9f}"
)

print(
    f"Background leakage    : "
    f"{df['background_leakage'].mean():.9f}"
)

print(
    f"Peak inside GT        : "
    f"{df['heatmap_peak_inside_gt'].mean() * 100:.2f}%"
)

print(
    f"Heatmap entropy       : "
    f"{df['heatmap_entropy'].mean():.9f}"
)

print()

print(
    f"Top-10 precision      : "
    f"{df['top_10_precision'].mean():.6f}"
)

print(
    f"Top-10 recall         : "
    f"{df['top_10_recall'].mean():.6f}"
)

print(
    f"Top-25 precision      : "
    f"{df['top_25_precision'].mean():.6f}"
)

print(
    f"Top-25 recall         : "
    f"{df['top_25_recall'].mean():.6f}"
)

print(
    f"Top-50 precision      : "
    f"{df['top_50_precision'].mean():.6f}"
)

print(
    f"Top-50 recall         : "
    f"{df['top_50_recall'].mean():.6f}"
)

print(
    f"Top-75 precision      : "
    f"{df['top_75_precision'].mean():.6f}"
)

print(
    f"Top-75 recall         : "
    f"{df['top_75_recall'].mean():.6f}"
)

print()

print(
    "Output CSV:"
)

print(
    OUTPUT_CSV
)

print(
    "Output summary:"
)

print(
    OUTPUT_JSON
)

print("=" * 80)