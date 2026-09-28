import os
import sys
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# EXACT PROJECT IMPLEMENTATION
# ============================================================

from src.models.train_reference_grounding import (
    DofaShardCache,
    GroundingTextCache,
    ReferenceGroundingDataset,
    collate_reference_batch,
)


# ============================================================
# PATHS
# ============================================================

VAL_PARQUET = (
    r"D:\SatQueryAI\data\processed"
    r"\reference_grounding_split"
    r"\reference_val.parquet"
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

CHECKPOINT_PATH = (
    r"D:\SatQueryAI\models\checkpoints"
    r"\c1_high_resolution"
    r"\best.pt"
)

OUTPUT_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\c1_diagnostic_topmass"
)

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cpu")


# ============================================================
# IOU
# ============================================================

def box_iou(
    boxes1,
    boxes2,
    eps=1e-6,
):
    x1 = torch.maximum(
        boxes1[:, 0],
        boxes2[:, 0],
    )

    y1 = torch.maximum(
        boxes1[:, 1],
        boxes2[:, 1],
    )

    x2 = torch.minimum(
        boxes1[:, 2],
        boxes2[:, 2],
    )

    y2 = torch.minimum(
        boxes1[:, 3],
        boxes2[:, 3],
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
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0)
    )

    union = (
        area1
        + area2
        - intersection
    ).clamp_min(eps)

    return intersection / union


# ============================================================
# EXACT C1 MODEL
# ============================================================

class ExactC1GroundingHeatmapModel(
    nn.Module
):

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

        # ----------------------------------------------------
        # VISUAL FUSION
        # ----------------------------------------------------

        self.visual_fusion = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        # ----------------------------------------------------
        # SPATIAL POSITION
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.zeros(
                1,
                grid_size * grid_size,
                hidden_dim,
            )
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02,
        )

        # ----------------------------------------------------
        # HIGH-RESOLUTION DECODER
        # 14 -> 28 -> 56
        # ----------------------------------------------------

        self.highres1 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.GroupNorm(
                32,
                hidden_dim,
            ),
        )

        self.highres2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.GroupNorm(
                32,
                hidden_dim,
            ),
        )

        # ----------------------------------------------------
        # QUERY PROJECTION
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        # ----------------------------------------------------
        # VISUAL / QUERY MATCHING
        # ----------------------------------------------------

        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.query_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.match_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.match_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1,
            ),
        )

        # ----------------------------------------------------
        # SPATIAL SUMMARY
        # ----------------------------------------------------

        self.spatial_summary = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        # ----------------------------------------------------
        # CORRECTION HEAD
        # ----------------------------------------------------

        self.correction_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + 8,
                256,
            ),
            nn.GELU(),
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                4,
            ),
        )

    # ========================================================
    # SPATIAL GEOMETRY
    # ========================================================

    def extract_spatial_geometry(
        self,
        heatmap,
    ):

        B, H, W = heatmap.shape

        device = heatmap.device
        dtype = heatmap.dtype

        xs = (
            torch.arange(
                W,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) / W

        ys = (
            torch.arange(
                H,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) / H

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        xx = xx.reshape(
            1,
            -1,
        )

        yy = yy.reshape(
            1,
            -1,
        )

        p = heatmap.reshape(
            B,
            -1,
        )

        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = (
            p * xx
        ).sum(
            dim=1,
        )

        center_y = (
            p * yy
        ).sum(
            dim=1,
        )

        # ----------------------------------------------------
        # VARIANCE
        # ----------------------------------------------------

        variance_x = (
            p
            *
            (
                xx
                -
                center_x.unsqueeze(1)
            ).pow(2)
        ).sum(
            dim=1,
        )

        variance_y = (
            p
            *
            (
                yy
                -
                center_y.unsqueeze(1)
            ).pow(2)
        ).sum(
            dim=1,
        )

        std_x = torch.sqrt(
            variance_x + 1e-8,
        )

        std_y = torch.sqrt(
            variance_y + 1e-8,
        )

        # ----------------------------------------------------
        # UNIFORM-RECTANGLE APPROXIMATION
        # ----------------------------------------------------

        estimated_width = (
            std_x
            * math.sqrt(12.0)
        ).clamp(
            min=0.02,
            max=1.0,
        )

        estimated_height = (
            std_y
            * math.sqrt(12.0)
        ).clamp(
            min=0.02,
            max=1.0,
        )

        # ----------------------------------------------------
        # BASE BOX
        # ----------------------------------------------------

        estimated_x1 = (
            center_x
            -
            estimated_width / 2.0
        ).clamp(
            min=0.0,
            max=1.0,
        )

        estimated_y1 = (
            center_y
            -
            estimated_height / 2.0
        ).clamp(
            min=0.0,
            max=1.0,
        )

        estimated_x2 = (
            center_x
            +
            estimated_width / 2.0
        ).clamp(
            min=0.0,
            max=1.0,
        )

        estimated_y2 = (
            center_y
            +
            estimated_height / 2.0
        ).clamp(
            min=0.0,
            max=1.0,
        )

        geometry = torch.stack(
            [
                center_x,
                center_y,
                std_x,
                std_y,
                estimated_x1,
                estimated_y1,
                estimated_x2,
                estimated_y2,
            ],
            dim=-1,
        )

        base_box = geometry[
            :,
            4:8,
        ]

        return geometry, base_box

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        s1,
        s2,
        text_embedding,
    ):

        B = s1.size(0)

        # ----------------------------------------------------
        # 1. S1 + S2
        # ----------------------------------------------------

        visual = torch.cat(
            [
                s1,
                s2,
            ],
            dim=-1,
        )

        visual = self.visual_fusion(
            visual
        )

        # ----------------------------------------------------
        # 2. 14x14 POSITION
        # ----------------------------------------------------

        visual = (
            visual
            +
            self.spatial_pos
        )

        # ----------------------------------------------------
        # 3. 14x14 -> FEATURE MAP
        # ----------------------------------------------------

        feature_map = (
            visual
            .transpose(
                1,
                2,
            )
            .reshape(
                B,
                self.hidden_dim,
                self.grid_size,
                self.grid_size,
            )
        )

        # ----------------------------------------------------
        # 4. 14 -> 28
        # ----------------------------------------------------

        feature_map = F.interpolate(
            feature_map,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )

        feature_map = self.highres1(
            feature_map
        )

        # ----------------------------------------------------
        # 5. 28 -> 56
        # ----------------------------------------------------

        feature_map = F.interpolate(
            feature_map,
            size=(56, 56),
            mode="bilinear",
            align_corners=False,
        )

        feature_map = self.highres2(
            feature_map
        )

        # ----------------------------------------------------
        # 6. 56x56 TOKENS
        # ----------------------------------------------------

        visual_tokens = (
            feature_map
            .flatten(2)
            .transpose(1, 2)
        )

        # ----------------------------------------------------
        # 7. QUERY
        # ----------------------------------------------------

        query = self.query_proj(
            text_embedding
        )

        # ----------------------------------------------------
        # 8. QUERY-CONDITIONED MATCHING
        # ----------------------------------------------------

        v = self.visual_match(
            visual_tokens
        )

        q = self.query_match(
            query
        ).unsqueeze(1)

        matched = self.match_norm(
            v + q
        )

        # ----------------------------------------------------
        # 9. LOCALIZATION LOGITS
        # ----------------------------------------------------

        logits = self.match_head(
            matched
        ).squeeze(-1)

        heatmap_logits = logits.reshape(
            B,
            56,
            56,
        )

        # ----------------------------------------------------
        # 10. SPATIAL SOFTMAX
        # ----------------------------------------------------

        heatmap = F.softmax(
            heatmap_logits.reshape(
                B,
                -1,
            ),
            dim=-1,
        ).reshape(
            B,
            56,
            56,
        )

        # ----------------------------------------------------
        # 11. SPATIAL CONTEXT
        # ----------------------------------------------------

        spatial_context = (
            visual_tokens
            *
            heatmap.reshape(
                B,
                -1,
                1,
            )
        ).sum(
            dim=1,
        )

        spatial_context = (
            self.spatial_summary(
                spatial_context
            )
        )

        # ----------------------------------------------------
        # 12. GEOMETRY
        # ----------------------------------------------------

        geometry, base_box = (
            self.extract_spatial_geometry(
                heatmap
            )
        )

        # ----------------------------------------------------
        # 13. CORRECTION INPUT
        # ----------------------------------------------------

        correction_input = torch.cat(
            [
                spatial_context,
                query,
                geometry,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # 14. CORRECTION
        # ----------------------------------------------------

        correction = (
            torch.tanh(
                self.correction_head(
                    correction_input
                )
            )
            * 0.2
        )

        # ----------------------------------------------------
        # 15. FINAL BOX
        # ----------------------------------------------------

        box = (
            base_box
            +
            correction
        )

        x1 = torch.minimum(
            box[:, 0],
            box[:, 2],
        )

        y1 = torch.minimum(
            box[:, 1],
            box[:, 3],
        )

        x2 = torch.maximum(
            box[:, 0],
            box[:, 2],
        )

        y2 = torch.maximum(
            box[:, 1],
            box[:, 3],
        )

        final_box = torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        ).clamp(
            0.0,
            1.0,
        )

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "geometry": geometry,
            "base_box": base_box,
            "correction": correction,
            "box": final_box,
        }


# ============================================================
# TARGET HEATMAP
# ============================================================

def create_target_heatmap(
    target_box,
    grid_size=56,
):
    """
    Exact C1 target construction.

    Each grid cell receives the area of its
    intersection with the target box.

    The resulting map is normalized to sum to 1.
    """

    device = target_box.device
    dtype = target_box.dtype

    edges = torch.linspace(
        0.0,
        1.0,
        grid_size + 1,
        device=device,
        dtype=dtype,
    )

    x_left = edges[
        :-1
    ].view(
        1,
        grid_size,
    )

    x_right = edges[
        1:
    ].view(
        1,
        grid_size,
    )

    y_top = edges[
        :-1
    ].view(
        grid_size,
        1,
    )

    y_bottom = edges[
        1:
    ].view(
        grid_size,
        1,
    )

    x1 = target_box[0]
    y1 = target_box[1]
    x2 = target_box[2]
    y2 = target_box[3]

    inter_w = (
        torch.minimum(
            x_right,
            x2,
        )
        -
        torch.maximum(
            x_left,
            x1,
        )
    ).clamp_min(
        0.0
    )

    inter_h = (
        torch.minimum(
            y_bottom,
            y2,
        )
        -
        torch.maximum(
            y_top,
            y1,
        )
    ).clamp_min(
        0.0
    )

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
# GRID BOX MASK
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
            dtype=box.dtype,
        )
        + 0.5
    ) / W

    ys = (
        torch.arange(
            H,
            device=box.device,
            dtype=box.dtype,
        )
        + 0.5
    ) / H

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
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
    """
    Select highest-probability cells until approximately
    mass_fraction of total probability is accumulated.
    """

    flat = heatmap.reshape(
        -1
    )

    values, indices = torch.sort(
        flat,
        descending=True,
    )

    cumulative = torch.cumsum(
        values,
        dim=0,
    )

    selected = (
        cumulative
        <= mass_fraction
    )

    # Always include the highest-probability cell.

    selected[0] = True

    # Include the cell that crosses the target fraction.

    crossing = torch.nonzero(
        cumulative >= mass_fraction,
        as_tuple=False,
    )

    if crossing.numel() > 0:

        first_cross = (
            crossing[
                0,
                0,
            ]
            .item()
        )

        selected[
            first_cross
        ] = True

    selected_indices = (
        indices[selected]
    )

    mask = torch.zeros(
        flat.shape,
        dtype=torch.bool,
        device=heatmap.device,
    )

    mask[
        selected_indices
    ] = True

    return mask.reshape(
        heatmap.shape
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("C1 — EXACT HEATMAP DIAGNOSTIC")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)

print(
    f"Validation parquet: {VAL_PARQUET}"
)

print(
    f"DOFA cache: {DOFA_ROOT}"
)

print(
    f"Text cache: {TEXT_CACHE_DIR}"
)

print(
    f"Output: {OUTPUT_DIR}"
)

print(
    "GPU OFF — CPU inference only."
)


# ============================================================
# CHECKPOINT
# ============================================================

if not os.path.isfile(
    CHECKPOINT_PATH
):
    raise FileNotFoundError(
        CHECKPOINT_PATH
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print()
print(
    "Checkpoint epoch:"
)

print(
    checkpoint.get(
        "epoch",
        "unknown",
    )
)

print(
    "Stored validation IoU:"
)

print(
    checkpoint.get(
        "val_iou",
        "unknown",
    )
)


# ============================================================
# MODEL
# ============================================================

model = ExactC1GroundingHeatmapModel()

missing, unexpected = (
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=False,
    )
)

print()
print(
    "Missing keys:",
    len(missing),
)

print(
    "Unexpected keys:",
    len(unexpected),
)

if missing or unexpected:
    raise RuntimeError(
        "Checkpoint does not exactly match "
        "the reconstructed C1 architecture."
    )

model.to(
    DEVICE
)

model.eval()

print()
print(
    "Exact C1 checkpoint loaded successfully."
)


# ============================================================
# VERIFY PATHS
# ============================================================

print()
print("=" * 80)
print("VERIFYING DATA PATHS")
print("=" * 80)

for name, path in [
    (
        "Validation parquet",
        VAL_PARQUET,
    ),
    (
        "DOFA cache",
        DOFA_ROOT,
    ),
    (
        "Text cache",
        TEXT_CACHE_DIR,
    ),
]:

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(
        f"{name}: FOUND"
    )


# ============================================================
# CACHES
# ============================================================

print()
print("=" * 80)
print("LOADING EXACT PROJECT CACHES")
print("=" * 80)

dofa_cache = DofaShardCache(
    DOFA_ROOT
)

text_cache = GroundingTextCache(
    TEXT_CACHE_DIR
)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING EXACT VALIDATION DATASET")
print("=" * 80)

val_dataset = ReferenceGroundingDataset(
    VAL_PARQUET,
    dofa_cache,
    text_cache,
)

print(
    f"Validation rows: "
    f"{len(val_dataset):,}"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_reference_batch,
)

print(
    f"Validation batches: "
    f"{len(val_loader):,}"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# STORAGE
# ============================================================

records = []

TOP_MASS_LEVELS = [
    0.10,
    0.25,
    0.50,
    0.75,
]


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING EXACT C1 HEATMAP ANALYSIS")
print("=" * 80)

processed = 0

with torch.no_grad():

    for batch_idx, batch in enumerate(
        val_loader,
        start=1,
    ):

        s1 = batch[
            "s1"
        ].to(
            DEVICE
        )

        s2 = batch[
            "s2"
        ].to(
            DEVICE
        )

        text_embedding = batch[
            "text_embedding"
        ].to(
            DEVICE
        )

        target_boxes = batch[
            "target_box"
        ].to(
            DEVICE
        )

        outputs = model(
            s1,
            s2,
            text_embedding,
        )

        heatmaps = outputs[
            "heatmap"
        ]

        base_boxes = outputs[
            "base_box"
        ]

        B = (
            target_boxes.size(0)
        )

        base_ious = box_iou(
            base_boxes,
            target_boxes,
        )

        # ----------------------------------------------------
        # PER SAMPLE
        # ----------------------------------------------------

        for i in range(B):

            heatmap = heatmaps[i]

            target_box = target_boxes[i]

            base_iou_value = (
                base_ious[i]
                .item()
            )

            # ------------------------------------------------
            # EXACT TARGET HEATMAP
            # ------------------------------------------------

            target_heatmap = (
                create_target_heatmap(
                    target_box,
                    grid_size=56,
                )
            )

            # ------------------------------------------------
            # TARGET MASK
            # ------------------------------------------------

            target_mask = box_mask(
                target_box
            )

            # ------------------------------------------------
            # PREDICTED HEATMAP CENTROID
            # ------------------------------------------------

            xs = (
                torch.arange(
                    56,
                    device=DEVICE,
                    dtype=heatmap.dtype,
                )
                + 0.5
            ) / 56.0

            ys = (
                torch.arange(
                    56,
                    device=DEVICE,
                    dtype=heatmap.dtype,
                )
                + 0.5
            ) / 56.0

            yy, xx = torch.meshgrid(
                ys,
                xs,
                indexing="ij",
            )

            pred_cx = (
                heatmap * xx
            ).sum()

            pred_cy = (
                heatmap * yy
            ).sum()

            # ------------------------------------------------
            # TARGET HEATMAP CENTROID
            # ------------------------------------------------

            target_cx = (
                target_heatmap * xx
            ).sum()

            target_cy = (
                target_heatmap * yy
            ).sum()

            # ------------------------------------------------
            # CENTROID ERROR
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

            # ------------------------------------------------
            # PREDICTED SPREAD
            # ------------------------------------------------

            pred_var_x = (
                heatmap
                *
                (
                    xx
                    -
                    pred_cx
                ).pow(2)
            ).sum()

            pred_var_y = (
                heatmap
                *
                (
                    yy
                    -
                    pred_cy
                ).pow(2)
            ).sum()

            pred_std_x = torch.sqrt(
                pred_var_x
                +
                1e-12
            )

            pred_std_y = torch.sqrt(
                pred_var_y
                +
                1e-12
            )

            # ------------------------------------------------
            # TARGET SPREAD
            # ------------------------------------------------

            target_var_x = (
                target_heatmap
                *
                (
                    xx
                    -
                    target_cx
                ).pow(2)
            ).sum()

            target_var_y = (
                target_heatmap
                *
                (
                    yy
                    -
                    target_cy
                ).pow(2)
            ).sum()

            target_std_x = torch.sqrt(
                target_var_x
                +
                1e-12
            )

            target_std_y = torch.sqrt(
                target_var_y
                +
                1e-12
            )

            # ------------------------------------------------
            # SPREAD ERROR
            # ------------------------------------------------

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

            # ------------------------------------------------
            # GT MASS
            # ------------------------------------------------

            gt_mass = (
                heatmap
                *
                target_mask.float()
            ).sum()

            # ------------------------------------------------
            # BACKGROUND LEAKAGE
            # ------------------------------------------------

            background_leakage = (
                1.0
                -
                gt_mass
            )

            # ------------------------------------------------
            # HEATMAP CROSS-ENTROPY
            # ------------------------------------------------

            predicted_log_probs = torch.log(
                heatmap.clamp_min(
                    1e-12
                )
            )

            heatmap_ce = -(
                target_heatmap
                *
                predicted_log_probs
            ).sum()

            # ------------------------------------------------
            # HEATMAP KL DIVERGENCE
            # ------------------------------------------------

            target_positive = (
                target_heatmap
                .clamp_min(
                    1e-12
                )
            )

            kl_div = (
                target_positive
                *
                (
                    torch.log(
                        target_positive
                    )
                    -
                    predicted_log_probs
                )
            ).sum()

            # ------------------------------------------------
            # MAXIMUM PEAK
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
                peak_index // 56
            )

            peak_x = (
                peak_index % 56
            )

            peak_x_norm = (
                peak_x + 0.5
            ) / 56.0

            peak_y_norm = (
                peak_y + 0.5
            ) / 56.0

            peak_inside = (
                (
                    peak_x_norm
                    >= target_box[0]
                    .item()
                )
                and
                (
                    peak_x_norm
                    <= target_box[2]
                    .item()
                )
                and
                (
                    peak_y_norm
                    >= target_box[1]
                    .item()
                )
                and
                (
                    peak_y_norm
                    <= target_box[3]
                    .item()
                )
            )

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            record = {
                "dataset_index":
                    processed + i,

                "row_index":
                    int(
                        batch[
                            "row_index"
                        ][i]
                    ),

                "patch_id":
                    str(
                        batch[
                            "patch_id"
                        ][i]
                    ),

                "phrase":
                    str(
                        batch[
                            "phrase"
                        ][i]
                    ),

                "target_area":
                    float(
                        (
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
                    ),

                "base_iou":
                    base_iou_value,

                "pred_centroid_x":
                    float(
                        pred_cx.item()
                    ),

                "pred_centroid_y":
                    float(
                        pred_cy.item()
                    ),

                "target_centroid_x":
                    float(
                        target_cx.item()
                    ),

                "target_centroid_y":
                    float(
                        target_cy.item()
                    ),

                "centroid_error":
                    float(
                        centroid_error.item()
                    ),

                "pred_std_x":
                    float(
                        pred_std_x.item()
                    ),

                "pred_std_y":
                    float(
                        pred_std_y.item()
                    ),

                "target_std_x":
                    float(
                        target_std_x.item()
                    ),

                "target_std_y":
                    float(
                        target_std_y.item()
                    ),

                "spread_error":
                    float(
                        spread_error.item()
                    ),

                "gt_mass":
                    float(
                        gt_mass.item()
                    ),

                "background_leakage":
                    float(
                        background_leakage.item()
                    ),

                "heatmap_cross_entropy":
                    float(
                        heatmap_ce.item()
                    ),

                "heatmap_kl_divergence":
                    float(
                        kl_div.item()
                    ),

                "peak_value":
                    float(
                        peak_value
                    ),

                "peak_x":
                    float(
                        peak_x_norm
                    ),

                "peak_y":
                    float(
                        peak_y_norm
                    ),

                "peak_inside_gt":
                    int(
                        peak_inside
                    ),
            }

            # ------------------------------------------------
            # TOP MASS ANALYSIS
            # ------------------------------------------------

            for fraction in (
                TOP_MASS_LEVELS
            ):

                pct = int(
                    fraction * 100
                )

                mask = (
                    top_mass_mask(
                        heatmap,
                        fraction,
                    )
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
                    (
                        heatmap
                        *
                        (
                            mask
                            &
                            target_mask
                        ).float()
                    )
                    .sum()
                    .item()
                )

                record[
                    f"top_{pct}_selected_mass"
                ] = (
                    heatmap[mask]
                    .sum()
                    .item()
                )

                record[
                    f"top_{pct}_precision"
                ] = precision

                record[
                    f"top_{pct}_recall"
                ] = recall

                record[
                    f"top_{pct}_gt_mass"
                ] = selected_gt_mass

            records.append(
                record
            )

        processed += B

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(
                val_loader
            )
        ):

            print(
                f"Processed "
                f"{processed:,}/"
                f"{len(val_dataset):,}"
            )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)

if len(df) != len(
    val_dataset
):
    raise RuntimeError(
        f"Expected "
        f"{len(val_dataset)} "
        f"records but got "
        f"{len(df)}."
    )


# ============================================================
# POSITION / SPREAD SUMMARY
# ============================================================

print()
print("=" * 80)
print("HEATMAP POSITION / SPREAD DIAGNOSTICS")
print("=" * 80)

print()

print(
    f"Mean centroid error:       "
    f"{df['centroid_error'].mean():.6f}"
)

print(
    f"Median centroid error:     "
    f"{df['centroid_error'].median():.6f}"
)

print(
    f"Mean background leakage:   "
    f"{df['background_leakage'].mean():.6f}"
)

print(
    f"Median background leakage: "
    f"{df['background_leakage'].median():.6f}"
)

print(
    f"Mean spread error:          "
    f"{df['spread_error'].mean():.6f}"
)

print(
    f"Median spread error:        "
    f"{df['spread_error'].median():.6f}"
)

print()

print(
    "Mean heatmap cross-entropy: "
    f"{df['heatmap_cross_entropy'].mean():.6f}"
)

print(
    "Mean heatmap KL divergence: "
    f"{df['heatmap_kl_divergence'].mean():.6f}"
)

print()

print(
    "Correlation with base IoU:"
)

print(
    f"  centroid error:           "
    f"{df['centroid_error'].corr(df['base_iou']):.6f}"
)

print(
    f"  background leakage:       "
    f"{df['background_leakage'].corr(df['base_iou']):.6f}"
)

print(
    f"  spread error:             "
    f"{df['spread_error'].corr(df['base_iou']):.6f}"
)

print(
    f"  heatmap CE:               "
    f"{df['heatmap_cross_entropy'].corr(df['base_iou']):.6f}"
)

print(
    f"  heatmap KL:               "
    f"{df['heatmap_kl_divergence'].corr(df['base_iou']):.6f}"
)


# ============================================================
# TOP MASS SUMMARY
# ============================================================

print()
print("=" * 80)
print("TOP PROBABILITY MASS RESULTS")
print("=" * 80)

top_rows = []

for fraction in (
    TOP_MASS_LEVELS
):

    pct = int(
        fraction * 100
    )

    precision_col = (
        f"top_{pct}_precision"
    )

    recall_col = (
        f"top_{pct}_recall"
    )

    gt_mass_col = (
        f"top_{pct}_gt_mass"
    )

    top_rows.append(
        {
            "top_mass_fraction":
                fraction,

            "mean_gt_precision":
                df[
                    precision_col
                ].mean(),

            "median_gt_precision":
                df[
                    precision_col
                ].median(),

            "mean_gt_recall":
                df[
                    recall_col
                ].mean(),

            "median_gt_recall":
                df[
                    recall_col
                ].median(),

            "mean_gt_probability_mass":
                df[
                    gt_mass_col
                ].mean(),

            "median_gt_probability_mass":
                df[
                    gt_mass_col
                ].median(),

            "correlation_precision_base_iou":
                df[
                    precision_col
                ].corr(
                    df["base_iou"]
                ),

            "correlation_recall_base_iou":
                df[
                    recall_col
                ].corr(
                    df["base_iou"]
                ),

            "correlation_gt_mass_base_iou":
                df[
                    gt_mass_col
                ].corr(
                    df["base_iou"]
                ),
        }
    )

top_summary_df = pd.DataFrame(
    top_rows
)

print()

print(
    top_summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# BASE BOX SUMMARY
# ============================================================

print()
print("=" * 80)
print("BASE BOX")
print("=" * 80)

print(
    f"Mean IoU:   "
    f"{df['base_iou'].mean():.9f}"
)

print(
    f"Median IoU: "
    f"{df['base_iou'].median():.9f}"
)

print(
    f"IoU >= .25: "
    f"{(
        df['base_iou'] >= 0.25
    ).mean() * 100:.2f}%"
)

print(
    f"IoU >= .50: "
    f"{(
        df['base_iou'] >= 0.50
    ).mean() * 100:.2f}%"
)


# ============================================================
# SAVE FILES
# ============================================================

per_sample_path = os.path.join(
    OUTPUT_DIR,
    "heatmap_analysis_per_sample.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "heatmap_analysis_summary.csv",
)

df.to_csv(
    per_sample_path,
    index=False,
)

top_summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print()

print(
    "Per-sample:"
)

print(
    per_sample_path
)

print()

print(
    "Top-mass summary:"
)

print(
    summary_path
)

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print()
print(
    "Checkpoint was only read."
)

print(
    "No training was performed."
)

print(
    "No checkpoint was modified."
)