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
# EXACT LOCAL DATA/CACHE IMPLEMENTATION
#
# IMPORTANT:
# We reuse the project's actual:
#   DofaShardCache
#   GroundingTextCache
#   ReferenceGroundingDataset
#   collate_reference_batch
#
# We do NOT recreate the cache indexing logic here.
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
    r"\c1_diagnostic"
)

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

def box_iou(boxes1, boxes2, eps=1e-6):
    """
    Batched IoU for boxes in [x1, y1, x2, y2] format.
    This matches the IoU calculation used by the actual C1
    Kaggle training/validation code.
    """

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
# HEADER
# ============================================================

print("=" * 80)
print("C1 — EXACT CHECKPOINT DIAGNOSTIC")
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
# EXACT C1 MODEL
#
# This is the architecture that matches the checkpoint:
#
#   S1 + S2
#      ↓
#   visual fusion
#      ↓
#   14x14
#      ↓
#   28x28
#      ↓
#   56x56
#      ↓
#   query-conditioned heatmap
#      ↓
#   geometry
#      ↓
#   base box
#      ↓
#   correction
#      ↓
#   final box
#
# This definition intentionally does NOT use the current
# local GroundingHeatmapModel, because that is a different
# architecture with box_correction_head.
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
        # 14x14 SPATIAL POSITION
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
        #
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
        # QUERY
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
        #
        # spatial_context = 256
        # query          = 256
        # geometry       = 8
        #
        # total = 520
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
    # EXACT SPATIAL GEOMETRY
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
        #
        # std = width / sqrt(12)
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
    # EXACT FORWARD
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
        # 2. 14x14 SPATIAL POSITION
        # ----------------------------------------------------

        visual = (
            visual
            +
            self.spatial_pos
        )

        # ----------------------------------------------------
        # 3. 14x14 TOKENS -> FEATURE MAP
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
        # 6. 56x56 FEATURE MAP -> TOKENS
        # ----------------------------------------------------

        visual_tokens = (
            feature_map
            .flatten(2)
            .transpose(1, 2)
        )

        # [B, 3136, 256]

        # ----------------------------------------------------
        # 7. QUERY
        # ----------------------------------------------------

        query = self.query_proj(
            text_embedding
        )

        # ----------------------------------------------------
        # 8. QUERY-CONDITIONED MATCHING
        #
        # EXACT:
        #   v + q
        #   then LayerNorm
        #
        # There is NO GELU here.
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
        # 11. HEATMAP-WEIGHTED SPATIAL CONTEXT
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

        spatial_context = self.spatial_summary(
            spatial_context
        )

        # ----------------------------------------------------
        # 12. HEATMAP -> GEOMETRY
        # ----------------------------------------------------

        geometry, base_box = (
            self.extract_spatial_geometry(
                heatmap
            )
        )

        # ----------------------------------------------------
        # 13. CORRECTION INPUT
        #
        # 256 + 256 + 8 = 520
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
        # 14. LEARNED CORRECTION
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
# CHECKPOINT
# ============================================================

print()
print("=" * 80)
print("LOADING C1 CHECKPOINT")
print("=" * 80)

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

print(
    "Checkpoint keys:",
    list(
        checkpoint.keys()
    )
)

print(
    "Epoch:",
    checkpoint.get(
        "epoch",
        "unknown",
    )
)

print(
    "Stored validation IoU:",
    checkpoint.get(
        "val_iou",
        "unknown",
    )
)

print(
    "Stored config:",
    checkpoint.get(
        "config",
        "unknown",
    )
)


# ============================================================
# MODEL
# ============================================================

model = ExactC1GroundingHeatmapModel()

state_dict = checkpoint[
    "model_state_dict"
]

missing, unexpected = (
    model.load_state_dict(
        state_dict,
        strict=False,
    )
)

print()
print(
    "Missing keys:",
    len(missing)
)

print(
    "Unexpected keys:",
    len(unexpected)
)

if missing:
    print()
    print("Missing:")
    for key in missing:
        print("  ", key)

if unexpected:
    print()
    print("Unexpected:")
    for key in unexpected:
        print("  ", key)

if missing or unexpected:
    raise RuntimeError(
        "Checkpoint does not exactly match "
        "the reconstructed C1 architecture."
    )

model.to(
    DEVICE
)

model.eval()

print(
    "\nExact C1 checkpoint loaded successfully."
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
# EXACT CACHE INITIALIZATION
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
# EXACT VALIDATION DATASET
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


# ============================================================
# EXACT COLLATE / LOADER
# ============================================================

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
# PREP OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

records = []


# ============================================================
# DIAGNOSTIC INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING EXACT C1 DIAGNOSTIC")
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

        final_boxes = outputs[
            "box"
        ]

        corrections = outputs[
            "correction"
        ]

        B = target_boxes.size(0)

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        final_ious = box_iou(
            final_boxes,
            target_boxes,
        )

        base_ious = box_iou(
            base_boxes,
            target_boxes,
        )

        # ----------------------------------------------------
        # HEATMAP PEAK
        # ----------------------------------------------------

        flat = heatmaps.reshape(
            B,
            -1,
        )

        peak_indices = flat.argmax(
            dim=1,
        )

        peak_y = (
            peak_indices // 56
        )

        peak_x = (
            peak_indices % 56
        )

        peak_x = (
            peak_x.float()
            + 0.5
        ) / 56.0

        peak_y = (
            peak_y.float()
            + 0.5
        ) / 56.0

        peak_inside = (
            (peak_x >= target_boxes[:, 0])
            &
            (peak_x <= target_boxes[:, 2])
            &
            (peak_y >= target_boxes[:, 1])
            &
            (peak_y <= target_boxes[:, 3])
        )

        # ----------------------------------------------------
        # HEATMAP MASS INSIDE GT BOX
        # ----------------------------------------------------

        xs = (
            torch.arange(
                56,
                device=DEVICE,
                dtype=heatmaps.dtype,
            )
            + 0.5
        ) / 56.0

        ys = (
            torch.arange(
                56,
                device=DEVICE,
                dtype=heatmaps.dtype,
            )
            + 0.5
        ) / 56.0

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        gt_x1 = target_boxes[
            :,
            0,
        ].view(
            B,
            1,
            1,
        )

        gt_y1 = target_boxes[
            :,
            1,
        ].view(
            B,
            1,
            1,
        )

        gt_x2 = target_boxes[
            :,
            2,
        ].view(
            B,
            1,
            1,
        )

        gt_y2 = target_boxes[
            :,
            3,
        ].view(
            B,
            1,
            1,
        )

        gt_mask = (
            (xx >= gt_x1)
            &
            (xx <= gt_x2)
            &
            (yy >= gt_y1)
            &
            (yy <= gt_y2)
        )

        heatmap_mass_inside = (
            heatmaps
            *
            gt_mask.float()
        ).sum(
            dim=(1, 2),
        )

        # ----------------------------------------------------
        # TARGET AREA
        # ----------------------------------------------------

        target_area = (
            (
                target_boxes[:, 2]
                -
                target_boxes[:, 0]
            )
            *
            (
                target_boxes[:, 3]
                -
                target_boxes[:, 1]
            )
        )

        # ----------------------------------------------------
        # HEATMAP ENTROPY
        # ----------------------------------------------------

        positive = heatmaps.clamp_min(
            1e-12
        )

        entropy = -(
            positive
            *
            torch.log(
                positive
            )
        ).sum(
            dim=(1, 2)
        )

        # ----------------------------------------------------
        # SAVE PER-SAMPLE DATA
        # ----------------------------------------------------

        for i in range(B):

            target = (
                target_boxes[i]
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            base = (
                base_boxes[i]
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            final = (
                final_boxes[i]
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

            corr = (
                corrections[i]
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

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

                "target_x1":
                    float(
                        target[0]
                    ),

                "target_y1":
                    float(
                        target[1]
                    ),

                "target_x2":
                    float(
                        target[2]
                    ),

                "target_y2":
                    float(
                        target[3]
                    ),

                "target_area":
                    float(
                        target_area[i]
                        .item()
                    ),

                "base_x1":
                    float(
                        base[0]
                    ),

                "base_y1":
                    float(
                        base[1]
                    ),

                "base_x2":
                    float(
                        base[2]
                    ),

                "base_y2":
                    float(
                        base[3]
                    ),

                "final_x1":
                    float(
                        final[0]
                    ),

                "final_y1":
                    float(
                        final[1]
                    ),

                "final_x2":
                    float(
                        final[2]
                    ),

                "final_y2":
                    float(
                        final[3]
                    ),

                "base_iou":
                    float(
                        base_ious[i]
                        .item()
                    ),

                "final_iou":
                    float(
                        final_ious[i]
                        .item()
                    ),

                "iou_change":
                    float(
                        (
                            final_ious[i]
                            -
                            base_ious[i]
                        ).item()
                    ),

                "correction_dx1":
                    float(
                        corr[0]
                    ),

                "correction_dy1":
                    float(
                        corr[1]
                    ),

                "correction_dx2":
                    float(
                        corr[2]
                    ),

                "correction_dy2":
                    float(
                        corr[3]
                    ),

                "correction_abs_mean":
                    float(
                        np.abs(
                            corr
                        ).mean()
                    ),

                "heatmap_peak_x":
                    float(
                        peak_x[i]
                        .item()
                    ),

                "heatmap_peak_y":
                    float(
                        peak_y[i]
                        .item()
                    ),

                "heatmap_peak_inside_gt":
                    int(
                        peak_inside[i]
                        .item()
                    ),

                "heatmap_mass_inside_gt":
                    float(
                        heatmap_mass_inside[i]
                        .item()
                    ),

                "heatmap_entropy":
                    float(
                        entropy[i]
                        .item()
                    ),
            }

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
        f"Expected {len(val_dataset)} "
        f"records but got {len(df)}."
    )


# ============================================================
# AGGREGATE METRICS
# ============================================================

final_iou = df[
    "final_iou"
].to_numpy(
    dtype=np.float64
)

base_iou = df[
    "base_iou"
].to_numpy(
    dtype=np.float64
)

iou_change = df[
    "iou_change"
].to_numpy(
    dtype=np.float64
)

peak_inside = df[
    "heatmap_peak_inside_gt"
].to_numpy(
    dtype=np.float64
)

mass_inside = df[
    "heatmap_mass_inside_gt"
].to_numpy(
    dtype=np.float64
)

target_area = df[
    "target_area"
].to_numpy(
    dtype=np.float64
)

correction_mag = df[
    "correction_abs_mean"
].to_numpy(
    dtype=np.float64
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "num_examples":
        int(
            len(df)
        ),

    "stored_checkpoint_val_iou":
        float(
            checkpoint.get(
                "val_iou",
                float("nan"),
            )
        ),

    "recomputed_final_mean_iou":
        float(
            final_iou.mean()
        ),

    "recomputed_final_median_iou":
        float(
            np.median(
                final_iou
            )
        ),

    "recomputed_final_iou_ge_025":
        float(
            (
                final_iou >= 0.25
            ).mean()
        ),

    "recomputed_final_iou_ge_050":
        float(
            (
                final_iou >= 0.50
            ).mean()
        ),

    "recomputed_final_iou_ge_075":
        float(
            (
                final_iou >= 0.75
            ).mean()
        ),

    "base_mean_iou":
        float(
            base_iou.mean()
        ),

    "base_median_iou":
        float(
            np.median(
                base_iou
            )
        ),

    "base_iou_ge_025":
        float(
            (
                base_iou >= 0.25
            ).mean()
        ),

    "base_iou_ge_050":
        float(
            (
                base_iou >= 0.50
            ).mean()
        ),

    "mean_iou_change_from_correction":
        float(
            iou_change.mean()
        ),

    "correction_improves_pct":
        float(
            (
                iou_change > 1e-8
            ).mean()
            * 100.0
        ),

    "correction_hurts_pct":
        float(
            (
                iou_change < -1e-8
            ).mean()
            * 100.0
        ),

    "correction_unchanged_pct":
        float(
            (
                np.abs(iou_change) <= 1e-8
            ).mean()
            * 100.0
        ),

    "heatmap_peak_inside_gt_pct":
        float(
            peak_inside.mean()
            * 100.0
        ),

    "mean_heatmap_mass_inside_gt":
        float(
            mass_inside.mean()
        ),

    "median_heatmap_mass_inside_gt":
        float(
            np.median(
                mass_inside
            )
        ),

    "mean_heatmap_entropy":
        float(
            df[
                "heatmap_entropy"
            ].mean()
        ),

    "mean_correction_abs":
        float(
            correction_mag.mean()
        ),

    "median_correction_abs":
        float(
            np.median(
                correction_mag
            )
        ),

    "mean_target_area":
        float(
            target_area.mean()
        ),

    "median_target_area":
        float(
            np.median(
                target_area
            )
        ),
}


# ============================================================
# SIZE-BIN ANALYSIS
#
# IMPORTANT:
# These are exactly the bins used in the actual C1
# training history.
# ============================================================

size_bins = [
    (
        "tiny_0.00_0.03",
        0.00,
        0.03,
    ),

    (
        "small_0.03_0.0672",
        0.03,
        0.0672,
    ),

    (
        "medium_0.0672_0.196",
        0.0672,
        0.196,
    ),

    (
        "large_0.196_0.4148",
        0.196,
        0.4148,
    ),

    (
        "very_large_0.4148_0.62",
        0.4148,
        0.62,
    ),

    (
        "huge_0.62_1.00",
        0.62,
        1.0,
    ),
]

size_rows = []

for name, low, high in size_bins:

    mask = (
        (target_area >= low)
        &
        (target_area < high)
    )

    if not mask.any():
        continue

    size_rows.append(
        {
            "size_bin":
                name,

            "n":
                int(
                    mask.sum()
                ),

            "mean_target_area":
                float(
                    target_area[mask].mean()
                ),

            "base_mean_iou":
                float(
                    base_iou[mask].mean()
                ),

            "final_mean_iou":
                float(
                    final_iou[mask].mean()
                ),

            "mean_iou_change":
                float(
                    iou_change[mask].mean()
                ),

            "final_iou_ge_025_pct":
                float(
                    (
                        final_iou[mask]
                        >= 0.25
                    ).mean()
                    * 100.0
                ),

            "final_iou_ge_050_pct":
                float(
                    (
                        final_iou[mask]
                        >= 0.50
                    ).mean()
                    * 100.0
                ),

            "heatmap_peak_inside_pct":
                float(
                    peak_inside[mask].mean()
                    * 100.0
                ),

            "heatmap_mass_inside_mean":
                float(
                    mass_inside[mask].mean()
                ),
        }
    )


size_df = pd.DataFrame(
    size_rows
)


# ============================================================
# SAVE
# ============================================================

per_sample_path = os.path.join(
    OUTPUT_DIR,
    "exact_c1_per_sample.csv",
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "exact_c1_summary.json",
)

size_path = os.path.join(
    OUTPUT_DIR,
    "exact_c1_size_summary.csv",
)

df.to_csv(
    per_sample_path,
    index=False,
)

size_df.to_csv(
    size_path,
    index=False,
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 80)
print("EXACT C1 DIAGNOSTIC RESULTS")
print("=" * 80)

print()
print(
    f"Validation examples: "
    f"{len(df):,}"
)

print()
print("CHECKPOINT")

print(
    f"  Stored Val IoU:       "
    f"{summary['stored_checkpoint_val_iou']:.9f}"
)

print()
print("FINAL BOX")

print(
    f"  Mean IoU:             "
    f"{summary['recomputed_final_mean_iou']:.9f}"
)

print(
    f"  Median IoU:           "
    f"{summary['recomputed_final_median_iou']:.9f}"
)

print(
    f"  IoU >= 0.25:          "
    f"{summary['recomputed_final_iou_ge_025'] * 100:.2f}%"
)

print(
    f"  IoU >= 0.50:          "
    f"{summary['recomputed_final_iou_ge_050'] * 100:.2f}%"
)

print(
    f"  IoU >= 0.75:          "
    f"{summary['recomputed_final_iou_ge_075'] * 100:.2f}%"
)

print()
print("BASE BOX")

print(
    f"  Mean IoU:             "
    f"{summary['base_mean_iou']:.9f}"
)

print(
    f"  Median IoU:           "
    f"{summary['base_median_iou']:.9f}"
)

print(
    f"  IoU >= 0.25:          "
    f"{summary['base_iou_ge_025'] * 100:.2f}%"
)

print(
    f"  IoU >= 0.50:          "
    f"{summary['base_iou_ge_050'] * 100:.2f}%"
)

print()
print("CORRECTION HEAD")

print(
    f"  Mean IoU change:      "
    f"{summary['mean_iou_change_from_correction']:+.9f}"
)

print(
    f"  Improved:             "
    f"{summary['correction_improves_pct']:.2f}%"
)

print(
    f"  Hurt:                 "
    f"{summary['correction_hurts_pct']:.2f}%"
)

print(
    f"  Unchanged:            "
    f"{summary['correction_unchanged_pct']:.2f}%"
)

print(
    f"  Mean abs correction:  "
    f"{summary['mean_correction_abs']:.6f}"
)

print(
    f"  Median abs correction:"
    f" {summary['median_correction_abs']:.6f}"
)

print()
print("HEATMAP")

print(
    f"  Peak inside GT:       "
    f"{summary['heatmap_peak_inside_gt_pct']:.2f}%"
)

print(
    f"  Mean mass inside GT:  "
    f"{summary['mean_heatmap_mass_inside_gt']:.6f}"
)

print(
    f"  Median mass inside:   "
    f"{summary['median_heatmap_mass_inside_gt']:.6f}"
)

print(
    f"  Mean entropy:         "
    f"{summary['mean_heatmap_entropy']:.6f}"
)

print()
print("TARGET SIZE")

print(
    f"  Mean area:            "
    f"{summary['mean_target_area']:.6f}"
)

print(
    f"  Median area:          "
    f"{summary['median_target_area']:.6f}"
)


print()
print("=" * 80)
print("SIZE-BIN ANALYSIS")
print("=" * 80)

if len(size_df) > 0:

    print(
        size_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"\nPer-sample:\n"
    f"{per_sample_path}"
)

print(
    f"\nSummary:\n"
    f"{summary_path}"
)

print(
    f"\nSize summary:\n"
    f"{size_path}"
)

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    "\nCheckpoint was only read."
)

print(
    "No training was performed."
)

print(
    "No checkpoint was modified."
)