import os
import math
import numpy as np
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

CHECKPOINT = r"D:\SatQueryAI\models\checkpoints\c2_heatmap_only\best.pt"

DOFA_ROOT = r"D:\SatQueryAI\data\processed\dofa_features_download\dofa_features"

TEXT_ROOT = (
    r"D:\SatQueryAI\data\processed\grounding_text_embeddings"
    r"\grounding_text_embedding_shards"
)

VAL_PARQUET = (
    r"D:\SatQueryAI\data\processed\reference_grounding_split"
    r"\reference_val.parquet"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 80)
print("C2 LOCAL EXACT REPRODUCTION")
print("=" * 80)

print("Device:", device)
print("Checkpoint:", CHECKPOINT)


# ============================================================
# CHECK FILES
# ============================================================

for name, path in [
    ("Checkpoint", CHECKPOINT),
    ("DOFA cache", DOFA_ROOT),
    ("Text cache", TEXT_ROOT),
    ("Validation parquet", VAL_PARQUET),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name}: FOUND")


# ============================================================
# EXACT C2 MODEL
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
# HEATMAP -> BOX
# ============================================================

def heatmap_moments(
    heatmap,
):
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

    xx = xx.reshape(
        1,
        -1
    )

    yy = yy.reshape(
        1,
        -1
    )

    p = heatmap.reshape(
        B,
        -1
    )

    cx = (
        p * xx
    ).sum(
        dim=1
    )

    cy = (
        p * yy
    ).sum(
        dim=1
    )

    var_x = (
        p
        *
        (
            xx
            -
            cx.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )

    var_y = (
        p
        *
        (
            yy
            -
            cy.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )

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


def heatmap_to_box(
    heatmap,
):

    cx, cy, std_x, std_y = heatmap_moments(
        heatmap
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
# IOU
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
# LOAD CACHES
# ============================================================

print()
print("=" * 80)
print("LOADING CACHES")
print("=" * 80)

dofa_cache = DofaShardCache(
    DOFA_ROOT
)

text_cache = GroundingTextCache(
    TEXT_ROOT
)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING VALIDATION DATASET")
print("=" * 80)

val_dataset = ReferenceGroundingDataset(
    VAL_PARQUET,
    dofa_cache,
    text_cache,
)

print(
    "Validation rows:",
    len(val_dataset)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_reference_batch,
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 80)
print("LOADING C2 CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
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

model.to(device)
model.eval()

print()
print(
    "C2 model loaded successfully."
)

print(
    "Parameter count:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# EXACT VALIDATION
# ============================================================

print()
print("=" * 80)
print("RUNNING EXACT C2 VALIDATION")
print("=" * 80)

ious = []

all_gt_masses = []
all_centroid_errors = []
all_spread_errors = []

with torch.no_grad():

    for batch_idx, batch in enumerate(
        val_loader,
        start=1
    ):

        s1 = batch[
            "s1"
        ].to(device)

        s2 = batch[
            "s2"
        ].to(device)

        text_embedding = batch[
            "text_embedding"
        ].to(device)

        target_box = batch[
            "target_box"
        ].to(device)


        outputs = model(
            s1,
            s2,
            text_embedding
        )

        heatmap = outputs[
            "heatmap"
        ]

        predicted_box = (
            heatmap_to_box(
                heatmap
            )
        )

        batch_iou = box_iou(
            predicted_box,
            target_box
        )

        ious.extend(
            batch_iou.cpu().numpy()
        )


        # ----------------------------------------------------
        # MOMENT DIAGNOSTICS
        # ----------------------------------------------------

        (
            pred_cx,
            pred_cy,
            pred_std_x,
            pred_std_y
        ) = heatmap_moments(
            heatmap
        )


        # Target heatmap
        B = target_box.size(0)
        H = 56
        W = 56

        edges = torch.linspace(
            0.0,
            1.0,
            W + 1,
            device=device,
            dtype=target_box.dtype
        )

        x_left = edges[:-1].view(
            1, 1, W
        )

        x_right = edges[1:].view(
            1, 1, W
        )

        y_top = edges[:-1].view(
            1, H, 1
        )

        y_bottom = edges[1:].view(
            1, H, 1
        )

        x1 = target_box[:, 0].view(
            B, 1, 1
        )

        y1 = target_box[:, 1].view(
            B, 1, 1
        )

        x2 = target_box[:, 2].view(
            B, 1, 1
        )

        y2 = target_box[:, 3].view(
            B, 1, 1
        )

        inter_w = (
            torch.minimum(x_right, x2)
            -
            torch.maximum(x_left, x1)
        ).clamp_min(0.0)

        inter_h = (
            torch.minimum(y_bottom, y2)
            -
            torch.maximum(y_top, y1)
        ).clamp_min(0.0)

        area = (
            inter_w * inter_h
        )

        target_heatmap = (
            area
            /
            area.sum(
                dim=(1, 2),
                keepdim=True
            ).clamp_min(1e-8)
        )


        (
            target_cx,
            target_cy,
            target_std_x,
            target_std_y
        ) = heatmap_moments(
            target_heatmap
        )


        centroid_error = torch.sqrt(
            (
                pred_cx - target_cx
            ).pow(2)
            +
            (
                pred_cy - target_cy
            ).pow(2)
        )

        spread_error = torch.sqrt(
            (
                pred_std_x - target_std_x
            ).pow(2)
            +
            (
                pred_std_y - target_std_y
            ).pow(2)
        )


        all_centroid_errors.extend(
            centroid_error.cpu().numpy()
        )

        all_spread_errors.extend(
            spread_error.cpu().numpy()
        )


        # ----------------------------------------------------
        # GT MASS
        # ----------------------------------------------------

        xs = (
            torch.arange(
                W,
                device=device,
                dtype=heatmap.dtype
            ) + 0.5
        ) / W

        ys = (
            torch.arange(
                H,
                device=device,
                dtype=heatmap.dtype
            ) + 0.5
        ) / H

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij"
        )

        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        gt_mask = (
            (xx >= x1)
            &
            (xx <= x2)
            &
            (yy >= y1)
            &
            (yy <= y2)
        )

        gt_mass = (
            heatmap
            *
            gt_mask.float()
        ).sum(
            dim=(1, 2)
        )

        all_gt_masses.extend(
            gt_mass.cpu().numpy()
        )


        if (
            batch_idx % 20 == 0
        ):
            print(
                f"Processed "
                f"{batch_idx}/"
                f"{len(val_loader)} batches"
            )


# ============================================================
# RESULTS
# ============================================================

ious = np.asarray(
    ious,
    dtype=np.float64
)

gt_masses = np.asarray(
    all_gt_masses,
    dtype=np.float64
)

centroid_errors = np.asarray(
    all_centroid_errors,
    dtype=np.float64
)

spread_errors = np.asarray(
    all_spread_errors,
    dtype=np.float64
)


mean_iou = float(
    ious.mean()
)

median_iou = float(
    np.median(ious)
)

iou25 = float(
    (ious >= 0.25).mean()
    * 100.0
)

iou50 = float(
    (ious >= 0.50).mean()
    * 100.0
)

iou75 = float(
    (ious >= 0.75).mean()
    * 100.0
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("C2 LOCAL REPRODUCTION RESULTS")
print("=" * 80)

print(
    f"Mean IoU        : {mean_iou:.9f}"
)

print(
    f"Median IoU      : {median_iou:.9f}"
)

print(
    f"IoU >= 0.25     : {iou25:.2f}%"
)

print(
    f"IoU >= 0.50     : {iou50:.2f}%"
)

print(
    f"IoU >= 0.75     : {iou75:.2f}%"
)

print()

print(
    f"Mean GT mass    : {gt_masses.mean():.9f}"
)

print(
    f"Mean centroid   : {centroid_errors.mean():.9f}"
)

print(
    f"Mean spread err : {spread_errors.mean():.9f}"
)

print()

print(
    "Kaggle stored IoU:",
    checkpoint.get("val_iou")
)

print(
    "Absolute difference:",
    abs(
        mean_iou
        -
        float(
            checkpoint.get(
                "val_iou"
            )
        )
    )
)

print()
print("=" * 80)