# Extracted from Kaggle notebook: notebook8dfff136e6(2).ipynb
# Cells 7-9 combined: alignment, V1 model/sanity setup, and V1 training.
# This preserves the training lineage for the frozen single-modality grounding V1 checkpoints.
# Kaggle paths are intentionally preserved for now.

# ============================================================
# SOURCE CELL 7
# ============================================================

# ============================================================
# SATQUERYAI — SINGLE-MODALITY GROUNDING DATA ALIGNMENT
# ============================================================

from pathlib import Path
import re
import ast
import torch
import pandas as pd

ROOT = Path(
    "/kaggle/input/datasets/rajdeepbiswas1/"
    "satqueryai-single-modality-training"
)

DOFA_DIR = ROOT / "dofa_features"

TRAIN_PARQUET = (
    ROOT
    / "grounding"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    ROOT
    / "grounding"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

TEXT_DIR = (
    ROOT
    / "grounding"
    / "grounding_text_embedding_shards"
)

print("=" * 70)
print("SATQUERYAI — GROUNDING DATA ALIGNMENT CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Load splits
# ------------------------------------------------------------

train_df = pd.read_parquet(TRAIN_PARQUET)
val_df = pd.read_parquet(VAL_PARQUET)

print("\nReference split:")
print("Train rows:", len(train_df))
print("Val rows:", len(val_df))

print("\nRequired columns:")
required = [
    "ID",
    "patch_id",
    "input",
    "output",
    "original_row_index",
]

for c in required:
    print(
        f"{'PASS' if c in train_df.columns else 'FAIL'}  {c}"
    )

# ------------------------------------------------------------
# Split integrity
# ------------------------------------------------------------

train_patches = set(
    train_df["patch_id"].astype(str)
)

val_patches = set(
    val_df["patch_id"].astype(str)
)

print("\nPatch split integrity:")
print(
    "Train patches:",
    len(train_patches)
)

print(
    "Val patches:",
    len(val_patches)
)

print(
    "Patch overlap:",
    len(train_patches & val_patches)
)

# ------------------------------------------------------------
# DOFA patch IDs
# ------------------------------------------------------------

print("\nLoading DOFA patch index...")

dofa_patch_ids = []

for path in sorted(DOFA_DIR.glob("*.pt")):
    d = torch.load(
        path,
        map_location="cpu"
    )

    dofa_patch_ids.extend(
        [str(x) for x in d["patch_ids"]]
    )

dofa_patch_set = set(dofa_patch_ids)

print(
    "DOFA patches:",
    len(dofa_patch_set)
)

print(
    "Train missing DOFA patches:",
    len(train_patches - dofa_patch_set)
)

print(
    "Val missing DOFA patches:",
    len(val_patches - dofa_patch_set)
)

# ------------------------------------------------------------
# Grounding text cache
# ------------------------------------------------------------

print("\nLoading grounding text index...")

text_files = sorted(
    TEXT_DIR.glob("*.pt")
)

print(
    "Text shards:",
    len(text_files)
)

text_lookup = {}

for path in text_files:

    d = torch.load(
        path,
        map_location="cpu"
    )

    source_rows = d["source_row_index"]
    embeddings = d["text_embeddings"]

    for i in range(len(source_rows)):
        row_id = int(source_rows[i])

        if row_id in text_lookup:
            raise RuntimeError(
                f"Duplicate text-cache row: {row_id}"
            )

        text_lookup[row_id] = embeddings[i]

print(
    "Indexed text embeddings:",
    len(text_lookup)
)

# ------------------------------------------------------------
# Check text alignment
# ------------------------------------------------------------

train_row_ids = set(
    train_df["original_row_index"]
    .astype(int)
)

val_row_ids = set(
    val_df["original_row_index"]
    .astype(int)
)

print("\nText alignment:")

print(
    "Train rows missing text:",
    len(train_row_ids - set(text_lookup))
)

print(
    "Val rows missing text:",
    len(val_row_ids - set(text_lookup))
)

# ------------------------------------------------------------
# Bounding-box parser
# ------------------------------------------------------------

def parse_box(output):

    if output is None:
        return None

    text = str(output)

    patterns = [
        r"\[\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*,\s*"
        r"([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*\]",

        r"\[\s*([0-9eE+\-.]+)\s*,\s*"
        r"([0-9eE+\-.]+)\s*,\s*"
        r"([0-9eE+\-.]+)\s*,\s*"
        r"([0-9eE+\-.]+)\s*\]",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)

        if m:
            return [
                float(m.group(1)),
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            ]

    return None

# ------------------------------------------------------------
# Parse all boxes
# ------------------------------------------------------------

def inspect_boxes(df, name):

    parsed = []

    invalid = 0

    for value in df["output"]:

        box = parse_box(value)

        if box is None:
            invalid += 1
            continue

        x1, y1, x2, y2 = box

        valid = (
            0.0 <= x1 <= 1.0
            and 0.0 <= y1 <= 1.0
            and 0.0 <= x2 <= 1.0
            and 0.0 <= y2 <= 1.0
            and x2 > x1
            and y2 > y1
        )

        if valid:
            parsed.append(box)
        else:
            invalid += 1

    print(
        f"{name}: parsed={len(parsed)}, invalid={invalid}"
    )

    if parsed:
        t = torch.tensor(
            parsed,
            dtype=torch.float32
        )

        print(
            f"{name} box tensor:",
            tuple(t.shape)
        )

        print(
            f"{name} min:",
            t.min(dim=0).values.tolist()
        )

        print(
            f"{name} max:",
            t.max(dim=0).values.tolist()
        )

    return parsed

print("\nBounding-box validation:")

train_boxes = inspect_boxes(
    train_df,
    "Train"
)

val_boxes = inspect_boxes(
    val_df,
    "Val"
)

# ------------------------------------------------------------
# Real examples
# ------------------------------------------------------------

print("\nReal training examples:")

for i in range(min(5, len(train_df))):

    row = train_df.iloc[i]

    print("\nExample", i)
    print("ID:", row["ID"])
    print("Patch:", row["patch_id"])
    print("Row index:", row["original_row_index"])
    print("Input:", row["input"])
    print("Output:", row["output"])
    print("Parsed box:", parse_box(row["output"]))

    text_embedding = text_lookup[
        int(row["original_row_index"])
    ]

    print(
        "Text embedding:",
        tuple(text_embedding.shape),
        text_embedding.dtype
    )

# ------------------------------------------------------------
# Final assertions
# ------------------------------------------------------------

assert len(train_boxes) == len(train_df)
assert len(val_boxes) == len(val_df)

assert len(train_patches & val_patches) == 0

assert len(train_row_ids - set(text_lookup)) == 0
assert len(val_row_ids - set(text_lookup)) == 0

assert len(train_patches - dofa_patch_set) == 0
assert len(val_patches - dofa_patch_set) == 0

print("\n" + "=" * 70)
print("GROUNDING DATA ALIGNMENT: PASS")
print("=" * 70)

# ============================================================
# SOURCE CELL 8
# ============================================================

# ============================================================
# SATQUERYAI — SINGLE-MODALITY GROUNDING SANITY TEST
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# BOX -> 56x56 NORMALIZED AREA HEATMAP
# ============================================================

def boxes_to_heatmaps(
    boxes,
    height=56,
    width=56,
):
    """
    Convert normalized [x1,y1,x2,y2] boxes into
    normalized area-overlap heatmaps.

    Each cell receives the area of intersection
    between the cell and the target box.
    """

    device = boxes.device
    dtype = boxes.dtype

    batch_size = boxes.shape[0]

    x1 = boxes[:, 0].view(-1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1)

    ys = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / height

    xs = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / width

    cy = ys.view(1, height, 1)
    cx = xs.view(1, 1, width)

    cell_x1 = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        )
        / width
    ).view(1, 1, width)

    cell_x2 = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 1
    ).view(1, 1, width) / width

    cell_y1 = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        )
        / height
    ).view(1, height, 1)

    cell_y2 = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 1
    ).view(1, height, 1) / height

    overlap_w = (
        torch.minimum(x2, cell_x2)
        - torch.maximum(x1, cell_x1)
    ).clamp_min(0.0)

    overlap_h = (
        torch.minimum(y2, cell_y2)
        - torch.maximum(y1, cell_y1)
    ).clamp_min(0.0)

    heatmap = overlap_w * overlap_h

    heatmap = heatmap.reshape(
        batch_size,
        height,
        width
    )

    heatmap_sum = (
        heatmap.sum(dim=(1, 2), keepdim=True)
        .clamp_min(1e-8)
    )

    return heatmap / heatmap_sum


# ============================================================
# HEATMAP MOMENTS
# ============================================================

def heatmap_moments(
    heatmap,
):
    """
    Differentiable centroid and spread.

    Returns:
        cx, cy, std_x, std_y
    """

    batch_size, height, width = heatmap.shape

    device = heatmap.device
    dtype = heatmap.dtype

    ys = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / height

    xs = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / width

    yy = ys.view(1, height, 1)
    xx = xs.view(1, 1, width)

    mass = heatmap.sum(
        dim=(1, 2)
    ).clamp_min(1e-8)

    cx = (
        heatmap * xx
    ).sum(
        dim=(1, 2)
    ) / mass

    cy = (
        heatmap * yy
    ).sum(
        dim=(1, 2)
    ) / mass

    variance_x = (
        heatmap
        * (xx - cx.view(-1, 1, 1)) ** 2
    ).sum(
        dim=(1, 2)
    ) / mass

    variance_y = (
        heatmap
        * (yy - cy.view(-1, 1, 1)) ** 2
    ).sum(
        dim=(1, 2)
    ) / mass

    std_x = torch.sqrt(
        variance_x + 1e-8
    )

    std_y = torch.sqrt(
        variance_y + 1e-8
    )

    return cx, cy, std_x, std_y


def heatmap_to_box(
    heatmap,
):
    """
    Convert heatmap moments into a normalized
    bounding box.

    For a roughly uniform rectangle:
        std = width / sqrt(12)
    """

    cx, cy, std_x, std_y = heatmap_moments(
        heatmap
    )

    width = (
        std_x * math.sqrt(12.0)
    ).clamp(
        min=0.02,
        max=1.0
    )

    height = (
        std_y * math.sqrt(12.0)
    ).clamp(
        min=0.02,
        max=1.0
    )

    x1 = (
        cx - width / 2.0
    ).clamp(0.0, 1.0)

    y1 = (
        cy - height / 2.0
    ).clamp(0.0, 1.0)

    x2 = (
        cx + width / 2.0
    ).clamp(0.0, 1.0)

    y2 = (
        cy + height / 2.0
    ).clamp(0.0, 1.0)

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1
    )


# ============================================================
# SINGLE-MODALITY GROUNDING MODEL
# ============================================================

class SingleModalityGrounding(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
        highres_grid=56,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.highres_grid = highres_grid
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Visual projection
        # ----------------------------------------------------

        self.visual_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Learnable spatial encoding
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.randn(
                1,
                grid_size * grid_size,
                hidden_dim
            ) * 0.02
        )

        # ----------------------------------------------------
        # Query-conditioned token representation
        # ----------------------------------------------------

        self.match = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Coarse 14x14 localization score
        # ----------------------------------------------------

        self.coarse_head = nn.Linear(
            hidden_dim,
            1
        )

        # ----------------------------------------------------
        # High-resolution decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            ),
        )

        # ----------------------------------------------------
        # Global query-conditioned refinement
        # ----------------------------------------------------

        self.refinement = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        visual,
        text,
    ):

        if visual.ndim != 3:
            raise ValueError(
                f"Expected visual [B,196,768], "
                f"got {tuple(visual.shape)}"
            )

        if text.ndim != 2:
            raise ValueError(
                f"Expected text [B,768], "
                f"got {tuple(text.shape)}"
            )

        b = visual.shape[0]

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        v = self.visual_proj(
            visual.float()
        )

        q = self.query_proj(
            text.float()
        )

        v = v + self.spatial_pos

        # ----------------------------------------------------
        # Question-conditioned matching
        # ----------------------------------------------------

        q_expanded = q.unsqueeze(1).expand(
            -1,
            v.shape[1],
            -1
        )

        combined = torch.cat(
            [v, q_expanded],
            dim=-1
        )

        conditioned = self.match(
            combined
        )

        # ----------------------------------------------------
        # 14x14 coarse feature map
        # ----------------------------------------------------

        coarse_logits = self.coarse_head(
            conditioned
        ).squeeze(-1)

        coarse_map = conditioned.transpose(
            1,
            2
        ).reshape(
            b,
            self.hidden_dim,
            self.grid_size,
            self.grid_size
        )

        # ----------------------------------------------------
        # High-resolution heatmap
        # ----------------------------------------------------

        heatmap_logits = self.decoder(
            coarse_map
        ).squeeze(1)

        # ----------------------------------------------------
        # Query-conditioned refinement
        # ----------------------------------------------------

        global_visual = conditioned.mean(
            dim=1
        )

        refinement = self.refinement(
            torch.cat(
                [global_visual, q],
                dim=-1
            )
        )

        heatmap_logits = (
            heatmap_logits
            + refinement.mean(
                dim=-1
            ).view(
                b, 1, 1
            )
        )

        heatmap = F.softmax(
            heatmap_logits.flatten(1),
            dim=1
        ).reshape(
            b,
            self.highres_grid,
            self.highres_grid
        )

        predicted_box = heatmap_to_box(
            heatmap
        )

        cx, cy, std_x, std_y = heatmap_moments(
            heatmap
        )

        return {
            "coarse_logits": coarse_logits,
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "cx": cx,
            "cy": cy,
            "std_x": std_x,
            "std_y": std_y,
            "predicted_box": predicted_box,
        }


# ============================================================
# LOSS
# ============================================================

def grounding_loss(
    outputs,
    target_boxes,
):
    target_heatmap = boxes_to_heatmaps(
        target_boxes,
        height=56,
        width=56
    )

    predicted_heatmap = outputs[
        "heatmap"
    ].clamp_min(1e-8)

    target_heatmap = target_heatmap.clamp_min(
        1e-8
    )

    # --------------------------------------------------------
    # Heatmap distribution loss
    # --------------------------------------------------------

    heatmap_loss = F.kl_div(
        predicted_heatmap.log(),
        target_heatmap,
        reduction="batchmean"
    )

    # --------------------------------------------------------
    # Target moments
    # --------------------------------------------------------

    target_cx, target_cy, target_std_x, target_std_y = (
        heatmap_moments(
            target_heatmap
        )
    )

    target_moments = torch.stack(
        [
            target_cx,
            target_cy,
            target_std_x,
            target_std_y,
        ],
        dim=-1
    )

    predicted_moments = torch.stack(
        [
            outputs["cx"],
            outputs["cy"],
            outputs["std_x"],
            outputs["std_y"],
        ],
        dim=-1
    )

    moment_loss = F.smooth_l1_loss(
        predicted_moments,
        target_moments
    )

    total = (
        heatmap_loss
        + 10.0 * moment_loss
    )

    return {
        "total": total,
        "heatmap": heatmap_loss,
        "moment": moment_loss,
        "target_heatmap": target_heatmap,
    }


# ============================================================
# IoU
# ============================================================

@torch.no_grad()
def box_iou(
    boxes1,
    boxes2,
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

    iw = (
        x2 - x1
    ).clamp_min(0.0)

    ih = (
        y2 - y1
    ).clamp_min(0.0)

    intersection = iw * ih

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0])
        .clamp_min(0.0)
        *
        (boxes1[:, 3] - boxes1[:, 1])
        .clamp_min(0.0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0])
        .clamp_min(0.0)
        *
        (boxes2[:, 3] - boxes2[:, 1])
        .clamp_min(0.0)
    )

    union = (
        area1
        + area2
        - intersection
    ).clamp_min(1e-8)

    return intersection / union


# ============================================================
# GROUNDING DATASET
# ============================================================

class SingleModalityGroundingDataset(
    Dataset
):

    def __init__(
        self,
        df,
        visual_features,
        text_embeddings,
        text_lookup,
        patch_lookup,
    ):

        self.visual_features = visual_features
        self.text_embeddings = text_embeddings

        patch_indices = []
        text_indices = []
        boxes = []

        for _, row in df.iterrows():

            patch_id = str(
                row["patch_id"]
            )

            row_index = int(
                row["original_row_index"]
            )

            if patch_id not in patch_lookup:
                raise RuntimeError(
                    f"Missing DOFA patch: {patch_id}"
                )

            if row_index not in text_lookup:
                raise RuntimeError(
                    f"Missing text row: {row_index}"
                )

            output = str(
                row["output"]
            )

            values = re.findall(
                r"[-+]?(?:\d*\.\d+|\d+\.?)",
                output
            )

            if len(values) < 4:
                raise RuntimeError(
                    f"Cannot parse box: {output}"
                )

            x1, y1, x2, y2 = map(
                float,
                values[:4]
            )

            patch_indices.append(
                patch_lookup[patch_id]
            )

            text_indices.append(
                text_lookup[row_index]
            )

            boxes.append(
                [x1, y1, x2, y2]
            )

        self.patch_indices = torch.tensor(
            patch_indices,
            dtype=torch.long
        )

        self.text_indices = torch.tensor(
            text_indices,
            dtype=torch.long
        )

        self.boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):

        return (
            self.visual_features[
                self.patch_indices[idx]
            ],
            self.text_embeddings[
                self.text_indices[idx]
            ],
            self.boxes[idx]
        )


# ============================================================
# BUILD TEXT INDEX
# ============================================================

print("=" * 70)
print("BUILDING GROUNDING SANITY DATA")
print("=" * 70)

import re

text_files = sorted(
    TEXT_DIR.glob("*.pt")
)

text_lookup = {}
text_vectors = []

for path in text_files:

    d = torch.load(
        path,
        map_location="cpu"
    )

    source_rows = d[
        "source_row_index"
    ]

    embeddings = d[
        "text_embeddings"
    ].float()

    start = len(text_vectors)

    text_vectors.append(
        embeddings
    )

    for i, row_id in enumerate(
        source_rows
    ):
        text_lookup[
            int(row_id)
        ] = start + i

text_embeddings = torch.cat(
    text_vectors,
    dim=0
)

print(
    "Text embeddings:",
    tuple(text_embeddings.shape)
)

# ------------------------------------------------------------
# DOFA lookup
# ------------------------------------------------------------

dofa_patch_ids = []

for path in sorted(
    DOFA_DIR.glob("*.pt")
):

    d = torch.load(
        path,
        map_location="cpu"
    )

    dofa_patch_ids.extend(
        [str(x) for x in d["patch_ids"]]
    )

patch_lookup = {
    patch_id: i
    for i, patch_id in enumerate(
        dofa_patch_ids
    )
}

print(
    "DOFA patches:",
    len(patch_lookup)
)

# ============================================================
# DATASET
# ============================================================

train_grounding_dataset = (
    SingleModalityGroundingDataset(
        train_df,
        dofa_s1,
        text_embeddings,
        text_lookup,
        patch_lookup
    )
)

val_grounding_dataset = (
    SingleModalityGroundingDataset(
        val_df,
        dofa_s1,
        text_embeddings,
        text_lookup,
        patch_lookup
    )
)

train_loader = DataLoader(
    train_grounding_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_grounding_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# ============================================================
# S1 SANITY
# ============================================================

print("\nS1 grounding sanity...")

visual, text, boxes = next(
    iter(train_loader)
)

visual = visual.to(DEVICE)
text = text.to(DEVICE)
boxes = boxes.to(DEVICE)

model = SingleModalityGrounding().to(
    DEVICE
)

outputs = model(
    visual,
    text
)

losses = grounding_loss(
    outputs,
    boxes
)

losses["total"].backward()

print(
    "S1 model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print(
    "S1 heatmap:",
    tuple(
        outputs["heatmap"].shape
    )
)

print(
    "S1 predicted box:",
    tuple(
        outputs["predicted_box"].shape
    )
)

print(
    "S1 total loss:",
    float(
        losses["total"]
        .detach()
        .cpu()
    )
)

print(
    "S1 backward:",
    "PASS"
)

del model, outputs, losses

torch.cuda.empty_cache()

# ============================================================
# S2 SANITY
# ============================================================

print("\nS2 grounding sanity...")

train_grounding_dataset_s2 = (
    SingleModalityGroundingDataset(
        train_df,
        dofa_s2,
        text_embeddings,
        text_lookup,
        patch_lookup
    )
)

loader_s2 = DataLoader(
    train_grounding_dataset_s2,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

visual, text, boxes = next(
    iter(loader_s2)
)

visual = visual.to(DEVICE)
text = text.to(DEVICE)
boxes = boxes.to(DEVICE)

model = SingleModalityGrounding().to(
    DEVICE
)

outputs = model(
    visual,
    text
)

losses = grounding_loss(
    outputs,
    boxes
)

losses["total"].backward()

print(
    "S2 heatmap:",
    tuple(
        outputs["heatmap"].shape
    )
)

print(
    "S2 predicted box:",
    tuple(
        outputs["predicted_box"].shape
    )
)

print(
    "S2 total loss:",
    float(
        losses["total"]
        .detach()
        .cpu()
    )
)

print(
    "S2 backward:",
    "PASS"
)

print("\n" + "=" * 70)
print("SINGLE-MODALITY GROUNDING SANITY: PASS")
print("=" * 70)

# ============================================================
# SOURCE CELL 9
# ============================================================

# ============================================================
# SATQUERYAI — TRAIN S1/S2 SINGLE-MODALITY GROUNDING
# EXACTLY 10 EPOCHS
# ============================================================

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
PIN_MEMORY = True

TRAIN_ROOT = Path(
    "/kaggle/working/satqueryai_single_modality_models"
)

TRAIN_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# DATA LOADERS
# ============================================================

def make_grounding_loaders(
    visual_features,
):

    train_dataset = SingleModalityGroundingDataset(
        train_df,
        visual_features,
        text_embeddings,
        text_lookup,
        patch_lookup,
    )

    val_dataset = SingleModalityGroundingDataset(
        val_df,
        visual_features,
        text_embeddings,
        text_lookup,
        patch_lookup,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_grounding(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    for visual, text, target_boxes in loader:

        visual = visual.to(
            DEVICE,
            non_blocking=True
        )

        text = text.to(
            DEVICE,
            non_blocking=True
        )

        target_boxes = target_boxes.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):

            outputs = model(
                visual,
                text
            )

            losses = grounding_loss(
                outputs,
                target_boxes
            )

        predicted_boxes = outputs[
            "predicted_box"
        ]

        batch_iou = box_iou(
            predicted_boxes.float(),
            target_boxes.float()
        )

        batch_size = target_boxes.size(0)

        total_loss += (
            losses["total"].float().item()
            * batch_size
        )

        total_iou += (
            batch_iou.sum().item()
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "iou": total_iou / total_samples,
    }


# ============================================================
# ONE TRAINING RUN
# ============================================================

def train_grounding_specialist(
    modality_name,
    visual_features,
):

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {modality_name} "
        f"SINGLE-MODALITY GROUNDING"
    )
    print("=" * 70)

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = make_grounding_loaders(
        visual_features
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    model = SingleModalityGrounding().to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    checkpoint_dir = (
        TRAIN_ROOT
        / modality_name.lower()
        / "grounding"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        checkpoint_dir
        / "best.pt"
    )

    last_checkpoint = (
        checkpoint_dir
        / "last.pt"
    )

    history_path = (
        checkpoint_dir
        / "training_history.json"
    )

    history = []

    best_val_iou = -1.0
    best_val_loss = float("inf")
    best_epoch = None

    total_start = time.time()

    print("\nStarting training...")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Device:", DEVICE)

    # ========================================================
    # EPOCH LOOP — ALWAYS ALL 10 EPOCHS
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        model.train()

        running_total = 0.0
        running_heatmap = 0.0
        running_moment = 0.0

        running_iou = 0.0
        running_samples = 0

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for (
            visual,
            text,
            target_boxes,
        ) in train_loader:

            visual = visual.to(
                DEVICE,
                non_blocking=True
            )

            text = text.to(
                DEVICE,
                non_blocking=True
            )

            target_boxes = target_boxes.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):

                outputs = model(
                    visual,
                    text
                )

                losses = grounding_loss(
                    outputs,
                    target_boxes
                )

                loss = losses["total"]

            scaler.scale(
                loss
            ).backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # Training metrics
            # ------------------------------------------------

            with torch.no_grad():

                predictions = (
                    outputs[
                        "predicted_box"
                    ]
                    .float()
                )

                batch_iou = box_iou(
                    predictions,
                    target_boxes.float()
                )

            batch_size = target_boxes.size(0)

            running_total += (
                losses["total"]
                .detach()
                .float()
                .item()
                * batch_size
            )

            running_heatmap += (
                losses["heatmap"]
                .detach()
                .float()
                .item()
                * batch_size
            )

            running_moment += (
                losses["moment"]
                .detach()
                .float()
                .item()
                * batch_size
            )

            running_iou += (
                batch_iou.sum().item()
            )

            running_samples += batch_size

        scheduler.step()

        train_total = (
            running_total
            / running_samples
        )

        train_heatmap = (
            running_heatmap
            / running_samples
        )

        train_moment = (
            running_moment
            / running_samples
        )

        train_iou = (
            running_iou
            / running_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation = evaluate_grounding(
            model,
            val_loader
        )

        val_loss = validation["loss"]
        val_iou = validation["iou"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss": train_total,
            "train_heatmap_loss": train_heatmap,
            "train_moment_loss": train_moment,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "epoch_time_sec": epoch_time,
        }

        history.append(
            record
        )

        # ----------------------------------------------------
        # Save LAST
        # ----------------------------------------------------

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "modality": modality_name,
            "model_type": (
                "single_modality_grounding"
            ),
            "parameter_count": total_parameters,
            "config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "input_dim": 768,
                "hidden_dim": 256,
                "grid_size": 14,
                "highres_grid": 56,
                "heatmap_loss": "KL",
                "moment_weight": 10.0,
            },
            "metrics": record,
        }

        torch.save(
            checkpoint_payload,
            last_checkpoint
        )

        # ----------------------------------------------------
        # Save BEST by validation IoU
        # ----------------------------------------------------

        is_better = (
            val_iou > best_val_iou
            or (
                val_iou == best_val_iou
                and val_loss < best_val_loss
            )
        )

        if is_better:

            best_val_iou = val_iou
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                checkpoint_payload,
                best_checkpoint
            )

            marker = (
                "  <-- NEW BEST"
            )

        else:
            marker = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_total:.4f} | "
            f"Train IoU {train_iou:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val IoU {val_iou:.4f} | "
            f"{epoch_time:.1f}s"
            f"{marker}"
        )

    # ========================================================
    # FINAL HISTORY
    # ========================================================

    total_time = (
        time.time()
        - total_start
    )

    history_payload = {
        "modality": modality_name,
        "model_type": (
            "single_modality_grounding"
        ),
        "epochs_completed": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "best_val_loss": best_val_loss,
        "parameter_count": total_parameters,
        "total_time_sec": total_time,
        "history": history,
    }

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_payload,
            f,
            indent=2
        )

    print("\nTraining complete:")
    print("Modality:", modality_name)
    print("Epochs completed:", EPOCHS)
    print("Best epoch:", best_epoch)
    print(
        "Best validation IoU:",
        f"{best_val_iou:.6f}"
    )
    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}"
    )
    print(
        "Best checkpoint:",
        best_checkpoint
    )
    print(
        "Total time:",
        f"{total_time:.1f} sec"
    )

    return history_payload


# ============================================================
# TRAIN S1
# ============================================================

s1_grounding_result = train_grounding_specialist(
    modality_name="S1",
    visual_features=dofa_s1,
)

torch.cuda.empty_cache()


# ============================================================
# TRAIN S2
# ============================================================

s2_grounding_result = train_grounding_specialist(
    modality_name="S2",
    visual_features=dofa_s2,
)

torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("S1/S2 SINGLE-MODALITY GROUNDING TRAINING FINISHED")
print("=" * 70)

print(
    "S1 best epoch:",
    s1_grounding_result["best_epoch"]
)

print(
    "S1 best val IoU:",
    f"{s1_grounding_result['best_val_iou']:.6f}"
)

print(
    "S2 best epoch:",
    s2_grounding_result["best_epoch"]
)

print(
    "S2 best val IoU:",
    f"{s2_grounding_result['best_val_iou']:.6f}"
)

print("\nSaved under:")
print(TRAIN_ROOT)

print(
    "\nBoth grounding specialists completed "
    "the full 10-epoch schedule."
)