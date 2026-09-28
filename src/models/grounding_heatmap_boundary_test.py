import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_heatmap_model import (
    GroundingHeatmapModel,
    box_iou,
)


# ============================================================
# Configuration
# ============================================================

NUM_EXAMPLES = 256
BATCH_SIZE = 32

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "grounding_heatmap_geometry"
    / "model_256_reference.pt"
)

QUANTILES = [
    (0.05, 0.95),
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
]


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):
    return {
        "s1": torch.stack(
            [x["s1"] for x in batch]
        ),
        "s2": torch.stack(
            [x["s2"] for x in batch]
        ),
        "text_embedding": torch.stack(
            [x["text_embedding"] for x in batch]
        ),
        "target_box": torch.stack(
            [x["target_box"] for x in batch]
        ),
    }


# ============================================================
# Convert a 1D discrete distribution to a continuous quantile
# ============================================================

def discrete_quantile(
    distribution,
    quantile,
    grid_size=14,
):
    """
    Approximate a quantile in normalized [0,1] coordinates.

    distribution:
        [B,14]

    Each bin represents one spatial cell.

    We locate the bin containing the requested cumulative
    probability, then linearly interpolate within that bin.

    This function is used only for evaluation/extraction;
    it is not part of gradient-based training.
    """

    batch_size = distribution.size(0)

    device = distribution.device
    dtype = distribution.dtype

    # --------------------------------------------------------
    # Normalize defensively.
    # --------------------------------------------------------

    distribution = distribution / (
        distribution.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    cdf = torch.cumsum(
        distribution,
        dim=1,
    )

    # --------------------------------------------------------
    # First bin where CDF >= requested quantile.
    # --------------------------------------------------------

    mask = (
        cdf >= quantile
    )

    indices = mask.float().argmax(
        dim=1
    )

    indices = indices.clamp(
        min=0,
        max=grid_size - 1,
    )

    # --------------------------------------------------------
    # CDF before selected bin.
    # --------------------------------------------------------

    previous_indices = (
        indices - 1
    ).clamp(
        min=0
    )

    previous_cdf = cdf[
        torch.arange(
            batch_size,
            device=device,
        ),
        previous_indices,
    ]

    # For the first bin there is no previous probability.
    previous_cdf = torch.where(
        indices == 0,
        torch.zeros_like(
            previous_cdf
        ),
        previous_cdf,
    )

    selected_probability = distribution[
        torch.arange(
            batch_size,
            device=device,
        ),
        indices,
    ]

    # --------------------------------------------------------
    # Fraction through selected cell.
    # --------------------------------------------------------

    fraction = (
        (
            quantile
            - previous_cdf
        )
        /
        (
            selected_probability
            + 1e-8
        )
    ).clamp(
        min=0.0,
        max=1.0,
    )

    # --------------------------------------------------------
    # Spatial coordinates.
    #
    # Cell i occupies:
    #
    #     [i/14, (i+1)/14]
    # --------------------------------------------------------

    start = (
        indices.float()
        / grid_size
    )

    end = (
        indices.float()
        + 1.0
    ) / grid_size

    coordinate = (
        start
        +
        fraction
        * (end - start)
    )

    return coordinate


# ============================================================
# Heatmap -> marginal distributions
# ============================================================

def heatmap_to_marginals(
    heatmap,
):
    """
    Convert [B,14,14] into:

        x_distribution: [B,14]
        y_distribution: [B,14]
    """

    x_distribution = heatmap.sum(
        dim=1
    )

    y_distribution = heatmap.sum(
        dim=2
    )

    x_distribution = (
        x_distribution
        /
        (
            x_distribution.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )
    )

    y_distribution = (
        y_distribution
        /
        (
            y_distribution.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )
    )

    return (
        x_distribution,
        y_distribution,
    )


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("BOUNDARY-AWARE HEATMAP BOX EXTRACTION")
print("=" * 70)

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

dataset = GroundingDataset()

reference_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "reference"
]

assert len(reference_indices) >= NUM_EXAMPLES

reference_subset = reference_indices[
    :NUM_EXAMPLES
]

loader = DataLoader(
    Subset(
        dataset,
        reference_subset,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

print(
    f"Examples: {NUM_EXAMPLES}"
)

print(
    f"Checkpoint: {CHECKPOINT}"
)


# ============================================================
# Load saved heatmap model
# ============================================================

model = GroundingHeatmapModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint[
        "model_state_dict"
    ]
else:
    state_dict = checkpoint

# ============================================================
# Load only heatmap-compatible weights
# ============================================================
#
# The checkpoint was trained with the previous geometry model,
# whose final regression module was named "box_head".
#
# This boundary-extraction experiment does NOT need the box
# refinement head. It only needs:
#
#   visual_fusion
#   query_proj
#   spatial_pos
#   visual_match
#   query_match
#   match_norm
#   match_head
#
# Therefore we load compatible parameters and ignore the old
# box-head parameters.
# ============================================================

current_state = model.state_dict()

compatible_state = {}

for key, value in state_dict.items():

    if key not in current_state:
        continue

    if current_state[key].shape != value.shape:
        continue

    compatible_state[key] = value

missing_keys = [
    key
    for key in current_state
    if key not in compatible_state
]

unexpected_keys = [
    key
    for key in state_dict
    if key not in compatible_state
]

model.load_state_dict(
    compatible_state,
    strict=False,
)

print(
    f"Loaded compatible parameters: "
    f"{len(compatible_state):,}"
)

print(
    f"Ignored checkpoint parameters: "
    f"{len(unexpected_keys):,}"
)

print(
    f"Uninitialized current-model parameters: "
    f"{len(missing_keys):,}"
)

for key in missing_keys:
    print(
        f"  [not loaded] {key}"
    )

model.eval()

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# Collect predictions
# ============================================================

all_heatmaps = []
all_targets = []

with torch.no_grad():

    for batch in loader:

        outputs = model(
            s1=batch["s1"],
            s2=batch["s2"],
            text_embedding=batch[
                "text_embedding"
            ],
        )

        all_heatmaps.append(
            outputs["heatmap"]
        )

        all_targets.append(
            batch["target_box"]
        )


heatmaps = torch.cat(
    all_heatmaps,
    dim=0,
)

targets = torch.cat(
    all_targets,
    dim=0,
)


# ============================================================
# Marginal distributions
# ============================================================

x_distribution, y_distribution = (
    heatmap_to_marginals(
        heatmaps
    )
)


# ============================================================
# Evaluate multiple boundary quantiles
# ============================================================

print()
print("=" * 70)
print("QUANTILE SEARCH")
print("=" * 70)

best_quantile = None
best_iou = -1.0

for lower_q, upper_q in QUANTILES:

    # --------------------------------------------------------
    # X boundaries
    # --------------------------------------------------------

    x1 = discrete_quantile(
        x_distribution,
        lower_q,
    )

    x2 = discrete_quantile(
        x_distribution,
        upper_q,
    )

    # --------------------------------------------------------
    # Y boundaries
    # --------------------------------------------------------

    y1 = discrete_quantile(
        y_distribution,
        lower_q,
    )

    y2 = discrete_quantile(
        y_distribution,
        upper_q,
    )

    # --------------------------------------------------------
    # Build box
    # --------------------------------------------------------

    predicted_box = torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1,
    )

    # Ensure ordering.
    predicted_box_x1 = torch.minimum(
        predicted_box[:, 0],
        predicted_box[:, 2],
    )

    predicted_box_y1 = torch.minimum(
        predicted_box[:, 1],
        predicted_box[:, 3],
    )

    predicted_box_x2 = torch.maximum(
        predicted_box[:, 0],
        predicted_box[:, 2],
    )

    predicted_box_y2 = torch.maximum(
        predicted_box[:, 1],
        predicted_box[:, 3],
    )

    predicted_box = torch.stack(
        [
            predicted_box_x1,
            predicted_box_y1,
            predicted_box_x2,
            predicted_box_y2,
        ],
        dim=-1,
    )

    ious = box_iou(
        predicted_box,
        targets,
    )

    mean_iou = ious.mean().item()

    print(
        f"{lower_q:.2f} - {upper_q:.2f}"
        f" | Mean IoU: {mean_iou:.6f}"
        f" | >=0.50: "
        f"{(ious >= 0.50).sum().item():3d}/256"
        f" | >=0.25: "
        f"{(ious >= 0.25).sum().item():3d}/256"
    )

    if mean_iou > best_iou:

        best_iou = mean_iou

        best_quantile = (
            lower_q,
            upper_q,
        )


# ============================================================
# Recompute best prediction
# ============================================================

lower_q, upper_q = best_quantile

best_x1 = discrete_quantile(
    x_distribution,
    lower_q,
)

best_x2 = discrete_quantile(
    x_distribution,
    upper_q,
)

best_y1 = discrete_quantile(
    y_distribution,
    lower_q,
)

best_y2 = discrete_quantile(
    y_distribution,
    upper_q,
)

best_boxes = torch.stack(
    [
        best_x1,
        best_y1,
        best_x2,
        best_y2,
    ],
    dim=-1,
)

# Ensure valid ordering.

best_boxes = torch.stack(
    [
        torch.minimum(
            best_boxes[:, 0],
            best_boxes[:, 2],
        ),
        torch.minimum(
            best_boxes[:, 1],
            best_boxes[:, 3],
        ),
        torch.maximum(
            best_boxes[:, 0],
            best_boxes[:, 2],
        ),
        torch.maximum(
            best_boxes[:, 1],
            best_boxes[:, 3],
        ),
    ],
    dim=-1,
)

best_ious = box_iou(
    best_boxes,
    targets,
)


# ============================================================
# Final results
# ============================================================

print()
print("=" * 70)
print("BEST BOUNDARY EXTRACTION")
print("=" * 70)

print(
    f"Best quantiles: "
    f"{lower_q:.2f} - {upper_q:.2f}"
)

print(
    f"Mean IoU: "
    f"{best_ious.mean().item():.6f}"
)

print(
    f"IoU >= 0.50: "
    f"{(best_ious >= 0.50).sum().item():,} / "
    f"{len(best_ious):,}"
)

print(
    f"IoU >= 0.25: "
    f"{(best_ious >= 0.25).sum().item():,} / "
    f"{len(best_ious):,}"
)


# ============================================================
# Examples
# ============================================================

print()
print("=" * 70)
print("EXAMPLE BOXES")
print("=" * 70)

for i in range(
    min(10, len(best_boxes))
):

    print()

    print(
        f"Example {i:02d}"
    )

    print(
        "Target: ["
        + ", ".join(
            f"{v:.4f}"
            for v in targets[i].tolist()
        )
        + "]"
    )

    print(
        "Boundary box: ["
        + ", ".join(
            f"{v:.4f}"
            for v in best_boxes[i].tolist()
        )
        + "]"
    )

    print(
        f"IoU: "
        f"{best_ious[i].item():.4f}"
    )


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if best_iou >= 0.45:

    print(
        "STRONG RESULT."
    )

    print()
    print(
        "The heatmap contains enough boundary information"
    )

    print(
        "to reconstruct the target box quite accurately."
    )

elif best_iou >= 0.40:

    print(
        "USEFUL RESULT."
    )

    print()
    print(
        "Direct boundary extraction is competitive with"
    )

    print(
        "the previous geometry baseline."
    )

else:

    print(
        "LIMITED RESULT."
    )

    print()
    print(
        "The heatmap identifies the target region well,"
    )

    print(
        "but its probability extent is not precise enough"
    )

    print(
        "for direct boundary reconstruction."
    )

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)