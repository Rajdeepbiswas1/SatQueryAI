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
    grounding_heatmap_loss,
    box_iou,
    boxes_to_heatmaps,
)


# ============================================================
# Configuration
# ============================================================

NUM_EXAMPLES = 256
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3


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
# Heatmap metrics
# ============================================================

def heatmap_metrics(
    predicted_heatmap,
    target_heatmap,
):
    batch_size = predicted_heatmap.size(0)

    predicted_flat = predicted_heatmap.reshape(
        batch_size,
        -1,
    )

    target_flat = target_heatmap.reshape(
        batch_size,
        -1,
    )

    target_binary = (
        target_flat > 0
    ).float()

    mass_inside = (
        predicted_flat
        * target_binary
    ).sum(dim=1)

    weighted_mass = (
        predicted_flat
        * target_flat
    ).sum(dim=1)

    peak_indices = predicted_flat.argmax(
        dim=1
    )

    peak_inside = target_binary[
        torch.arange(
            batch_size,
            device=predicted_heatmap.device,
        ),
        peak_indices,
    ]

    return {
        "mass_inside": mass_inside,
        "weighted_mass": weighted_mass,
        "peak_inside": peak_inside,
    }


# ============================================================
# Geometry metrics
# ============================================================

def geometry_metrics(
    geometry,
    target_boxes,
):
    """
    geometry:
        [center_x, center_y,
         std_x, std_y,
         x1, y1, x2, y2]
    """

    geometry_box = geometry[:, 4:8]

    geometry_iou = box_iou(
        geometry_box,
        target_boxes,
    )

    target_center_x = (
        target_boxes[:, 0]
        + target_boxes[:, 2]
    ) / 2.0

    target_center_y = (
        target_boxes[:, 1]
        + target_boxes[:, 3]
    ) / 2.0

    center_error = (
        (
            geometry[:, 0]
            - target_center_x
        ).abs()
        +
        (
            geometry[:, 1]
            - target_center_y
        ).abs()
    )

    return {
        "iou": geometry_iou,
        "center_error": center_error,
    }


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("REFERENCE HEATMAP + GEOMETRY + CORRECTION OVERFIT")
print("=" * 70)

dataset = GroundingDataset()

reference_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "reference"
]

print(
    f"Available reference examples: "
    f"{len(reference_indices):,}"
)

assert len(reference_indices) >= NUM_EXAMPLES

reference_subset = reference_indices[:NUM_EXAMPLES]

loader = DataLoader(
    Subset(
        dataset,
        reference_subset,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

print(
    f"Training examples: {NUM_EXAMPLES}"
)

print(
    f"Batch size:        {BATCH_SIZE}"
)

print(
    f"Epochs:            {EPOCHS}"
)


# ============================================================
# Model
# ============================================================

model = GroundingHeatmapModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_box_loss = 0.0

    total_mass_inside = 0.0
    total_weighted_mass = 0.0
    total_peak_inside = 0.0

    total_box_iou = 0.0
    total_base_box_iou = 0.0

    total_geometry_center_error = 0.0

    total_correction_magnitude = 0.0

    total = 0

    for batch in loader:

        outputs = model(
            s1=batch["s1"],
            s2=batch["s2"],
            text_embedding=batch[
                "text_embedding"
            ],
        )

        losses = grounding_heatmap_loss(
            outputs,
            batch["target_box"],
            heatmap_weight=1.0,
            box_weight=0.25,
        )

        loss = losses["total"]

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            hm = heatmap_metrics(
                outputs["heatmap"],
                losses["target_heatmap"],
            )

            final_iou = box_iou(
                outputs["box"],
                batch["target_box"],
            )

            base_iou = box_iou(
                outputs["base_box"],
                batch["target_box"],
            )

            geom = geometry_metrics(
                outputs["geometry"],
                batch["target_box"],
            )

            correction_magnitude = (
                outputs["correction"]
                .abs()
                .mean(dim=1)
            )

        n = batch["target_box"].size(0)

        total_loss += (
            loss.item() * n
        )

        total_heatmap_loss += (
            losses["heatmap"].item()
            * n
        )

        total_box_loss += (
            losses["box"].item()
            * n
        )

        total_mass_inside += (
            hm["mass_inside"]
            .sum()
            .item()
        )

        total_weighted_mass += (
            hm["weighted_mass"]
            .sum()
            .item()
        )

        total_peak_inside += (
            hm["peak_inside"]
            .sum()
            .item()
        )

        total_box_iou += (
            final_iou.sum()
            .item()
        )

        total_base_box_iou += (
            base_iou.sum()
            .item()
        )

        total_geometry_center_error += (
            geom["center_error"]
            .sum()
            .item()
        )

        total_correction_magnitude += (
            correction_magnitude
            .sum()
            .item()
        )

        total += n

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Total: {total_loss / total:.4f} | "
        f"HM: {total_heatmap_loss / total:.4f} | "
        f"BoxLoss: {total_box_loss / total:.4f} | "
        f"MassInTarget: {total_mass_inside / total:.4f} | "
        f"Peak@Target: {total_peak_inside / total:.4f} | "
        f"BaseIoU: {total_base_box_iou / total:.4f} | "
        f"FinalIoU: {total_box_iou / total:.4f} | "
        f"CenterErr: {total_geometry_center_error / total:.4f} | "
        f"Correction: {total_correction_magnitude / total:.4f}"
    )


# ============================================================
# Final evaluation
# ============================================================

print()
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

model.eval()

eval_loader = DataLoader(
    Subset(
        dataset,
        reference_subset,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

all_final_boxes = []
all_base_boxes = []
all_geometry = []
all_targets = []

all_heatmaps = []
all_target_heatmaps = []
all_corrections = []

with torch.no_grad():

    for batch in eval_loader:

        outputs = model(
            s1=batch["s1"],
            s2=batch["s2"],
            text_embedding=batch[
                "text_embedding"
            ],
        )

        target_heatmap = boxes_to_heatmaps(
            batch["target_box"]
        )

        all_final_boxes.append(
            outputs["box"]
        )

        all_base_boxes.append(
            outputs["base_box"]
        )

        all_geometry.append(
            outputs["geometry"]
        )

        all_targets.append(
            batch["target_box"]
        )

        all_heatmaps.append(
            outputs["heatmap"]
        )

        all_target_heatmaps.append(
            target_heatmap
        )

        all_corrections.append(
            outputs["correction"]
        )


final_boxes = torch.cat(
    all_final_boxes,
    dim=0,
)

base_boxes = torch.cat(
    all_base_boxes,
    dim=0,
)

geometry = torch.cat(
    all_geometry,
    dim=0,
)

targets = torch.cat(
    all_targets,
    dim=0,
)

pred_heatmaps = torch.cat(
    all_heatmaps,
    dim=0,
)

target_heatmaps = torch.cat(
    all_target_heatmaps,
    dim=0,
)

corrections = torch.cat(
    all_corrections,
    dim=0,
)


# ============================================================
# Final metrics
# ============================================================

final_iou = box_iou(
    final_boxes,
    targets,
)

base_iou = box_iou(
    base_boxes,
    targets,
)

geometry_iou = box_iou(
    geometry[:, 4:8],
    targets,
)

hm_metrics = heatmap_metrics(
    pred_heatmaps,
    target_heatmaps,
)

geometry_metric_values = geometry_metrics(
    geometry,
    targets,
)


# ============================================================
# Results
# ============================================================

print(
    f"Final mean box IoU: "
    f"{final_iou.mean().item():.6f}"
)

print(
    f"Base geometry-box IoU: "
    f"{base_iou.mean().item():.6f}"
)

print(
    f"Geometry-box IoU: "
    f"{geometry_iou.mean().item():.6f}"
)

print(
    f"Mean probability mass inside target: "
    f"{hm_metrics['mass_inside'].mean().item():.6f}"
)

print(
    f"Peak inside target: "
    f"{hm_metrics['peak_inside'].mean().item():.6f}"
)

print(
    f"Mean correction magnitude: "
    f"{corrections.abs().mean().item():.6f}"
)

print()
print(
    f"Final Box IoU >= 0.50: "
    f"{(final_iou >= 0.50).sum().item():,} / "
    f"{len(final_iou):,}"
)

print(
    f"Base Box IoU >= 0.50: "
    f"{(base_iou >= 0.50).sum().item():,} / "
    f"{len(base_iou):,}"
)

print(
    f"Geometry IoU >= 0.50: "
    f"{(geometry_iou >= 0.50).sum().item():,} / "
    f"{len(geometry_iou):,}"
)


# ============================================================
# Does correction improve the base box?
# ============================================================

improvement = (
    final_iou
    -
    base_iou
)

print()
print("=" * 70)
print("CORRECTION EFFECT")
print("=" * 70)

print(
    f"Mean IoU improvement over base: "
    f"{improvement.mean().item():.6f}"
)

print(
    f"Examples improved: "
    f"{(improvement > 0).sum().item():,} / "
    f"{len(improvement):,}"
)

print(
    f"Examples unchanged: "
    f"{(improvement.abs() < 1e-4).sum().item():,} / "
    f"{len(improvement):,}"
)

print(
    f"Examples worsened: "
    f"{(improvement < 0).sum().item():,} / "
    f"{len(improvement):,}"
)


# ============================================================
# Example results
# ============================================================

print()
print("=" * 70)
print("EXAMPLE LOCALIZATION RESULTS")
print("=" * 70)

for i in range(
    min(10, len(final_boxes))
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
        "Base:   ["
        + ", ".join(
            f"{v:.4f}"
            for v in base_boxes[i].tolist()
        )
        + "]"
    )

    print(
        "Final:  ["
        + ", ".join(
            f"{v:.4f}"
            for v in final_boxes[i].tolist()
        )
        + "]"
    )

    print(
        "Geometry: ["
        + ", ".join(
            f"{v:.4f}"
            for v in geometry[i, 4:8].tolist()
        )
        + "]"
    )

    print(
        f"Base IoU:     {base_iou[i].item():.4f}"
    )

    print(
        f"Final IoU:    {final_iou[i].item():.4f}"
    )

    print(
        f"Geometry IoU: {geometry_iou[i].item():.4f}"
    )

    print(
        "Correction: ["
        + ", ".join(
            f"{v:.4f}"
            for v in corrections[i].tolist()
        )
        + "]"
    )


# ============================================================
# Save checkpoint
# ============================================================

checkpoint_dir = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "grounding_heatmap_correction"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint_path = (
    checkpoint_dir
    / "model_256_reference.pt"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_examples": NUM_EXAMPLES,
        "epochs": EPOCHS,
    },
    checkpoint_path,
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Checkpoint: "
    f"{checkpoint_path}"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

mean_base = base_iou.mean().item()
mean_final = final_iou.mean().item()
mean_geometry = geometry_iou.mean().item()
mean_improvement = improvement.mean().item()

if mean_final > mean_base + 0.03:

    print(
        "SUCCESS: GEOMETRY-PRESERVING CORRECTION HELPS."
    )

    print()
    print(
        "The learned correction improves the heatmap-derived"
    )

    print(
        "base box instead of replacing its spatial information."
    )

elif mean_final >= mean_base - 0.01:

    print(
        "NEUTRAL: CORRECTION PRESERVES THE BASE BOX."
    )

    print()
    print(
        "The correction head is not substantially damaging"
    )

    print(
        "the geometry-derived localization."
    )

else:

    print(
        "NEGATIVE: CORRECTION DEGRADES THE BASE BOX."
    )

    print()
    print(
        "The correction head is learning an unhelpful"
    )

    print(
        "adjustment and needs to be reconsidered."
    )

print()
print(
    f"Base IoU:       {mean_base:.4f}"
)

print(
    f"Final IoU:      {mean_final:.4f}"
)

print(
    f"Geometry IoU:   {mean_geometry:.4f}"
)

print(
    f"Improvement:    {mean_improvement:+.4f}"
)

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)