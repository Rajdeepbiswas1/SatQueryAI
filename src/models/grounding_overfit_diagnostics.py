import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.grounding_dataset import GroundingDataset


def collate_fn(batch):
    return {
        "target_box": torch.stack(
            [x["target_box"] for x in batch]
        ),
        "mode_id": torch.stack(
            [x["mode_id"] for x in batch]
        ),
        "mode": [
            x["mode"]
            for x in batch
        ],
    }


print("=" * 70)
print("GROUNDING 256-EXAMPLE DIAGNOSTICS")
print("=" * 70)

dataset = GroundingDataset()

subset = Subset(
    dataset,
    range(256),
)

loader = DataLoader(
    subset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

all_boxes = []
all_modes = []

for batch in loader:
    all_boxes.append(batch["target_box"])
    all_modes.append(batch["mode_id"])

boxes = torch.cat(
    all_boxes,
    dim=0,
)

modes = torch.cat(
    all_modes,
    dim=0,
)

widths = boxes[:, 2] - boxes[:, 0]
heights = boxes[:, 3] - boxes[:, 1]
areas = widths * heights

print("\nTOTAL")
print(f"Examples: {len(boxes)}")

print("\nMODE SPLIT")

point_mask = modes == 1
reference_mask = modes == 0

print(
    f"Point:     {point_mask.sum().item()}"
)

print(
    f"Reference: {reference_mask.sum().item()}"
)

print("\nBOX AREA")

print(
    f"Min:    {areas.min().item():.4f}"
)

print(
    f"Median: {areas.median().item():.4f}"
)

print(
    f"Mean:   {areas.mean().item():.4f}"
)

print(
    f"Max:    {areas.max().item():.4f}"
)

print("\nBOX WIDTH")

print(
    f"Min:    {widths.min().item():.4f}"
)

print(
    f"Median: {widths.median().item():.4f}"
)

print(
    f"Mean:   {widths.mean().item():.4f}"
)

print(
    f"Max:    {widths.max().item():.4f}"
)

print("\nBOX HEIGHT")

print(
    f"Min:    {heights.min().item():.4f}"
)

print(
    f"Median: {heights.median().item():.4f}"
)

print(
    f"Mean:   {heights.mean().item():.4f}"
)

print(
    f"Max:    {heights.max().item():.4f}"
)

# ------------------------------------------------------------
# Mode-specific statistics
# ------------------------------------------------------------

for name, mask in [
    ("POINT", point_mask),
    ("REFERENCE", reference_mask),
]:

    if mask.sum() == 0:
        continue

    mode_areas = areas[mask]
    mode_widths = widths[mask]
    mode_heights = heights[mask]

    print("\n" + "=" * 70)
    print(f"{name} BOX STATISTICS")
    print("=" * 70)

    print(
        f"Examples: {mask.sum().item()}"
    )

    print(
        f"Area mean:   {mode_areas.mean().item():.4f}"
    )

    print(
        f"Area median: {mode_areas.median().item():.4f}"
    )

    print(
        f"Width mean:  {mode_widths.mean().item():.4f}"
    )

    print(
        f"Height mean: {mode_heights.mean().item():.4f}"
    )

    print(
        f"Small boxes (area < 0.05): "
        f"{(mode_areas < 0.05).sum().item()}"
    )

    print(
        f"Medium boxes (0.05-0.20): "
        f"{((mode_areas >= 0.05) & (mode_areas < 0.20)).sum().item()}"
    )

    print(
        f"Large boxes (>= 0.20): "
        f"{(mode_areas >= 0.20).sum().item()}"
    )

# ------------------------------------------------------------
# Overall size groups
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL BOX SIZE GROUPS")
print("=" * 70)

small = areas < 0.05
medium = (areas >= 0.05) & (areas < 0.20)
large = areas >= 0.20

print(
    f"Small  (<0.05):   {small.sum().item()}"
)

print(
    f"Medium (0.05-0.20): {medium.sum().item()}"
)

print(
    f"Large  (>=0.20):  {large.sum().item()}"
)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)