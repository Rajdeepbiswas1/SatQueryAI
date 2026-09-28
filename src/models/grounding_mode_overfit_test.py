import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_model import GroundingModel, box_iou


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
        "point": torch.stack(
            [x["point"] for x in batch]
        ),
        "target_box": torch.stack(
            [x["target_box"] for x in batch]
        ),
        "mode_id": torch.stack(
            [x["mode_id"] for x in batch]
        ),
    }


def run_mode_test(
    name,
    indices,
    dataset,
    epochs=30,
    batch_size=32,
):
    print("\n" + "=" * 70)
    print(f"{name} OVERFIT TEST")
    print("=" * 70)

    subset = Subset(
        dataset,
        indices,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Examples: {len(subset)}")
    print(f"Batch size: {batch_size}")

    model = GroundingModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_iou = 0.0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):

        model.train()

        epoch_loss = 0.0
        epoch_iou = 0.0
        total = 0

        for batch in loader:

            predicted = model(
                s1=batch["s1"],
                s2=batch["s2"],
                text_embedding=batch["text_embedding"],
                point=batch["point"],
                mode_id=batch["mode_id"],
            )

            target = batch["target_box"]

            loss = F.smooth_l1_loss(
                predicted,
                target,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_iou = box_iou(
                predicted.detach(),
                target,
            )

            n = target.size(0)

            epoch_loss += (
                loss.item() * n
            )

            epoch_iou += (
                batch_iou.sum().item()
            )

            total += n

        epoch_loss /= total
        epoch_iou /= total

        best_iou = max(
            best_iou,
            epoch_iou,
        )

        best_loss = min(
            best_loss,
            epoch_loss,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Mean IoU: {epoch_iou:.4f}"
        )

    # --------------------------------------------------------
    # Final evaluation on the same examples
    # --------------------------------------------------------

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            pred = model(
                s1=batch["s1"],
                s2=batch["s2"],
                text_embedding=batch["text_embedding"],
                point=batch["point"],
                mode_id=batch["mode_id"],
            )

            predictions.append(pred)
            targets.append(batch["target_box"])

    predictions = torch.cat(
        predictions,
        dim=0,
    )

    targets = torch.cat(
        targets,
        dim=0,
    )

    final_loss = F.smooth_l1_loss(
        predictions,
        targets,
    )

    final_ious = box_iou(
        predictions,
        targets,
    )

    final_mean_iou = final_ious.mean().item()

    print("\n" + "-" * 70)
    print(f"{name} FINAL RESULTS")
    print("-" * 70)

    print(
        f"Final SmoothL1 loss: {final_loss.item():.6f}"
    )

    print(
        f"Final mean IoU:      {final_mean_iou:.6f}"
    )

    print(
        f"Best training IoU:   {best_iou:.6f}"
    )

    # --------------------------------------------------------
    # IoU distribution
    # --------------------------------------------------------

    print("\nIoU distribution:")

    print(
        f"IoU >= 0.50: "
        f"{(final_ious >= 0.50).sum().item():,} / "
        f"{len(final_ious):,}"
    )

    print(
        f"IoU >= 0.25: "
        f"{(final_ious >= 0.25).sum().item():,} / "
        f"{len(final_ious):,}"
    )

    print(
        f"IoU < 0.10:  "
        f"{(final_ious < 0.10).sum().item():,} / "
        f"{len(final_ious):,}"
    )

    # --------------------------------------------------------
    # Some example predictions
    # --------------------------------------------------------

    print("\nExample predictions:")

    for i in range(
        min(10, len(predictions))
    ):

        example_iou = final_ious[
            i
        ].item()

        print(
            f"{i:02d} | "
            f"target={targets[i].tolist()} | "
            f"pred={predictions[i].tolist()} | "
            f"IoU={example_iou:.4f}"
        )

    return {
        "final_loss": final_loss.item(),
        "final_iou": final_mean_iou,
        "best_iou": best_iou,
    }


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("GROUNDING POINT vs REFERENCE OVERFIT DIAGNOSTIC")
print("=" * 70)

dataset = GroundingDataset()

# ------------------------------------------------------------
# Find indices for each mode.
# ------------------------------------------------------------

point_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "point"
]

reference_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "reference"
]

print(
    f"\nAvailable point examples: "
    f"{len(point_indices):,}"
)

print(
    f"Available reference examples: "
    f"{len(reference_indices):,}"
)

assert len(point_indices) >= 256
assert len(reference_indices) >= 256

# Use the first 256 of each mode for a controlled comparison.
point_subset = point_indices[:256]
reference_subset = reference_indices[:256]

# ------------------------------------------------------------
# Run point test
# ------------------------------------------------------------

point_results = run_mode_test(
    name="POINT",
    indices=point_subset,
    dataset=dataset,
)

# ------------------------------------------------------------
# Run reference test
# ------------------------------------------------------------

reference_results = run_mode_test(
    name="REFERENCE",
    indices=reference_subset,
    dataset=dataset,
)

# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODE COMPARISON")
print("=" * 70)

print(
    f"POINT     | "
    f"final IoU={point_results['final_iou']:.4f} | "
    f"best IoU={point_results['best_iou']:.4f}"
)

print(
    f"REFERENCE | "
    f"final IoU={reference_results['final_iou']:.4f} | "
    f"best IoU={reference_results['best_iou']:.4f}"
)

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)