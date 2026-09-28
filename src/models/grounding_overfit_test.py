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


print("=" * 70)
print("GROUNDING 256-EXAMPLE OVERFIT TEST")
print("=" * 70)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

dataset = GroundingDataset()

subset_size = 256

subset = Subset(
    dataset,
    range(subset_size),
)

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"Training examples: {len(subset)}")
print(f"Batch size: 32")

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = GroundingModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

epochs = 30

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

        batch_size = target.size(0)

        epoch_loss += (
            loss.item() * batch_size
        )

        epoch_iou += (
            batch_iou.sum().item()
        )

        total += batch_size

    epoch_loss /= total
    epoch_iou /= total

    if epoch_iou > best_iou:
        best_iou = epoch_iou

    if epoch_loss < best_loss:
        best_loss = epoch_loss

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Mean IoU: {epoch_iou:.4f}"
    )

# ------------------------------------------------------------
# Final evaluation on the same 256 examples
# ------------------------------------------------------------

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch in loader:

        predictions = model(
            s1=batch["s1"],
            s2=batch["s2"],
            text_embedding=batch["text_embedding"],
            point=batch["point"],
            mode_id=batch["mode_id"],
        )

        all_predictions.append(
            predictions
        )

        all_targets.append(
            batch["target_box"]
        )

predictions = torch.cat(
    all_predictions,
    dim=0,
)

targets = torch.cat(
    all_targets,
    dim=0,
)

final_loss = F.smooth_l1_loss(
    predictions,
    targets,
)

final_iou = box_iou(
    predictions,
    targets,
).mean()

print("\n" + "=" * 70)
print("FINAL OVERFIT RESULTS")
print("=" * 70)

print(
    f"Final SmoothL1 loss: {final_loss.item():.6f}"
)

print(
    f"Final mean IoU:       {final_iou.item():.6f}"
)

print(
    f"Best mean IoU:        {best_iou:.6f}"
)

# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("\nExample predictions:")

for i in range(10):

    example_iou = box_iou(
        predictions[i:i+1],
        targets[i:i+1],
    )[0].item()

    print(
        f"{i:02d} | "
        f"target={targets[i].tolist()} | "
        f"pred={predictions[i].tolist()} | "
        f"IoU={example_iou:.4f}"
    )

print("\n" + "=" * 70)

# ------------------------------------------------------------
# Simple success criterion
# ------------------------------------------------------------

if final_iou.item() >= 0.50:
    print(
        "OVERFIT TEST PASSED: "
        "model can learn the small grounding subset."
    )
else:
    print(
        "OVERFIT TEST DID NOT REACH IoU 0.50."
    )
    print(
        "We should inspect the architecture/training setup "
        "before full training."
    )

print("=" * 70)