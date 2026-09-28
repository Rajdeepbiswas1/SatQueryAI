from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, r"D:\SatQueryAI\src")

from data.bigearthnet_dataset import BigEarthNetDOFADataset
from data.dataloaders import make_bigearthnet_loader
from models.feature_fusion import CrossModalFusion


def main() -> None:
    device = torch.device("cpu")

    # --------------------------------------------------------
    # Load a small BigEarthNet batch
    # --------------------------------------------------------
    dataset = BigEarthNetDOFADataset(
        annotations_path=(
            r"D:\SatQueryAI\data\processed"
            r"\training_annotations.parquet"
        ),
        feature_dir=(
            r"D:\SatQueryAI\data\processed"
            r"\dofa_features_download\dofa_features"
        ),
        split="train",
    )

    loader = make_bigearthnet_loader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    s1 = batch["s1"].to(device)
    s2 = batch["s2"].to(device)

    print("Input S1:", tuple(s1.shape), s1.dtype)
    print("Input S2:", tuple(s2.shape), s2.dtype)

    # --------------------------------------------------------
    # Trainable model
    # --------------------------------------------------------
    fusion = CrossModalFusion(
        input_dim=768,
        hidden_dim=256,
    ).to(device)

    prediction_head = nn.Sequential(
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    ).to(device)

    optimizer = AdamW(
        list(fusion.parameters())
        + list(prediction_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    loss_fn = nn.MSELoss()

    # --------------------------------------------------------
    # Dummy target
    # --------------------------------------------------------
    target = torch.randn(
        s1.shape[0],
        1,
        device=device,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    fused = fusion(s1, s2)

    print("Fused:", tuple(fused.shape), fused.dtype)

    # Average spatial tokens only for this mechanical test.
    pooled = fused.mean(dim=1)

    print("Pooled:", tuple(pooled.shape), pooled.dtype)

    prediction_before = prediction_head(pooled)

    print(
        "Prediction:",
        tuple(prediction_before.shape),
        prediction_before.dtype,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    loss_before = loss_fn(
        prediction_before,
        target,
    )

    print("Loss before backward:", loss_before.item())

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------
    optimizer.zero_grad()

    loss_before.backward()

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------
    gradient_count = 0
    max_gradient = 0.0

    for name, parameter in list(fusion.named_parameters()) + list(
        prediction_head.named_parameters()
    ):
        if parameter.requires_grad and parameter.grad is not None:
            gradient_count += 1

            grad_max = parameter.grad.abs().max().item()
            max_gradient = max(max_gradient, grad_max)

            print(
                f"{name}: gradient max = {grad_max:.6f}"
            )

    print("Parameters with gradients:", gradient_count)
    print("Maximum gradient:", max_gradient)

    assert gradient_count > 0
    assert max_gradient > 0.0

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------
    optimizer.step()

    # --------------------------------------------------------
    # Forward again after update
    # --------------------------------------------------------
    with torch.no_grad():
        fused_after = fusion(s1, s2)
        pooled_after = fused_after.mean(dim=1)
        prediction_after = prediction_head(pooled_after)

        loss_after = loss_fn(
            prediction_after,
            target,
        )

    print("Loss after update:", loss_after.item())

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------
    assert torch.isfinite(fused_after).all()
    assert torch.isfinite(prediction_after).all()
    assert torch.isfinite(loss_before)
    assert torch.isfinite(loss_after)

    print()
    print("FORWARD + BACKWARD TEST PASSED")


if __name__ == "__main__":
    main()