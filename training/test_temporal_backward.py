from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(r"D:\SatQueryAI")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.train_temporal_kaggle as t


def main():
    print("=" * 70)
    print("SatQueryAI TEMPORAL LOSS + BACKWARD PREFLIGHT")
    print("=" * 70)
    print()

    print("[1/5] Loading temporal feature store...")
    feature_store = t.TemporalFeatureStore(t.FEATURE_DIR)
    print("    PASS")
    print()

    print("[2/5] Loading training question embeddings...")
    question_embeddings = t.load_question_embeddings(
        ROOT / "data" / "processed" / "temporal_question_embeddings",
        65967,
        "train",
    )
    print("    PASS")
    print()

    print("[3/5] Building one real training batch...")
    dataset = t.TemporalRAMDataset(
        t.TRAIN_MANIFEST,
        feature_store,
        question_embeddings,
        "train",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print(f"    batch size: {batch['t1'].shape[0]}")
    print(f"    t1: {tuple(batch['t1'].shape)}")
    print(f"    t2: {tuple(batch['t2'].shape)}")
    print(f"    diff: {tuple(batch['diff'].shape)}")
    print(f"    question: {tuple(batch['question'].shape)}")
    print(f"    question types: {batch['question_type']}")
    print(f"    targets: {batch['target'].tolist()}")
    print("    PASS")
    print()

    print("[4/5] Running forward pass and routed loss...")

    model = t.TemporalChangeVQA()

    expected_params = 1_744_925
    actual_params = sum(
        p.numel() for p in model.parameters()
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"Expected {expected_params:,} parameters, "
            f"found {actual_params:,}"
        )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    outputs = model(
        batch["t1"].float(),
        batch["t2"].float(),
        batch["diff"].float(),
        batch["question"].float(),
    )

    print(
        "    output answer heads:",
        list(outputs["answer_logits"].keys()),
    )

    loss, per_head_loss = t.routed_loss(
        outputs,
        batch["question_type"],
        batch["target"],
    )

    if not torch.is_tensor(loss):
        raise RuntimeError(
            f"routed_loss returned {type(loss)} instead of a tensor"
        )

    print(f"    loss: {loss.item():.6f}")
    print(f"    per-head losses: {per_head_loss}")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    print("    finite loss: PASS")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    print("    backward pass: PASS")

    gradient_count = 0
    finite_gradient_count = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_count += 1

            if torch.isfinite(parameter.grad).all():
                finite_gradient_count += 1

    print(f"    parameters with gradients: {gradient_count}")
    print(f"    finite gradients:          {finite_gradient_count}")

    if gradient_count == 0:
        raise RuntimeError(
            "No model parameters received gradients."
        )

    if gradient_count != finite_gradient_count:
        raise RuntimeError(
            "At least one parameter has a non-finite gradient."
        )

    print("    gradient validation: PASS")
    print()

    print("[5/5] Testing one optimizer update...")

    before = {}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            before[name] = parameter.detach().clone()
            break

    optimizer.step()

    changed = False

    for name, parameter in model.named_parameters():
        if name in before:
            if not torch.equal(before[name], parameter.detach()):
                changed = True
            break

    if not changed:
        raise RuntimeError(
            "Optimizer step completed but no checked parameter changed."
        )

    print("    optimizer step: PASS")
    print()

    print("=" * 70)
    print("RESULT: TEMPORAL LOSS + BACKWARD PREFLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()