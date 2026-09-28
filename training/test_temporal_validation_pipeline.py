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
    print("SatQueryAI TEMPORAL VALIDATION PREFLIGHT")
    print("=" * 70)
    print()

    print("[1/5] Loading temporal feature store...")
    feature_store = t.TemporalFeatureStore(t.FEATURE_DIR)
    print("    PASS")
    print()

    print("[2/5] Loading validation question embeddings...")

    val_question_embeddings = t.load_question_embeddings(
        ROOT / "data" / "processed" / "temporal_val_question_embeddings",
        16441,
        "validation",
    )

    print(
        f"    embeddings shape: "
        f"{tuple(val_question_embeddings.shape)}"
    )

    if tuple(val_question_embeddings.shape) != (16441, 768):
        raise RuntimeError(
            f"Unexpected validation embedding shape: "
            f"{tuple(val_question_embeddings.shape)}"
        )

    print("    PASS")
    print()

    print("[3/5] Building validation dataset and DataLoader...")

    dataset = t.TemporalRAMDataset(
        t.VAL_MANIFEST,
        feature_store,
        val_question_embeddings,
        "val",
    )

    print(f"    dataset size: {len(dataset)}")

    if len(dataset) != 16441:
        raise RuntimeError(
            f"Expected 16441 validation rows, found {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print(f"    t1:       {tuple(batch['t1'].shape)}")
    print(f"    t2:       {tuple(batch['t2'].shape)}")
    print(f"    diff:     {tuple(batch['diff'].shape)}")
    print(f"    question: {tuple(batch['question'].shape)}")
    print(f"    targets:  {batch['target'].tolist()}")
    print(f"    types:    {batch['question_type']}")

    print("    PASS")
    print()

    print("[4/5] Running validation forward + loss...")

    model = t.TemporalChangeVQA()
    model.eval()

    expected_params = 1_744_925
    actual_params = sum(
        p.numel() for p in model.parameters()
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"Expected {expected_params:,} parameters, "
            f"found {actual_params:,}"
        )

    with torch.no_grad():
        outputs = model(
            batch["t1"].float(),
            batch["t2"].float(),
            batch["diff"].float(),
            batch["question"].float(),
        )

        loss, per_head_loss = t.routed_loss(
            outputs,
            batch["question_type"],
            batch["target"],
        )

    print(f"    loss: {loss.item():.6f}")
    print(f"    per-head losses: {per_head_loss}")

    if not torch.isfinite(loss):
        raise RuntimeError("Validation loss is not finite.")

    print("    finite loss: PASS")

    for name, logits in outputs["answer_logits"].items():
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite values in answer head: {name}"
            )

    print("    output finiteness: PASS")
    print()

    print("[5/5] Checking validation accuracy routing...")

    head_map = {
        "change_or_not": "binary",
        "increase_or_not": "binary",
        "decrease_or_not": "binary",
        "smallest_change": "land_cover",
        "largest_change": "land_cover",
        "change_to_what": "land_cover",
        "change_ratio": "change_ratio",
        "change_ratio_types": "change_ratio_types",
    }

    correct = 0
    total = len(batch["question_type"])

    for i, qtype in enumerate(batch["question_type"]):
        head = head_map[qtype]
        prediction = outputs["answer_logits"][head][i].argmax().item()
        target = batch["target"][i].item()

        if prediction == target:
            correct += 1

    accuracy = correct / total

    print(f"    batch accuracy: {accuracy:.4f}")
    print(f"    correct: {correct}/{total}")
    print("    routing: PASS")
    print()

    print("=" * 70)
    print("RESULT: TEMPORAL VALIDATION PREFLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()