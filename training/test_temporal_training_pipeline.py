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
    print("SatQueryAI ACTUAL TRAINING PIPELINE PREFLIGHT")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. Load the exact feature store used by training
    # --------------------------------------------------------

    print("[1/4] Building TemporalFeatureStore...")
    feature_store = t.TemporalFeatureStore(t.FEATURE_DIR)

    print("    PASS: feature store loaded.")
    print()

    # --------------------------------------------------------
    # 2. Load the exact training question cache
    # --------------------------------------------------------

    print("[2/4] Loading training question embeddings...")

    question_embeddings = t.load_question_embeddings(
        ROOT / "data" / "processed" / "temporal_question_embeddings",
        65967,
        "train",
    )

    print(
        f"    embeddings shape: {tuple(question_embeddings.shape)}"
    )
    print("    PASS: training question cache loaded.")
    print()

    # --------------------------------------------------------
    # 3. Build the exact training dataset + DataLoader
    # --------------------------------------------------------

    print("[3/4] Building real training dataset and DataLoader...")

    dataset = t.TemporalRAMDataset(
        t.TRAIN_MANIFEST,
        feature_store,
        question_embeddings,
        "train",
    )

    print(f"    dataset size: {len(dataset)}")

    if len(dataset) != 65967:
        raise RuntimeError(
            f"Expected 65967 training examples, found {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print("    batch keys:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(
                f"        {key}: Tensor{tuple(value.shape)} "
                f"{value.dtype}"
            )
        else:
            print(
                f"        {key}: {type(value).__name__}"
            )

    print()
    print("    PASS: first real batch loaded.")
    print()

    # --------------------------------------------------------
    # 4. Run real batch through the frozen model
    # --------------------------------------------------------

    print("[4/4] Running real batch through TemporalChangeVQA...")

    model = t.TemporalChangeVQA()
    model.eval()

    expected_params = 1_744_925
    actual_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"    model parameters: {actual_params:,}")

    if actual_params != expected_params:
        raise RuntimeError(
            f"Expected {expected_params:,} parameters, "
            f"found {actual_params:,}"
        )

    # Preferred exact dataset field names.
    if all(name in batch for name in ("t1", "t2", "diff")):
        t1 = batch["t1"]
        t2 = batch["t2"]
        diff = batch["diff"]

    else:
        # Fallback: identify the three temporal tensors
        feature_candidates = []

        for key, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.ndim == 3
                and tuple(value.shape[1:]) == (196, 768)
            ):
                feature_candidates.append((key, value))

        if len(feature_candidates) != 3:
            raise RuntimeError(
                "Could not identify exactly three temporal "
                "[B,196,768] tensors."
            )

        print(
            "    detected temporal tensors:",
            [key for key, _ in feature_candidates],
        )

        # Try to identify by field name first.
        by_name = {
            key.lower(): value
            for key, value in feature_candidates
        }

        t1 = next(
            (
                value
                for key, value in feature_candidates
                if key.lower() == "t1"
            ),
            None,
        )
        t2 = next(
            (
                value
                for key, value in feature_candidates
                if key.lower() == "t2"
            ),
            None,
        )
        diff = next(
            (
                value
                for key, value in feature_candidates
                if key.lower() == "diff"
            ),
            None,
        )

        if t1 is None or t2 is None or diff is None:
            raise RuntimeError(
                "Temporal tensors are present, but their expected "
                "names t1/t2/diff could not be resolved."
            )

    # Identify the question embedding.
    question = None

    preferred_names = (
        "question_embedding",
        "question_embeddings",
        "question",
    )

    for name in preferred_names:
        if name in batch:
            value = batch[name]

            if (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == 768
            ):
                question = value
                break

    if question is None:
        candidates = []

        for key, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == 768
            ):
                candidates.append((key, value))

        if len(candidates) == 1:
            question = candidates[0][1]
            print(
                f"    detected question embedding field: "
                f"{candidates[0][0]}"
            )

    if question is None:
        raise RuntimeError(
            "Could not identify a [B,768] question embedding."
        )

    t1 = t1.float()
    t2 = t2.float()
    diff = diff.float()
    question = question.float()

    print(
        f"    t1:       {tuple(t1.shape)}"
    )
    print(
        f"    t2:       {tuple(t2.shape)}"
    )
    print(
        f"    diff:     {tuple(diff.shape)}"
    )
    print(
        f"    question: {tuple(question.shape)}"
    )

    with torch.no_grad():
        outputs = model(
            t1,
            t2,
            diff,
            question,
        )

    expected_heads = {
        "binary": (4, 2),
        "change_ratio": (4, 11),
        "change_ratio_types": (4, 9),
        "land_cover": (4, 6),
    }

    answer_logits = outputs["answer_logits"]

    for head_name, expected_shape in expected_heads.items():
        actual_shape = tuple(answer_logits[head_name].shape)

        print(
            f"    {head_name:20s}: {actual_shape}"
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{head_name}: expected {expected_shape}, "
                f"found {actual_shape}"
            )

    core_outputs = {
        "query_attention": (4, 14, 14),
        "change_logits": (4, 14, 14),
        "fused_tokens": (4, 196, 256),
        "joint": (4, 256),
    }

    for name, expected_shape in core_outputs.items():
        actual_shape = tuple(outputs[name].shape)

        print(
            f"    {name:20s}: {actual_shape}"
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name}: expected {expected_shape}, "
                f"found {actual_shape}"
            )

    all_finite = all(
        torch.isfinite(value).all().item()
        for value in answer_logits.values()
    )

    if not all_finite:
        raise RuntimeError(
            "Non-finite values detected in model outputs."
        )

    print()
    print("    PASS: real batch forward completed successfully.")
    print()

    print("=" * 70)
    print("RESULT: ACTUAL TRAINING PIPELINE PREFLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()