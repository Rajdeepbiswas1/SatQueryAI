from __future__ import annotations

import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_change_vqa import TemporalChangeVQA


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def main() -> None:
    print("=" * 72)
    print("SATQUERYAI — TEMPORAL CHANGE VQA SMOKE TEST")
    print("=" * 72)

    torch.manual_seed(42)

    batch_size = 4

    # ------------------------------------------------------------
    # Synthetic DOFA temporal inputs.
    # ------------------------------------------------------------

    t1 = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
    )

    t2 = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
    )

    diff = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
    )

    question = torch.randn(
        batch_size,
        768,
        dtype=torch.float32,
    )

    print()
    print("Input shapes:")
    print("T1       :", tuple(t1.shape))
    print("T2       :", tuple(t2.shape))
    print("Diff     :", tuple(diff.shape))
    print("Question :", tuple(question.shape))

    # ------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------

    model = TemporalChangeVQA()
    model.eval()

    parameter_count = count_parameters(model)

    print()
    print("Model parameters:", parameter_count)

    # Expected count for the current temporal architecture.
    expected_parameters = 1_744_925

    if parameter_count != expected_parameters:
        raise RuntimeError(
            "Unexpected parameter count. "
            f"Expected {expected_parameters}, "
            f"got {parameter_count}."
        )

    print("Parameter count check: PASS")

    # ------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------

    with torch.inference_mode():
        outputs = model(
            t1=t1,
            t2=t2,
            diff=diff,
            question=question,
        )

    print()
    print("Output shapes:")

    print(
        "binary logits:",
        tuple(
            outputs["answer_logits"]["binary"].shape
        ),
    )

    print(
        "change-ratio logits:",
        tuple(
            outputs["answer_logits"]["change_ratio"].shape
        ),
    )

    print(
        "change-ratio-types logits:",
        tuple(
            outputs["answer_logits"]["change_ratio_types"].shape
        ),
    )

    print(
        "land-cover logits:",
        tuple(
            outputs["answer_logits"]["land_cover"].shape
        ),
    )

    print(
        "query attention:",
        tuple(outputs["query_attention"].shape),
    )

    print(
        "change logits:",
        tuple(outputs["change_logits"].shape),
    )

    print(
        "fused tokens:",
        tuple(outputs["fused_tokens"].shape),
    )

    print(
        "joint:",
        tuple(outputs["joint"].shape),
    )

    # ------------------------------------------------------------
    # Shape assertions.
    # ------------------------------------------------------------

    assert outputs["answer_logits"]["binary"].shape == (
        batch_size,
        2,
    )

    assert outputs["answer_logits"]["change_ratio"].shape == (
        batch_size,
        11,
    )

    # 9 verified classes:
    # 0
    # 0_to_10
    # 10_to_20
    # 20_to_30
    # 30_to_40
    # 40_to_50
    # 50_to_60
    # 60_to_70
    # 70_to_80
    assert outputs["answer_logits"]["change_ratio_types"].shape == (
        batch_size,
        9,
    )

    assert outputs["answer_logits"]["land_cover"].shape == (
        batch_size,
        6,
    )

    assert outputs["query_attention"].shape == (
        batch_size,
        14,
        14,
    )

    assert outputs["change_logits"].shape == (
        batch_size,
        14,
        14,
    )

    assert outputs["fused_tokens"].shape == (
        batch_size,
        196,
        256,
    )

    assert outputs["joint"].shape == (
        batch_size,
        256,
    )

    print()
    print("Shape checks: PASS")

    # ------------------------------------------------------------
    # Finite-value checks.
    # ------------------------------------------------------------

    tensors_to_check = [
        outputs["answer_logits"]["binary"],
        outputs["answer_logits"]["change_ratio"],
        outputs["answer_logits"]["change_ratio_types"],
        outputs["answer_logits"]["land_cover"],
        outputs["query_attention"],
        outputs["change_logits"],
        outputs["fused_tokens"],
        outputs["joint"],
    ]

    for tensor in tensors_to_check:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                "Non-finite value detected in model output."
            )

    print("Finite-value checks: PASS")

    # ------------------------------------------------------------
    # Attention normalization.
    # ------------------------------------------------------------

    attention_sum = outputs["query_attention"].sum(
        dim=(1, 2)
    )

    if not torch.allclose(
        attention_sum,
        torch.ones_like(attention_sum),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "Query attention does not normalize to 1."
        )

    print("Attention normalization: PASS")

    # ------------------------------------------------------------
    # Question-type routing.
    # ------------------------------------------------------------

    test_types = [
        "change_or_not",
        "increase_or_not",
        "decrease_or_not",
        "smallest_change",
        "largest_change",
        "change_to_what",
        "change_ratio",
        "change_ratio_types",
    ]

    print()
    print("Question-type head routing:")

    for question_type in test_types:
        logits = model.select_answer_logits(
            outputs,
            question_type,
        )

        print(
            f"{question_type:20s} -> "
            f"{tuple(logits.shape)}"
        )

    print()
    print("=" * 72)
    print("TEMPORAL CHANGE VQA SMOKE TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()