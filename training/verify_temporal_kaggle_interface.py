from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(r"D:\SatQueryAI")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.temporal_change_vqa import TemporalChangeVQA


def describe_value(
    name: str,
    value,
    indent: int = 2,
) -> None:

    prefix = " " * indent

    if torch.is_tensor(value):

        finite = bool(
            torch.isfinite(value).all()
        )

        print(
            f"{prefix}{name:32s}"
            f"tensor "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype} "
            f"finite={finite}"
        )

        return

    if isinstance(value, dict):

        print(
            f"{prefix}{name:32s}"
            f"dict "
            f"keys={list(value.keys())}"
        )

        for child_name, child_value in value.items():

            describe_value(
                child_name,
                child_value,
                indent + 4,
            )

        return

    print(
        f"{prefix}{name:32s}"
        f"{type(value).__name__}"
    )


def main() -> None:

    print("=" * 72)
    print("SATQUERYAI — EXACT TEMPORAL MODEL OUTPUT INTERFACE")
    print("=" * 72)

    torch.set_num_threads(2)

    device = torch.device("cpu")

    print("Device:", device)

    # --------------------------------------------------------
    # Build exact frozen model.
    # --------------------------------------------------------

    model = TemporalChangeVQA()
    model.to(device)
    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("Model:", type(model).__name__)
    print("Parameters:", parameter_count)

    if parameter_count != 1_744_925:

        raise RuntimeError(
            "Parameter-count mismatch.\n"
            f"Expected: 1,744,925\n"
            f"Actual:   {parameter_count}"
        )

    print("Parameter count: PASS")

    # --------------------------------------------------------
    # Real verified input shapes.
    # --------------------------------------------------------

    batch_size = 4

    t1 = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
        device=device,
    )

    t2 = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
        device=device,
    )

    diff = torch.randn(
        batch_size,
        196,
        768,
        dtype=torch.float16,
        device=device,
    )

    question = torch.randn(
        batch_size,
        768,
        dtype=torch.float32,
        device=device,
    )

    print()
    print("Input shapes:")
    print("  t1       :", tuple(t1.shape))
    print("  t2       :", tuple(t2.shape))
    print("  diff     :", tuple(diff.shape))
    print("  question :", tuple(question.shape))

    # --------------------------------------------------------
    # Forward.
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            t1,
            t2,
            diff,
            question,
        )

    if not isinstance(outputs, dict):

        raise RuntimeError(
            "Model did not return a dictionary.\n"
            f"Returned: {type(outputs).__name__}"
        )

    # --------------------------------------------------------
    # Exact recursive output description.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EXACT RETURNED OUTPUT STRUCTURE")
    print("=" * 72)

    for key, value in outputs.items():

        describe_value(
            key,
            value,
            indent=2,
        )

    # --------------------------------------------------------
    # Core output checks.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CORE OUTPUT CHECKS")
    print("=" * 72)

    required_tensor_outputs = {
        "query_attention": (
            batch_size,
            14,
            14,
        ),
        "change_logits": (
            batch_size,
            14,
            14,
        ),
        "fused_tokens": (
            batch_size,
            196,
            256,
        ),
        "joint": (
            batch_size,
            256,
        ),
    }

    for key, expected_shape in (
        required_tensor_outputs.items()
    ):

        if key not in outputs:

            raise RuntimeError(
                f"Missing required output: {key}"
            )

        value = outputs[key]

        if not torch.is_tensor(value):

            raise RuntimeError(
                f"Expected tensor for '{key}', "
                f"got {type(value).__name__}"
            )

        actual_shape = tuple(
            value.shape
        )

        finite = bool(
            torch.isfinite(value).all()
        )

        print(
            f"{key:24s}"
            f"expected={expected_shape} "
            f"actual={actual_shape} "
            f"finite={finite}"
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{key} shape mismatch."
            )

        if not finite:
            raise RuntimeError(
                f"{key} contains non-finite values."
            )

    print(
        "Core tensor outputs: PASS"
    )

    # --------------------------------------------------------
    # answer_logits structure.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ANSWER-LOGIT STRUCTURE")
    print("=" * 72)

    if "answer_logits" not in outputs:

        raise RuntimeError(
            "answer_logits is missing."
        )

    answer_logits = outputs[
        "answer_logits"
    ]

    if not isinstance(
        answer_logits,
        dict,
    ):

        raise RuntimeError(
            "Expected answer_logits to be a dict, "
            f"got {type(answer_logits).__name__}"
        )

    print(
        "answer_logits keys:",
        list(answer_logits.keys()),
    )

    for key, value in answer_logits.items():

        describe_value(
            key,
            value,
            indent=2,
        )

    # --------------------------------------------------------
    # Validate every answer-logit tensor.
    # --------------------------------------------------------

    expected_answer_heads = {
        "binary": 2,
        "land_cover": 6,
        "ratio": 11,
        "ratio_type": 9,
    }

    print()
    print("Expected classification heads:")

    for head_name, class_count in (
        expected_answer_heads.items()
    ):

        if head_name not in answer_logits:

            print(
                f"  {head_name:16s}"
                f"NOT FOUND"
            )

            continue

        value = answer_logits[
            head_name
        ]

        if not torch.is_tensor(value):

            raise RuntimeError(
                f"Answer head '{head_name}' "
                f"is not a tensor."
            )

        expected_shape = (
            batch_size,
            class_count,
        )

        actual_shape = tuple(
            value.shape
        )

        print(
            f"  {head_name:16s}"
            f"expected={expected_shape} "
            f"actual={actual_shape}"
        )

        if actual_shape != expected_shape:

            raise RuntimeError(
                f"Answer head '{head_name}' "
                f"shape mismatch."
            )

        if not torch.isfinite(value).all():

            raise RuntimeError(
                f"Non-finite values in "
                f"answer head '{head_name}'."
            )

    # --------------------------------------------------------
    # Final output structure report.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL INTERFACE REPORT")
    print("=" * 72)

    print("Model parameter count: PASS")
    print("Forward execution: PASS")
    print("Core tensor outputs: PASS")
    print(
        "Nested answer-logit inspection: COMPLETE"
    )

    print()
    print(
        "Do NOT start Kaggle training yet."
    )

    print(
        "The exact answer_logits keys above will "
        "determine the final routed-loss implementation."
    )


if __name__ == "__main__":
    main()