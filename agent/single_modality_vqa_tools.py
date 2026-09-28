"""
SatQueryAI — True Single-Modality VQA Runtime Tools

Loads the four trained single-modality checkpoints:

    S1 Binary
    S2 Binary
    S1 MCQ
    S2 MCQ

and dispatches to the correct specialist according to which single
visual modality is supplied at runtime.

The existing paired S1+S2 VQA tools are intentionally untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.models.single_modality_vqa import (
    SingleModalityBinaryVQA,
    SingleModalityOptionAwareMCQVQA,
    count_parameters,
    load_single_modality_checkpoint,
)


PROJECT_ROOT = Path(r"D:\SatQueryAI")
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "single_modality_v1"
)

OPTION_LABELS = ("A", "B", "C", "D")


def _prepare_visual(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if value.ndim == 2:
        value = value.unsqueeze(0)

    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B,196,768], "
            f"got {tuple(value.shape)}"
        )

    if tuple(value.shape[1:]) != (196, 768):
        raise ValueError(
            f"{name} must have shape [B,196,768], "
            f"got {tuple(value.shape)}"
        )

    value = value.float().cpu()

    if not torch.isfinite(value).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return value


def _prepare_question(
    value: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "text_embedding must be a torch.Tensor."
        )

    if value.ndim == 1:
        value = value.unsqueeze(0)

    if value.ndim != 2 or value.shape[1] != 768:
        raise ValueError(
            "text_embedding must have shape [B,768], "
            f"got {tuple(value.shape)}"
        )

    value = value.float().cpu()

    if not torch.isfinite(value).all():
        raise ValueError(
            "text_embedding contains non-finite values."
        )

    return value


def _prepare_options(
    value: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "options must be a torch.Tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "options must have shape [B,4,768], "
            f"got {tuple(value.shape)}"
        )

    if tuple(value.shape[1:]) != (4, 768):
        raise ValueError(
            "options must have shape [B,4,768], "
            f"got {tuple(value.shape)}"
        )

    value = value.float().cpu()

    if not torch.isfinite(value).all():
        raise ValueError(
            "options contain non-finite values."
        )

    return value


def _select_single_modality(
    s1: Optional[torch.Tensor],
    s2: Optional[torch.Tensor],
) -> Tuple[str, torch.Tensor]:
    """
    Exactly one modality must be supplied.

    S1 + S2 together belong to the dedicated cross-modal capability.
    """

    has_s1 = s1 is not None
    has_s2 = s2 is not None

    if has_s1 and has_s2:
        raise ValueError(
            "Single-modality VQA received both S1 and S2. "
            "Use the cross-modal VQA capability for paired input."
        )

    if not has_s1 and not has_s2:
        raise ValueError(
            "Single-modality VQA requires exactly one visual input: "
            "either s1 or s2."
        )

    if has_s1:
        return "s1", _prepare_visual(s1, "s1")

    return "s2", _prepare_visual(s2, "s2")


def _get_validation_accuracy(
    checkpoint: Dict[str, Any],
) -> Optional[float]:
    for key in (
        "val_accuracy",
        "best_val_accuracy",
        "validation_accuracy",
    ):
        value = checkpoint.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


class SatQueryAISingleModalityBinaryVQATool:
    """
    Runtime dispatcher for the two trained single-modality Binary VQA models.
    """

    expected_parameter_count = 558_722

    def __init__(self) -> None:
        self.models: Dict[str, SingleModalityBinaryVQA] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

        paths = {
            "s1": (
                CHECKPOINT_ROOT
                / "s1"
                / "binary_vqa"
                / "best.pt"
            ),
            "s2": (
                CHECKPOINT_ROOT
                / "s2"
                / "binary_vqa"
                / "best.pt"
            ),
        }

        for modality, path in paths.items():
            model = SingleModalityBinaryVQA()

            checkpoint = load_single_modality_checkpoint(
                model,
                str(path),
                device="cpu",
            )

            parameter_count = count_parameters(model)

            if parameter_count != self.expected_parameter_count:
                raise RuntimeError(
                    f"{modality.upper()} Binary parameter mismatch: "
                    f"expected {self.expected_parameter_count}, "
                    f"got {parameter_count}"
                )

            for parameter in model.parameters():
                parameter.requires_grad = False

            model.eval()

            self.models[modality] = model
            self.checkpoints[modality] = checkpoint

        self.model_name = (
            "SingleModalityBinaryVQA"
        )

        self.parameter_count = (
            self.expected_parameter_count
        )

        self.checkpoint_epoch = {
            modality: self.checkpoints[modality].get("epoch")
            for modality in ("s1", "s2")
        }

        self.recorded_validation_accuracy = {
            modality: _get_validation_accuracy(
                self.checkpoints[modality]
            )
            for modality in ("s1", "s2")
        }

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        options=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        del options

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "Single-modality Binary VQA received a "
                "non-single-image-VQA request."
            )

        modality, visual = _select_single_modality(
            s1,
            s2,
        )

        if text_embedding is None:
            raise ValueError(
                "Binary VQA requires text_embedding."
            )

        question = _prepare_question(
            text_embedding
        )

        batch_size = visual.shape[0]

        if question.shape[0] == 1 and batch_size > 1:
            question = question.expand(
                batch_size,
                -1,
            )

        if question.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between visual input and "
                "text_embedding."
            )

        model = self.models[modality]

        with torch.no_grad():
            output = model(
                visual,
                question,
                return_attention=True,
            )

        logits = output["logits"]

        if tuple(logits.shape) != (
            batch_size,
            2,
        ):
            raise RuntimeError(
                "Binary VQA model returned unexpected logits shape: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Binary VQA logits contain non-finite values."
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = logits.argmax(
            dim=1
        )

        answers = [
            "yes" if int(index.item()) == 1 else "no"
            for index in predictions
        ]

        confidences = [
            float(
                probabilities[
                    i,
                    predictions[i],
                ].item()
            )
            for i in range(batch_size)
        ]

        result = {
            "answer": (
                answers[0]
                if batch_size == 1
                else answers
            ),
            "answer_confidence": (
                confidences[0]
                if batch_size == 1
                else confidences
            ),
            "probabilities": probabilities.cpu(),
            "logits": logits.cpu(),
            "specialist": (
                f"SingleModality{modality.upper()}BinaryVQA"
            ),
            "modality": modality,
            "checkpoint_epoch": (
                self.checkpoints[modality].get("epoch")
            ),
            "parameter_count": self.parameter_count,
            "recorded_validation_accuracy": (
                _get_validation_accuracy(
                    self.checkpoints[modality]
                )
            ),
            "question": query,
        }

        if "attention" in output:
            result["visual_question_attention"] = (
                output["attention"].cpu()
            )

        return result

    dispatch = __call__


class SatQueryAISingleModalityMCQVQATool:
    """
    Runtime dispatcher for the two trained single-modality MCQ models.
    """

    expected_parameter_count = 920_065

    def __init__(self) -> None:
        self.models: Dict[
            str,
            SingleModalityOptionAwareMCQVQA,
        ] = {}

        self.checkpoints: Dict[str, Dict[str, Any]] = {}

        paths = {
            "s1": (
                CHECKPOINT_ROOT
                / "s1"
                / "mcq_vqa"
                / "best.pt"
            ),
            "s2": (
                CHECKPOINT_ROOT
                / "s2"
                / "mcq_vqa"
                / "best.pt"
            ),
        }

        for modality, path in paths.items():
            model = SingleModalityOptionAwareMCQVQA()

            checkpoint = load_single_modality_checkpoint(
                model,
                str(path),
                device="cpu",
            )

            parameter_count = count_parameters(model)

            if parameter_count != self.expected_parameter_count:
                raise RuntimeError(
                    f"{modality.upper()} MCQ parameter mismatch: "
                    f"expected {self.expected_parameter_count}, "
                    f"got {parameter_count}"
                )

            for parameter in model.parameters():
                parameter.requires_grad = False

            model.eval()

            self.models[modality] = model
            self.checkpoints[modality] = checkpoint

        self.model_name = (
            "SingleModalityOptionAwareMCQVQA"
        )

        self.parameter_count = (
            self.expected_parameter_count
        )

        self.checkpoint_epoch = {
            modality: self.checkpoints[modality].get("epoch")
            for modality in ("s1", "s2")
        }

        self.recorded_validation_accuracy = {
            modality: _get_validation_accuracy(
                self.checkpoints[modality]
            )
            for modality in ("s1", "s2")
        }

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        options=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "Single-modality MCQ VQA received a "
                "non-single-image-VQA request."
            )

        modality, visual = _select_single_modality(
            s1,
            s2,
        )

        if text_embedding is None:
            raise ValueError(
                "MCQ VQA requires text_embedding."
            )

        if options is None:
            raise ValueError(
                "MCQ VQA requires four option embeddings."
            )

        question = _prepare_question(
            text_embedding
        )

        option_tensor = _prepare_options(
            options
        )

        batch_size = visual.shape[0]

        if question.shape[0] == 1 and batch_size > 1:
            question = question.expand(
                batch_size,
                -1,
            )

        if option_tensor.shape[0] == 1 and batch_size > 1:
            option_tensor = option_tensor.expand(
                batch_size,
                -1,
                -1,
            )

        if question.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between visual input and "
                "text_embedding."
            )

        if option_tensor.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between visual input and options."
            )

        model = self.models[modality]

        with torch.no_grad():
            output = model(
                visual,
                question,
                option_tensor,
                return_attention=True,
            )

        logits = output["logits"]

        if tuple(logits.shape) != (
            batch_size,
            4,
        ):
            raise RuntimeError(
                "MCQ VQA model returned unexpected logits shape: "
                f"{tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "MCQ VQA logits contain non-finite values."
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = logits.argmax(
            dim=1
        )

        answers = [
            OPTION_LABELS[int(index.item())]
            for index in predictions
        ]

        confidences = [
            float(
                probabilities[
                    i,
                    predictions[i],
                ].item()
            )
            for i in range(batch_size)
        ]

        result = {
            "answer": (
                answers[0]
                if batch_size == 1
                else answers
            ),
            "answer_confidence": (
                confidences[0]
                if batch_size == 1
                else confidences
            ),
            "probabilities": probabilities.cpu(),
            "logits": logits.cpu(),
            "specialist": (
                f"SingleModality{modality.upper()}MCQVQA"
            ),
            "modality": modality,
            "checkpoint_epoch": (
                self.checkpoints[modality].get("epoch")
            ),
            "parameter_count": self.parameter_count,
            "recorded_validation_accuracy": (
                _get_validation_accuracy(
                    self.checkpoints[modality]
                )
            ),
            "question": query,
            "options": OPTION_LABELS,
        }

        if "attention" in output:
            result["visual_question_attention"] = (
                output["attention"].cpu()
            )

        return result

    dispatch = __call__


def build_single_modality_binary_vqa_tool():
    return SatQueryAISingleModalityBinaryVQATool()


def build_single_modality_mcq_vqa_tool():
    return SatQueryAISingleModalityMCQVQATool()