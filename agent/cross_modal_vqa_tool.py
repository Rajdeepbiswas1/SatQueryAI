from pathlib import Path
from typing import Any
import importlib
import inspect

import torch
import torch.nn as nn


PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)


# =============================================================================
# CHECKPOINT PATHS
# =============================================================================

BINARY_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "cross_modal_vqa"
    / "binary"
    / "best.pt"
)

MCQ_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "cross_modal_vqa"
    / "mcq"
    / "best.pt"
)


EXPECTED_BINARY_PARAMS = 1_710_338
EXPECTED_MCQ_PARAMS = 2_104_065


# =============================================================================
# UTILITIES
# =============================================================================

def _parameter_count(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def _load_checkpoint(
    checkpoint_path: Path,
):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cross-modal checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected cross-modal checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "Invalid cross-modal state dictionary."
        )

    return checkpoint, state_dict


def _validate_common_inputs(
    s1,
    s2,
    text_embedding,
):

    if not isinstance(
        s1,
        torch.Tensor,
    ):
        raise TypeError(
            "s1 must be a torch.Tensor."
        )

    if not isinstance(
        s2,
        torch.Tensor,
    ):
        raise TypeError(
            "s2 must be a torch.Tensor."
        )

    if not isinstance(
        text_embedding,
        torch.Tensor,
    ):
        raise TypeError(
            "text_embedding must be a torch.Tensor."
        )

    if s1.ndim == 2:
        s1 = s1.unsqueeze(0)

    if s2.ndim == 2:
        s2 = s2.unsqueeze(0)

    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)

    if s1.ndim != 3:
        raise ValueError(
            "Expected s1 [B,196,768], "
            f"got {tuple(s1.shape)}"
        )

    if s2.ndim != 3:
        raise ValueError(
            "Expected s2 [B,196,768], "
            f"got {tuple(s2.shape)}"
        )

    if text_embedding.ndim != 2:
        raise ValueError(
            "Expected text_embedding [B,768], "
            f"got {tuple(text_embedding.shape)}"
        )

    if s1.shape[1:] != (
        196,
        768,
    ):
        raise ValueError(
            f"Invalid s1 shape: {tuple(s1.shape)}"
        )

    if s2.shape[1:] != (
        196,
        768,
    ):
        raise ValueError(
            f"Invalid s2 shape: {tuple(s2.shape)}"
        )

    if text_embedding.shape[1] != 768:
        raise ValueError(
            "Invalid text embedding shape: "
            f"{tuple(text_embedding.shape)}"
        )

    batch_size = s1.shape[0]

    if (
        s2.shape[0] != batch_size
        or text_embedding.shape[0] != batch_size
    ):
        raise ValueError(
            "Batch-size mismatch between "
            "s1, s2, and text_embedding."
        )

    for name, tensor in [
        ("s1", s1),
        ("s2", s2),
        ("text_embedding", text_embedding),
    ]:

        if not torch.isfinite(
            tensor
        ).all():

            raise ValueError(
                f"{name} contains non-finite values."
            )

    return (
        s1.float().cpu(),
        s2.float().cpu(),
        text_embedding.float().cpu(),
        batch_size,
    )


def _standardize_evidence(
    evidence,
):
    """
    Preserve the native cross-modal evidence while adding
    simple descriptive metadata for the agent/frontend.
    """

    if not isinstance(
        evidence,
        dict,
    ):
        raise RuntimeError(
            "Cross-modal model returned "
            "invalid evidence."
        )

    standardized = dict(
        evidence
    )

    if "s1_to_s2_attention" in evidence:
        standardized[
            "cross_attention_s1_to_s2"
        ] = evidence[
            "s1_to_s2_attention"
        ]

    if "s2_to_s1_attention" in evidence:
        standardized[
            "cross_attention_s2_to_s1"
        ] = evidence[
            "s2_to_s1_attention"
        ]

    if "query_attention" in evidence:
        standardized[
            "spatial_query_attention"
        ] = evidence[
            "query_attention"
        ]

    standardized[
        "evidence_type"
    ] = "cross_modal_joint"

    return standardized


# =============================================================================
# CROSS-MODAL BINARY VQA
# =============================================================================

class SatQueryAICrossModalBinaryVQATool:
    """
    Frozen Cross-Modal Binary VQA specialist.

    Inputs:
        s1              [B,196,768]
        s2              [B,196,768]
        text_embedding  [B,768]

    Output:
        yes / no
        probabilities
        logits
        joint optical/SAR evidence
        checkpoint metadata
    """

    def __init__(self) -> None:

        (
            checkpoint,
            state_dict,
        ) = _load_checkpoint(
            BINARY_CHECKPOINT
        )

        from src.models.cross_modal_vqa import (
            CrossModalVQA,
        )

        model = CrossModalVQA()

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        parameter_count = _parameter_count(
            model
        )

        if parameter_count != EXPECTED_BINARY_PARAMS:
            raise RuntimeError(
                "Cross-modal Binary VQA "
                "parameter-count mismatch: "
                f"expected {EXPECTED_BINARY_PARAMS}, "
                f"got {parameter_count}"
            )

        model = model.to("cpu")
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.parameter_count = parameter_count
        self.checkpoint_epoch = checkpoint.get(
            "epoch"
        )
        self.recorded_validation_accuracy = (
            checkpoint.get(
                "val_accuracy"
            )
        )

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        **kwargs: Any,
    ):

        del kwargs

        if analysis.task != "cross_modal":
            raise ValueError(
                "SatQueryAICrossModalBinaryVQATool "
                "received a non-cross-modal request."
            )

        if s1 is None:
            raise ValueError(
                "Cross-modal Binary VQA requires s1."
            )

        if s2 is None:
            raise ValueError(
                "Cross-modal Binary VQA requires s2."
            )

        if text_embedding is None:
            raise ValueError(
                "Cross-modal Binary VQA requires "
                "text_embedding."
            )

        (
            s1,
            s2,
            text_embedding,
            batch_size,
        ) = _validate_common_inputs(
            s1,
            s2,
            text_embedding,
        )

        with torch.no_grad():

            logits, evidence = self.model(
                s1,
                s2,
                text_embedding,
                return_evidence=True,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = logits.argmax(
                dim=1
            )

        labels = [
            "no",
            "yes",
        ]

        answers = [
            labels[int(index.item())]
            for index in predictions
        ]

        confidences = [
            float(
                probabilities[
                    i,
                    predictions[i],
                ].item()
            )
            for i in range(
                batch_size
            )
        ]

        return {
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
            "probabilities": (
                probabilities.cpu()
            ),
            "logits": (
                logits.cpu()
            ),
            "evidence": _standardize_evidence(
                evidence
            ),
            "specialist": (
                "CrossModalBinaryVQA"
            ),
            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "recorded_validation_accuracy": (
                self.recorded_validation_accuracy
            ),
        }


# =============================================================================
# CROSS-MODAL MCQ VQA
# =============================================================================

class SatQueryAICrossModalMCQVQATool:
    """
    Frozen Cross-Modal MCQ VQA specialist.

    Inputs:
        s1              [B,196,768]
        s2              [B,196,768]
        text_embedding  [B,768]
        options         [B,4,768]

    Output:
        A / B / C / D
        probabilities
        logits
        joint optical/SAR evidence
        checkpoint metadata
    """

    def __init__(self) -> None:

        (
            checkpoint,
            state_dict,
        ) = _load_checkpoint(
            MCQ_CHECKPOINT
        )

        module = importlib.import_module(
            "src.models.cross_modal_mcq_vqa"
        )

        candidates = []

        for name, obj in vars(
            module
        ).items():

            if not inspect.isclass(obj):
                continue

            if obj.__module__ != module.__name__:
                continue

            try:
                is_model = issubclass(
                    obj,
                    nn.Module,
                )
            except TypeError:
                is_model = False

            if is_model:
                candidates.append(
                    (
                        name,
                        obj,
                    )
                )

        if len(candidates) != 1:

            names = [
                name
                for name, _ in candidates
            ]

            raise RuntimeError(
                "Could not uniquely identify "
                "the Cross-Modal MCQ model class. "
                f"Candidates: {names}"
            )

        model_name, model_class = (
            candidates[0]
        )

        model = model_class(
            num_options=4
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        parameter_count = _parameter_count(
            model
        )

        if parameter_count != EXPECTED_MCQ_PARAMS:
            raise RuntimeError(
                "Cross-modal MCQ VQA "
                "parameter-count mismatch: "
                f"expected {EXPECTED_MCQ_PARAMS}, "
                f"got {parameter_count}"
            )

        model = model.to("cpu")
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.model_name = model_name
        self.parameter_count = parameter_count
        self.checkpoint_epoch = checkpoint.get(
            "epoch"
        )
        self.recorded_validation_accuracy = (
            checkpoint.get(
                "val_accuracy"
            )
        )

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        options=None,
        **kwargs: Any,
    ):

        del kwargs

        if analysis.task != "cross_modal":
            raise ValueError(
                "SatQueryAICrossModalMCQVQATool "
                "received a non-cross-modal request."
            )

        if s1 is None:
            raise ValueError(
                "Cross-modal MCQ VQA requires s1."
            )

        if s2 is None:
            raise ValueError(
                "Cross-modal MCQ VQA requires s2."
            )

        if text_embedding is None:
            raise ValueError(
                "Cross-modal MCQ VQA requires "
                "text_embedding."
            )

        if options is None:
            raise ValueError(
                "Cross-modal MCQ VQA requires "
                "four option embeddings."
            )

        (
            s1,
            s2,
            text_embedding,
            batch_size,
        ) = _validate_common_inputs(
            s1,
            s2,
            text_embedding,
        )

        if not isinstance(
            options,
            torch.Tensor,
        ):
            raise TypeError(
                "options must be a torch.Tensor."
            )

        if options.ndim == 2:
            options = options.unsqueeze(0)

        if options.ndim != 3:
            raise ValueError(
                "Expected options [B,4,768], "
                f"got {tuple(options.shape)}"
            )

        if options.shape[1:] != (
            4,
            768,
        ):
            raise ValueError(
                "Invalid option shape: "
                f"{tuple(options.shape)}"
            )

        if options.shape[0] != batch_size:
            raise ValueError(
                "Batch-size mismatch between "
                "MCQ options and cross-modal inputs."
            )

        if not torch.isfinite(
            options
        ).all():

            raise ValueError(
                "options contains non-finite values."
            )

        options = options.float().cpu()

        with torch.no_grad():

            logits, evidence = self.model(
                s1,
                s2,
                text_embedding,
                options,
                return_evidence=True,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = logits.argmax(
                dim=1
            )

        labels = [
            "A",
            "B",
            "C",
            "D",
        ]

        answers = [
            labels[int(index.item())]
            for index in predictions
        ]

        confidences = [
            float(
                probabilities[
                    i,
                    predictions[i],
                ].item()
            )
            for i in range(
                batch_size
            )
        ]

        return {
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
            "probabilities": (
                probabilities.cpu()
            ),
            "logits": (
                logits.cpu()
            ),
            "evidence": _standardize_evidence(
                evidence
            ),
            "specialist": (
                "CrossModalMCQVQA"
            ),
            "model_class": (
                self.model_name
            ),
            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "recorded_validation_accuracy": (
                self.recorded_validation_accuracy
            ),
        }


def build_cross_modal_binary_vqa_tool():
    return SatQueryAICrossModalBinaryVQATool()


def build_cross_modal_mcq_vqa_tool():
    return SatQueryAICrossModalMCQVQATool()


if __name__ == "__main__":

    print("=" * 72)
    print(
        "SATQUERYAI — CROSS-MODAL VQA TOOL INITIALIZATION TEST"
    )
    print("=" * 72)

    binary_tool = (
        build_cross_modal_binary_vqa_tool()
    )

    mcq_tool = (
        build_cross_modal_mcq_vqa_tool()
    )

    print()
    print(
        "Binary specialist:",
        "CrossModalBinaryVQA",
    )
    print(
        "Binary checkpoint:",
        BINARY_CHECKPOINT,
    )
    print(
        "Binary epoch:",
        binary_tool.checkpoint_epoch,
    )
    print(
        "Binary parameters:",
        binary_tool.parameter_count,
    )
    print(
        "Binary validation accuracy:",
        binary_tool.recorded_validation_accuracy,
    )

    print()
    print(
        "MCQ specialist:",
        "CrossModalMCQVQA",
    )
    print(
        "MCQ checkpoint:",
        MCQ_CHECKPOINT,
    )
    print(
        "MCQ epoch:",
        mcq_tool.checkpoint_epoch,
    )
    print(
        "MCQ parameters:",
        mcq_tool.parameter_count,
    )
    print(
        "MCQ validation accuracy:",
        mcq_tool.recorded_validation_accuracy,
    )

    if (
        binary_tool.parameter_count
        != EXPECTED_BINARY_PARAMS
    ):
        raise RuntimeError(
            "Cross-modal Binary parameter "
            "verification failed."
        )

    if (
        mcq_tool.parameter_count
        != EXPECTED_MCQ_PARAMS
    ):
        raise RuntimeError(
            "Cross-modal MCQ parameter "
            "verification failed."
        )

    print()
    print(
        "Cross-modal VQA tool initialization: PASS"
    )
    print("=" * 72)