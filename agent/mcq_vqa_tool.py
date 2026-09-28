from pathlib import Path
import importlib
import inspect
import sys
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(r"D:\SatQueryAI")

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "mcq_vqa_option_aware"
    / "best.pt"
)

EXPECTED_PARAMS = 1_380_097


def _parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def _load_checkpoint():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"MCQ checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected MCQ checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Invalid MCQ checkpoint state dictionary."
        )

    return checkpoint, state_dict


def _find_model_class():
    """
    Import the actual existing MCQ source and identify the model
    class defined there.

    We deliberately do not duplicate the architecture here.
    """

    module = importlib.import_module(
        "models.mcq_vqa_option_aware"
    )

    candidates = []

    for name, obj in vars(module).items():

        if not inspect.isclass(obj):
            continue

        if obj.__module__ != module.__name__:
            continue

        if obj is nn.Module:
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
                (name, obj)
            )

    if len(candidates) != 1:

        names = [
            name
            for name, _ in candidates
        ]

        raise RuntimeError(
            "Could not uniquely identify the MCQ model class. "
            f"Candidates: {names}"
        )

    return candidates[0]


class SatQueryAIMCQVQATool:
    """
    Frozen option-aware MCQ VQA adapter.

    Expected inputs:

        s1:
            [B, 196, 768]

        s2:
            [B, 196, 768]

        text_embedding:
            [B, 768]

        options:
            [B, 4, 768]

    Output:

        A / B / C / D
        probabilities
        logits
        model/checkpoint metadata
    """

    def __init__(self) -> None:

        model_name, model_class = (
            _find_model_class()
        )

        checkpoint, state_dict = (
            _load_checkpoint()
        )

        model = model_class()

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        parameter_count = (
            _parameter_count(model)
        )

        if parameter_count != EXPECTED_PARAMS:

            raise RuntimeError(
                "MCQ parameter-count mismatch: "
                f"expected {EXPECTED_PARAMS}, "
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
            checkpoint.get("val_accuracy")
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

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "SatQueryAIMCQVQATool received a "
                "non-single-image-VQA request."
            )

        if s1 is None:
            raise ValueError(
                "MCQ VQA requires s1."
            )

        if s2 is None:
            raise ValueError(
                "MCQ VQA requires s2."
            )

        if text_embedding is None:
            raise ValueError(
                "MCQ VQA requires text_embedding."
            )

        if options is None:
            raise ValueError(
                "MCQ VQA requires four option embeddings."
            )

        if not isinstance(s1, torch.Tensor):
            raise TypeError(
                "s1 must be a torch.Tensor."
            )

        if not isinstance(s2, torch.Tensor):
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

        if not isinstance(
            options,
            torch.Tensor,
        ):
            raise TypeError(
                "options must be a torch.Tensor."
            )

        if s1.ndim == 2:
            s1 = s1.unsqueeze(0)

        if s2.ndim == 2:
            s2 = s2.unsqueeze(0)

        if text_embedding.ndim == 1:
            text_embedding = (
                text_embedding.unsqueeze(0)
            )

        if options.ndim == 2:
            options = options.unsqueeze(0)

        if s1.ndim != 3:
            raise ValueError(
                f"Expected s1 [B,196,768], "
                f"got {tuple(s1.shape)}"
            )

        if s2.ndim != 3:
            raise ValueError(
                f"Expected s2 [B,196,768], "
                f"got {tuple(s2.shape)}"
            )

        if text_embedding.ndim != 2:
            raise ValueError(
                "Expected text_embedding [B,768], "
                f"got {tuple(text_embedding.shape)}"
            )

        if options.ndim != 3:
            raise ValueError(
                "Expected options [B,4,768], "
                f"got {tuple(options.shape)}"
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

        if options.shape[1:] != (
            4,
            768,
        ):
            raise ValueError(
                f"Invalid option shape: "
                f"{tuple(options.shape)}"
            )

        batch_size = s1.shape[0]

        if not (
            s2.shape[0]
            == text_embedding.shape[0]
            == options.shape[0]
            == batch_size
        ):
            raise ValueError(
                "Batch-size mismatch between "
                "MCQ inputs."
            )

        for name, tensor in [
            ("s1", s1),
            ("s2", s2),
            ("text_embedding", text_embedding),
            ("options", options),
        ]:

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} contains non-finite values."
                )

        s1 = s1.float().cpu()
        s2 = s2.float().cpu()
        text_embedding = (
            text_embedding.float().cpu()
        )
        options = options.float().cpu()

        # --------------------------------------------------------
        # Determine the actual model forward contract.
        # --------------------------------------------------------

        signature = inspect.signature(
            self.model.forward
        )

        parameter_names = [
            name
            for name in signature.parameters
        ]

        input_map = {
            "s1": s1,
            "s2": s2,
            "question": text_embedding,
            "text_embedding": text_embedding,
            "options": options,
            "question_options": options,
        }

        kwargs_for_model = {}

        for name in parameter_names:

            if name not in input_map:
                continue

            kwargs_for_model[name] = (
                input_map[name]
            )

        required_parameters = [
            parameter
            for parameter in signature.parameters.values()
            if (
                parameter.default
                is inspect.Parameter.empty
            )
        ]

        missing = [
            parameter.name
            for parameter in required_parameters
            if parameter.name not in kwargs_for_model
        ]

        if missing:
            raise RuntimeError(
                "Unable to satisfy MCQ model forward "
                f"contract. Missing: {missing}. "
                f"Forward parameters: {parameter_names}"
            )

        # --------------------------------------------------------
        # Frozen inference.
        # --------------------------------------------------------

        with torch.no_grad():

            logits = self.model(
                **kwargs_for_model
            )

        if not isinstance(
            logits,
            torch.Tensor,
        ):
            raise RuntimeError(
                "MCQ model forward did not return "
                "a tensor."
            )

        if logits.ndim != 2:
            raise RuntimeError(
                "Expected MCQ logits [B,4], "
                f"got {tuple(logits.shape)}"
            )

        if logits.shape != (
            batch_size,
            4,
        ):
            raise RuntimeError(
                "Expected four MCQ output classes, "
                f"got {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "MCQ logits contain non-finite values."
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
            for i in range(batch_size)
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
            "logits": logits.cpu(),
            "specialist": self.model_name,
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


def build_mcq_vqa_tool():
    return SatQueryAIMCQVQATool()


if __name__ == "__main__":

    print("=" * 72)
    print(
        "SATQUERYAI — MCQ VQA TOOL INITIALIZATION TEST"
    )
    print("=" * 72)

    tool = build_mcq_vqa_tool()

    print(
        "Model class:",
        tool.model_name,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Checkpoint epoch:",
        tool.checkpoint_epoch,
    )

    print(
        "Parameter count:",
        tool.parameter_count,
    )

    print(
        "Recorded validation accuracy:",
        tool.recorded_validation_accuracy,
    )

    if tool.parameter_count != EXPECTED_PARAMS:
        raise RuntimeError(
            "MCQ parameter-count verification failed."
        )

    print(
        "MCQ VQA tool initialization: PASS"
    )

    print("=" * 72)