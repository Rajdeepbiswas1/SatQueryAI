from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_DIR))

from models.binary_vqa_cached import CachedBinaryVQA


CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "binary_vqa"
    / "best.pt"
)

EXPECTED_PARAMS = 1_051_906


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def _load_checkpoint():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Binary VQA checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected Binary VQA checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Invalid Binary VQA state dictionary."
        )

    return checkpoint, state_dict


class SatQueryAIBinaryVQATool:
    """
    Frozen Binary VQA specialist adapter.

    Agent contract:
        query
        analysis
        s1
        s2
        text_embedding

    Specialist contract:
        s1       -> [B, 196, 768]
        s2       -> [B, 196, 768]
        question -> [B, 768]

    Output:
        answer       = yes / no
        probabilities
        answer_confidence
        logits
        specialist
        checkpoint_epoch
    """

    def __init__(self) -> None:

        checkpoint, state_dict = _load_checkpoint()

        model = CachedBinaryVQA()

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        parameter_count = _parameter_count(model)

        if parameter_count != EXPECTED_PARAMS:
            raise RuntimeError(
                "Binary VQA parameter-count mismatch: "
                f"expected {EXPECTED_PARAMS}, "
                f"got {parameter_count}"
            )

        # This specialist is frozen for inference.
        model = model.to("cpu")
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.parameter_count = parameter_count
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.recorded_validation_accuracy = checkpoint.get(
            "val_accuracy"
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

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "SatQueryAIBinaryVQATool received a "
                "non-single-image-VQA request."
            )

        if s1 is None:
            raise ValueError(
                "Binary VQA requires s1."
            )

        if s2 is None:
            raise ValueError(
                "Binary VQA requires s2."
            )

        if text_embedding is None:
            raise ValueError(
                "Binary VQA requires text_embedding."
            )

        if not isinstance(s1, torch.Tensor):
            raise TypeError("s1 must be a torch.Tensor.")

        if not isinstance(s2, torch.Tensor):
            raise TypeError("s2 must be a torch.Tensor.")

        if not isinstance(
            text_embedding,
            torch.Tensor,
        ):
            raise TypeError(
                "text_embedding must be a torch.Tensor."
            )

        # Accept a single sample as [196, 768] / [768]
        # and convert it to a batch of one.
        if s1.ndim == 2:
            s1 = s1.unsqueeze(0)

        if s2.ndim == 2:
            s2 = s2.unsqueeze(0)

        if text_embedding.ndim == 1:
            text_embedding = text_embedding.unsqueeze(0)

        if s1.ndim != 3:
            raise ValueError(
                f"Expected s1 [B,196,768], got {tuple(s1.shape)}"
            )

        if s2.ndim != 3:
            raise ValueError(
                f"Expected s2 [B,196,768], got {tuple(s2.shape)}"
            )

        if text_embedding.ndim != 2:
            raise ValueError(
                "Expected text_embedding [B,768], "
                f"got {tuple(text_embedding.shape)}"
            )

        if s1.shape[1:] != (196, 768):
            raise ValueError(
                f"Invalid s1 shape: {tuple(s1.shape)}"
            )

        if s2.shape[1:] != (196, 768):
            raise ValueError(
                f"Invalid s2 shape: {tuple(s2.shape)}"
            )

        if text_embedding.shape[1] != 768:
            raise ValueError(
                f"Invalid text embedding shape: "
                f"{tuple(text_embedding.shape)}"
            )

        if (
            s1.shape[0]
            != s2.shape[0]
            or s1.shape[0]
            != text_embedding.shape[0]
        ):
            raise ValueError(
                "Batch-size mismatch between s1, s2, "
                "and text_embedding."
            )

        if not torch.isfinite(s1).all():
            raise ValueError("s1 contains non-finite values.")

        if not torch.isfinite(s2).all():
            raise ValueError("s2 contains non-finite values.")

        if not torch.isfinite(text_embedding).all():
            raise ValueError(
                "text_embedding contains non-finite values."
            )

        s1 = s1.float().to("cpu")
        s2 = s2.float().to("cpu")
        text_embedding = text_embedding.float().to("cpu")

        with torch.no_grad():
            logits = self.model(
                s1,
                s2,
                text_embedding,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = logits.argmax(
                dim=1
            )

        answers = [
            "yes" if int(index.item()) == 1 else "no"
            for index in predicted_class
        ]

        confidences = [
            float(
                probabilities[i, predicted_class[i]].item()
            )
            for i in range(
                predicted_class.shape[0]
            )
        ]

        return {
            "answer": (
                answers[0]
                if len(answers) == 1
                else answers
            ),
            "answer_confidence": (
                confidences[0]
                if len(confidences) == 1
                else confidences
            ),
            "probabilities": probabilities.detach().cpu(),
            "logits": logits.detach().cpu(),
            "specialist": "CachedBinaryVQA",
            "checkpoint_epoch": self.checkpoint_epoch,
            "parameter_count": self.parameter_count,
            "recorded_validation_accuracy": (
                self.recorded_validation_accuracy
            ),
        }


def build_binary_vqa_tool() -> SatQueryAIBinaryVQATool:
    return SatQueryAIBinaryVQATool()


if __name__ == "__main__":

    print("=" * 72)
    print("SATQUERYAI — BINARY VQA TOOL INITIALIZATION TEST")
    print("=" * 72)

    tool = build_binary_vqa_tool()

    print("Specialist: CachedBinaryVQA")
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint epoch:", tool.checkpoint_epoch)
    print("Parameter count:", tool.parameter_count)
    print(
        "Recorded validation accuracy:",
        tool.recorded_validation_accuracy,
    )

    if tool.parameter_count != EXPECTED_PARAMS:
        raise RuntimeError(
            "Binary VQA parameter-count verification failed."
        )

    print("Binary VQA tool initialization: PASS")
    print("=" * 72)