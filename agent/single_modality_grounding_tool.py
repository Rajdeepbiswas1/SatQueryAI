from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import importlib.util

import torch
import torch.nn as nn


PROJECT_ROOT = Path(r"D:\SatQueryAI")

MODEL_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "single_modality_grounding.py"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "single_modality_v1"
)

EXPECTED_PARAMS = 1_161_986


def _load_model_class():
    """
    Load ONLY the known, controlled SingleModalityGrounding source file.

    This intentionally does not scan or import arbitrary modules under
    src.models because older project modules may execute training code
    at import time.
    """
    if not MODEL_SOURCE.exists():
        raise FileNotFoundError(
            f"Single-modality grounding source not found: {MODEL_SOURCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "satqueryai_single_modality_grounding",
        MODEL_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import specification for: {MODEL_SOURCE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, "SingleModalityGrounding", None)

    if model_class is None:
        raise RuntimeError(
            "SingleModalityGrounding class was not found in "
            f"{MODEL_SOURCE}"
        )

    if not isinstance(model_class, type):
        raise RuntimeError(
            "SingleModalityGrounding is not a class."
        )

    if not issubclass(model_class, nn.Module):
        raise RuntimeError(
            "SingleModalityGrounding must inherit from nn.Module."
        )

    return model_class


def _load_checkpoint(path: Path) -> Tuple[dict, dict]:
    """Load and validate one frozen checkpoint."""
    if not path.exists():
        raise FileNotFoundError(
            f"Single-modality grounding checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unexpected checkpoint format: {path}"
        )

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        state_dict = checkpoint.get("state_dict")

    if state_dict is None:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Invalid model state dictionary: {path}"
        )

    return checkpoint, state_dict


def _prepare_visual(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate visual DOFA features."""
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


def _prepare_text(
    value: torch.Tensor,
) -> torch.Tensor:
    """Validate the 768-dimensional query embedding."""
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


def _sanitize_boxes(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce normalized [x1,y1,x2,y2] ordering and [0,1] bounds.
    """
    boxes = boxes.float().cpu()

    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])

    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])

    return torch.stack(
        [x1, y1, x2, y2],
        dim=1,
    ).clamp(0.0, 1.0)


def _calculate_peak_probability(
    heatmap: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate peak concentration of the normalized heatmap.

    This is an evidence/concentration diagnostic, not a calibrated
    probability of correctness.
    """
    if heatmap.ndim != 3:
        raise ValueError(
            "Grounding heatmap must have shape [B,H,W]."
        )

    flattened = heatmap.reshape(
        heatmap.shape[0],
        -1,
    )

    return flattened.amax(dim=1)


class SatQueryAISingleModalityGroundingTool:
    """
    Runtime adapter for the frozen S1/S2 Grounding V1 specialists.

    Routing:
        S1 only -> S1 Grounding V1
        S2 only -> S2 Grounding V1

    This is a text-conditioned single-modality grounding specialist.
    Paired S1+S2 grounding is handled separately by UnifiedGroundingRouter.
    """

    expected_parameter_count = EXPECTED_PARAMS

    def __init__(self) -> None:
        self.model_class = _load_model_class()

        self.models: Dict[str, nn.Module] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

        checkpoint_paths = {
            "s1": (
                CHECKPOINT_ROOT
                / "s1"
                / "grounding"
                / "best.pt"
            ),
            "s2": (
                CHECKPOINT_ROOT
                / "s2"
                / "grounding"
                / "best.pt"
            ),
        }

        for modality, checkpoint_path in checkpoint_paths.items():
            checkpoint, state_dict = _load_checkpoint(
                checkpoint_path
            )

            model_type = checkpoint.get("model_type")

            if model_type not in (
                None,
                "single_modality_grounding",
            ):
                raise RuntimeError(
                    f"{modality.upper()} checkpoint has unexpected "
                    f"model_type: {model_type}"
                )

            model = self.model_class()

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
            )

            if parameter_count != self.expected_parameter_count:
                raise RuntimeError(
                    f"{modality.upper()} Grounding V1 parameter mismatch: "
                    f"expected {self.expected_parameter_count}, "
                    f"got {parameter_count}"
                )

            # Strict loading is intentional.
            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model = model.to("cpu")
            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad = False

            self.models[modality] = model
            self.checkpoints[modality] = checkpoint

        self.parameter_count = self.expected_parameter_count

        self.checkpoint_epoch = {
            modality: self.checkpoints[modality].get("epoch")
            for modality in ("s1", "s2")
        }

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the correct frozen single-modality specialist.
        """
        del kwargs

        task = getattr(analysis, "task", None)

        if task != "grounding":
            raise ValueError(
                "Single-modality grounding received a "
                f"non-grounding request: {task!r}"
            )

        # This adapter is deliberately single-modality only.
        if s1 is not None and s2 is not None:
            raise ValueError(
                "Single-modality grounding received both S1 and S2. "
                "Use the paired UnifiedGroundingRouter."
            )

        if s1 is None and s2 is None:
            raise ValueError(
                "Single-modality grounding requires exactly one "
                "of s1 or s2."
            )

        if text_embedding is None:
            raise ValueError(
                "Single-modality grounding requires text_embedding."
            )

        if s1 is not None:
            modality = "s1"
            visual = _prepare_visual(
                s1,
                "s1",
            )
        else:
            modality = "s2"
            visual = _prepare_visual(
                s2,
                "s2",
            )

        text = _prepare_text(
            text_embedding
        )

        batch_size = visual.shape[0]

        # Permit one query embedding to be reused for a batch.
        if text.shape[0] == 1 and batch_size > 1:
            text = text.expand(
                batch_size,
                -1,
            )

        if text.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between visual features and "
                "text_embedding: "
                f"visual batch={batch_size}, "
                f"text batch={text.shape[0]}"
            )

        model = self.models[modality]

        with torch.no_grad():
            output = model(
                visual,
                text,
            )

        if not isinstance(output, dict):
            raise RuntimeError(
                "SingleModalityGrounding must return a dictionary."
            )

        required_keys = {
            "predicted_box",
            "heatmap",
            "cx",
            "cy",
            "std_x",
            "std_y",
        }

        missing = required_keys.difference(output.keys())

        if missing:
            raise RuntimeError(
                "SingleModalityGrounding output is missing required "
                f"keys: {sorted(missing)}"
            )

        boxes = output["predicted_box"]

        if not isinstance(boxes, torch.Tensor):
            raise RuntimeError(
                "predicted_box must be a torch.Tensor."
            )

        if tuple(boxes.shape) != (batch_size, 4):
            raise RuntimeError(
                "Single-modality grounding returned invalid box shape: "
                f"{tuple(boxes.shape)}; expected "
                f"({batch_size}, 4)"
            )

        if not torch.isfinite(boxes).all():
            raise RuntimeError(
                "Single-modality grounding returned non-finite boxes."
            )

        boxes = _sanitize_boxes(boxes)

        heatmap = output["heatmap"]

        if not isinstance(heatmap, torch.Tensor):
            raise RuntimeError(
                "Grounding heatmap must be a torch.Tensor."
            )

        if tuple(heatmap.shape) != (batch_size, 56, 56):
            raise RuntimeError(
                "Single-modality grounding returned invalid heatmap "
                f"shape: {tuple(heatmap.shape)}; expected "
                f"({batch_size}, 56, 56)"
            )

        if not torch.isfinite(heatmap).all():
            raise RuntimeError(
                "Single-modality grounding returned a non-finite heatmap."
            )

        heatmap = heatmap.float().cpu()

        peak_probabilities = _calculate_peak_probability(
            heatmap
        )

        result: Dict[str, Any] = {
            "answer": "grounded region",
            "bbox": (
                boxes[0].tolist()
                if batch_size == 1
                else boxes.tolist()
            ),
            "predicted_box": boxes,
            "specialist": (
                f"SingleModality"
                f"{modality.upper()}"
                f"GroundingV1"
            ),
            "modality": modality,
            "model_source": (
                "src.models.single_modality_grounding."
                "SingleModalityGrounding"
            ),
            "checkpoint_epoch": self.checkpoints[
                modality
            ].get("epoch"),
            "parameter_count": self.parameter_count,
            "query": query,
            "grounding_mode": "reference",
            "bbox_source": "single_modality_grounding_v1",
            "evidence_confidence": float(
                peak_probabilities.mean().item()
            ),
            "peak_probability": (
                float(peak_probabilities[0].item())
                if batch_size == 1
                else peak_probabilities.tolist()
            ),
            "cx": (
                output["cx"][0].item()
                if batch_size == 1
                else output["cx"].tolist()
            ),
            "cy": (
                output["cy"][0].item()
                if batch_size == 1
                else output["cy"].tolist()
            ),
            "std_x": (
                output["std_x"][0].item()
                if batch_size == 1
                else output["std_x"].tolist()
            ),
            "std_y": (
                output["std_y"][0].item()
                if batch_size == 1
                else output["std_y"].tolist()
            ),
            "heatmap": heatmap,
        }

        # Preserve the coarse representation when useful for diagnostics.
        if isinstance(output.get("coarse_logits"), torch.Tensor):
            result["coarse_logits"] = (
                output["coarse_logits"].float().cpu()
            )

        if isinstance(output.get("heatmap_logits"), torch.Tensor):
            result["heatmap_logits"] = (
                output["heatmap_logits"].float().cpu()
            )

        return result

    dispatch = __call__


def build_single_modality_grounding_tool():
    return SatQueryAISingleModalityGroundingTool()