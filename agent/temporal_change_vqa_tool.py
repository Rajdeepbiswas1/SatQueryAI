# =============================================================================
# SATQUERYAI — TEMPORAL CHANGE V3 AGENT TOOL
# =============================================================================
#
# Application adapter for the frozen Temporal Change V3 specialist.
#
# IMPORTANT:
#   - Does NOT modify the temporal model architecture.
#   - Does NOT train or fine-tune anything.
#   - Loads the verified V3 checkpoint strictly.
#   - Uses the existing Agent handler contract:
#
#       handler(
#           query=query,
#           analysis=analysis,
#           **inputs,
#       )
#
# Required temporal inputs:
#   s1  -> T1 temporal feature tokens [B,196,768]
#   s2  -> T2 temporal feature tokens [B,196,768]
#   diff -> temporal difference tokens [B,196,768]
#   text_embedding -> question embedding [B,768]
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import sys
import time

import numpy as np
import torch


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "temporal_change_vqa_v3_ordinal"
    / "best.pt"
)


# =============================================================================
# TEMPORAL MODEL INVARIANTS
# =============================================================================

EXPECTED_PARAMETER_COUNT = 1_679_133
EXPECTED_CHECKPOINT_EPOCH = 10

INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_SPATIAL_TOKENS = 196
GRID_SIZE = 14


# =============================================================================
# QUESTION-TYPE VOCABULARY
# =============================================================================

BINARY_VOCAB = (
    "no",
    "yes",
)

LAND_COVER_VOCAB = (
    "buildings",
    "trees",
    "low_vegetation",
    "NVG_surface",
    "water",
    "playgrounds",
)

CHANGE_RATIO_VOCAB = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
)

CHANGE_RATIO_TYPES_VOCAB = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
)


QUESTION_TYPE_TO_HEAD = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio_types",
}


# =============================================================================
# MODEL IMPORT
# =============================================================================

def _import_temporal_model():
    """
    Import the exact current TemporalChangeVQAv2 architecture.

    V3 intentionally reuses the V2 architecture.
    """

    src_root = PROJECT_ROOT / "src"

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    if str(src_root) not in sys.path:
        sys.path.insert(
            0,
            str(src_root),
        )

    try:
        from src.models.temporal_change_vqa_v2 import (
            TemporalChangeVQAv2,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import TemporalChangeVQAv2 from "
            r"D:\SatQueryAI\src\models\temporal_change_vqa_v2.py"
        ) from exc

    return TemporalChangeVQAv2


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def _load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[dict, dict]:
    """
    Load checkpoint and extract its state dictionary.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Temporal V3 checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected temporal checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Temporal checkpoint does not contain a valid state dictionary."
        )

    return checkpoint, state_dict


def _parameter_count(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =============================================================================
# INPUT CONVERSION
# =============================================================================

def _to_tensor(
    value: Any,
    name: str,
) -> torch.Tensor:
    """
    Convert numpy/list/tensor input into float32 CPU tensor.
    """

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()

    else:
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return tensor


def _ensure_temporal_batch(
    value: Any,
    name: str,
) -> torch.Tensor:
    """
    Ensure temporal spatial-token tensor has shape [B,196,768].
    """

    tensor = _to_tensor(
        value,
        name,
    )

    if name == "text_embedding":
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"{name} must be 1D or 2D before batching, "
                f"got {tuple(tensor.shape)}"
            )
    else:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(
                f"{name} must be 2D or 3D before batching, "
                f"got {tuple(tensor.shape)}"
            )

    expected = (
        INPUT_DIM
        if name == "text_embedding"
        else NUM_SPATIAL_TOKENS
    )

    if name == "text_embedding":

        if tuple(tensor.shape) != (
            1,
            INPUT_DIM,
        ):
            raise ValueError(
                f"{name} must have shape "
                f"[1,768], got {tuple(tensor.shape)}"
            )

    else:

        if tuple(tensor.shape) != (
            1,
            NUM_SPATIAL_TOKENS,
            INPUT_DIM,
        ):
            raise ValueError(
                f"{name} must have shape "
                f"[1,196,768], got {tuple(tensor.shape)}"
            )

    return tensor


# =============================================================================
# QUESTION TYPE CLASSIFICATION
# =============================================================================

def _normalize_query(
    query: str,
) -> str:

    return " ".join(
        str(query)
        .strip()
        .lower()
        .split()
    )


def _contains_any(
    query: str,
    phrases,
) -> bool:

    return any(
        phrase in query
        for phrase in phrases
    )


def _detect_question_type(
    query: str,
) -> str:
    """
    Deterministically map a natural-language temporal question to the
    question-type labels used by the temporal training manifest.

    This is intentionally rule-based and observable.
    """

    normalized = _normalize_query(
        query
    )

    # -------------------------------------------------------------------------
    # Explicit transformation / land-cover questions
    # -------------------------------------------------------------------------

    if _contains_any(
        normalized,
        (
            "change to what",
            "changed to what",
            "what did it change to",
            "what did change to what",
            "what has it changed to",
            "changed to",
            "change into",
            "changed into",
            "change into what",
            "changed into what",
            "what did it mainly change to",
            "what have the regions changed to",
            "what changed",
            "what has changed",
            "what changed between",
            "what has changed between",
        ),
    ):
        return "change_to_what"

    if _contains_any(
        normalized,
        (
            "smallest change",
            "least change",
            "smallest area of change",
            "which class changed the least",
            "what changed the least",
            "change is the smallest",
            "change is smallest",
            "smallest type of change",
        ),
    ):
        return "smallest_change"

    if _contains_any(
        normalized,
        (
            "largest change",
            "greatest change",
            "largest area of change",
            "which class changed the most",
            "what changed the most",
            "change is the largest",
            "change is largest",
            "largest type of change",
        ),
    ):
        return "largest_change"

    # -------------------------------------------------------------------------
    # Ordered change-ratio questions
    # -------------------------------------------------------------------------

    if _contains_any(
        normalized,
        (
            "change ratio type",
            "type of change ratio",
            "which change interval",
            "which percentage range",
            "which percentage interval",
            "what percentage range",
            "what percentage interval",
            "how much area of",
            "how much area has changed",
            "how much of the area",
        ),
    ):
        return "change_ratio_types"

    if _contains_any(
        normalized,
        (
            "change ratio",
            "percentage change",
            "percent change",
            "percentage of change",
            "percent of change",
            "how much has it changed",
            "how much change",
        ),
    ):
        return "change_ratio"

    # -------------------------------------------------------------------------
    # Increase / decrease binary questions
    # -------------------------------------------------------------------------

    if _contains_any(
        normalized,
        (
            "increase",
            "increased",
            "grow",
            "grew",
            "growth",
            "rise",
            "rising",
        ),
    ):
        return "increase_or_not"

    if _contains_any(
        normalized,
        (
            "decrease",
            "decreased",
            "decline",
            "declined",
            "shrink",
            "shrunk",
            "reduction",
            "reduced",
        ),
    ):
        return "decrease_or_not"

    # -------------------------------------------------------------------------
    # General change/no-change question
    # -------------------------------------------------------------------------

    if _contains_any(
        normalized,
        (
            "change",
            "changed",
            "changes",
            "before and after",
            "compare over time",
            "temporal",
            "time series",
            "previous image",
            "earlier image",
            "later image",
            "between 2020 and 2024",
            "between 2021 and 2024",
            "between 2022 and 2024",
        ),
    ):
        return "change_or_not"

    # -------------------------------------------------------------------------
    # Conservative fallback
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # No supported temporal question type detected
    # -------------------------------------------------------------------------

    raise ValueError(
        "Unsupported temporal question. "
        "SatQueryAI currently supports these temporal question types: "
        "change_or_not, increase_or_not, decrease_or_not, "
        "smallest_change, largest_change, change_to_what, "
        "change_ratio, change_ratio_types."
    )


# =============================================================================
# ANSWER DECODING
# =============================================================================

def _decode_head(
    logits: torch.Tensor,
    vocabulary,
) -> Tuple[str, float, list]:
    """
    Decode one categorical answer head.

    Returns:
        answer
        top probability
        full probability vector
    """

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    index = int(
        torch.argmax(
            probabilities,
            dim=-1,
        ).item()
    )

    answer = vocabulary[index]
    top_probability = float(
        probabilities[0, index].item()
    )

    full_probabilities = [
        float(x)
        for x in probabilities[0].tolist()
    ]

    return (
        answer,
        top_probability,
        full_probabilities,
    )


# =============================================================================
# NUMERIC EVIDENCE SUMMARIES
# =============================================================================

def _tensor_summary(
    tensor: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compact numerical summary for frontend/backend evidence.
    """

    x = tensor.detach().cpu().float()

    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "l2_norm": float(
            torch.linalg.vector_norm(
                x
            ).item()
        ),
    }


def _spatial_map_to_list(
    tensor: torch.Tensor,
) -> list:
    """
    Convert [1,14,14] or [14,14] evidence map to nested Python lists.
    """

    x = tensor.detach().cpu().float()

    if x.ndim == 3:
        x = x[0]

    if tuple(x.shape) != (
        GRID_SIZE,
        GRID_SIZE,
    ):
        raise ValueError(
            "Expected a 14x14 spatial evidence map, "
            f"got {tuple(x.shape)}"
        )

    return x.tolist()


# =============================================================================
# TEMPORAL V3 TOOL
# =============================================================================

class SatQueryAITemporalChangeV3Tool:
    """
    Agent adapter for the frozen Temporal Change V3 specialist.

    The adapter owns:
        - checkpoint loading
        - strict architecture verification
        - question-type selection
        - model inference
        - structured temporal evidence
    """

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        )

        TemporalChangeVQAv2 = (
            _import_temporal_model()
        )

        checkpoint, state_dict = (
            _load_checkpoint(
                self.checkpoint_path
            )
        )

        model = TemporalChangeVQAv2(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_spatial_tokens=NUM_SPATIAL_TOKENS,
            grid_size=GRID_SIZE,
        )

        parameter_count = _parameter_count(
            model
        )

        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "Temporal V3 parameter-count mismatch: "
                f"expected {EXPECTED_PARAMETER_COUNT}, "
                f"got {parameter_count}"
            )

        # Strict checkpoint compatibility is mandatory.
        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model = model.to("cpu")
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        checkpoint_epoch = checkpoint.get(
            "epoch"
        )

        if checkpoint_epoch is not None:
            checkpoint_epoch = int(
                checkpoint_epoch
            )

            if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
                raise RuntimeError(
                    "Unexpected Temporal V3 checkpoint epoch: "
                    f"expected {EXPECTED_CHECKPOINT_EPOCH}, "
                    f"got {checkpoint_epoch}"
                )

        self.model = model
        self.checkpoint_metadata = checkpoint
        self.parameter_count = parameter_count
        self.checkpoint_epoch = checkpoint_epoch

    # -------------------------------------------------------------------------
    # Agent handler
    # -------------------------------------------------------------------------

    def __call__(
        self,
        query: str,
        analysis,
        t1=None,
        t2=None,
        diff=None,
        text_embedding=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        del kwargs

        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Agent-level validation
        # ---------------------------------------------------------------------

        if analysis.task != "temporal_change":
            raise ValueError(
                "SatQueryAITemporalChangeV3Tool received a "
                "non-temporal request."
            )

        if t1 is None:
            raise ValueError(
                "Required temporal input missing: t1"
            )

        if t2 is None:
            raise ValueError(
                "Required temporal input missing: t2"
            )

        if diff is None:
            raise ValueError(
                "Required temporal input missing: diff"
            )

        if text_embedding is None:
            raise ValueError(
                "Required temporal input missing: text_embedding"
            )

        # ---------------------------------------------------------------------
        # Tensor preparation
        # ---------------------------------------------------------------------

        t1_tensor = _ensure_temporal_batch(
            t1,
            "t1",
        )

        t2_tensor = _ensure_temporal_batch(
            t2,
            "t2",
        )

        diff_tensor = _ensure_temporal_batch(
            diff,
            "diff",
        )

        question_tensor = _ensure_temporal_batch(
            text_embedding,
            "text_embedding",
        )

        # ---------------------------------------------------------------------
        # Model forward pass
        # ---------------------------------------------------------------------

        question_type = _detect_question_type(
            query
        )

        selected_head = QUESTION_TYPE_TO_HEAD[
            question_type
        ]

        with torch.no_grad():

            outputs = self.model(
                t1=t1_tensor,
                t2=t2_tensor,
                diff=diff_tensor,
                question=question_tensor,
            )

        # ---------------------------------------------------------------------
        # Verify model output interface
        # ---------------------------------------------------------------------

        if "answer_logits" not in outputs:
            raise RuntimeError(
                "Temporal model output missing answer_logits."
            )

        required_heads = {
            "binary",
            "change_ratio",
            "change_ratio_types",
            "land_cover",
        }

        available_heads = set(
            outputs["answer_logits"].keys()
        )

        missing_heads = (
            required_heads
            - available_heads
        )

        if missing_heads:
            raise RuntimeError(
                "Temporal model output missing answer heads: "
                f"{sorted(missing_heads)}"
            )

        if "query_attention" not in outputs:
            raise RuntimeError(
                "Temporal model output missing query_attention."
            )

        if "change_logits" not in outputs:
            raise RuntimeError(
                "Temporal model output missing change_logits."
            )

        # ---------------------------------------------------------------------
        # Select answer vocabulary
        # ---------------------------------------------------------------------

        if selected_head == "binary":

            vocabulary = BINARY_VOCAB

        elif selected_head == "change_ratio":

            vocabulary = CHANGE_RATIO_VOCAB

        elif selected_head == "change_ratio_types":

            vocabulary = (
                CHANGE_RATIO_TYPES_VOCAB
            )

        elif selected_head == "land_cover":

            vocabulary = LAND_COVER_VOCAB

        else:
            raise RuntimeError(
                f"Unsupported temporal head: {selected_head}"
            )

        logits = outputs[
            "answer_logits"
        ][selected_head]

        answer, answer_probability, probabilities = (
            _decode_head(
                logits,
                vocabulary,
            )
        )

        # ---------------------------------------------------------------------
        # Structured evidence
        # ---------------------------------------------------------------------

        query_attention = (
            outputs["query_attention"]
        )

        change_logits = (
            outputs["change_logits"]
        )

        evidence = {
            "task": "temporal_change",

            "answer": answer,

            "answer_probability": (
                answer_probability
            ),

            "question_type": question_type,

            "selected_head": selected_head,

            "answer_vocabulary": list(
                vocabulary
            ),

            "answer_probabilities": probabilities,

            # ---------------------------------------------------------------
            # Temporal evidence
            # ---------------------------------------------------------------

            "t1_evidence": _tensor_summary(
                t1_tensor
            ),

            "t2_evidence": _tensor_summary(
                t2_tensor
            ),

            "difference_evidence": _tensor_summary(
                diff_tensor
            ),

            "query_attention": _spatial_map_to_list(
                query_attention
            ),

            "change_map": _spatial_map_to_list(
                change_logits
            ),

            # ---------------------------------------------------------------
            # Specialist provenance
            # ---------------------------------------------------------------

            "specialist": (
                "TemporalChangeVQAv3OrdinalLoss"
            ),

            "architecture_source": (
                "TemporalChangeVQAv2"
            ),

            "checkpoint": str(
                self.checkpoint_path
            ),

            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),

            "parameter_count": (
                self.parameter_count
            ),

            "device": "cpu",

            # ---------------------------------------------------------------
            # Query / routing information
            # ---------------------------------------------------------------

            "query": query,

            "normalized_query": _normalize_query(
                query
            ),

            "routing_confidence": float(
                analysis.confidence
            ),
        }

        # ---------------------------------------------------------------------
        # Specialist execution trace
        # ---------------------------------------------------------------------

        execution_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        evidence["execution_trace"] = {
            "task": "temporal_change",

            "selected_specialist": (
                "TemporalChangeV3"
            ),

            "question_type": question_type,

            "selected_head": selected_head,

            "input_shapes": {
                "t1": list(
                    t1_tensor.shape
                ),
                "t2": list(
                    t2_tensor.shape
                ),
                "diff": list(
                    diff_tensor.shape
                ),
                "text_embedding": list(
                    question_tensor.shape
                ),
            },

            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),

            "parameter_count": (
                self.parameter_count
            ),

            "device": "cpu",

            "execution_time_ms": (
                execution_ms
            ),

            "status": "success",
        }

        return evidence


# =============================================================================
# SIMPLE DIRECT INITIALIZATION TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "SATQUERYAI — TEMPORAL V3 TOOL INITIALIZATION TEST"
    )
    print("=" * 80)

    tool = SatQueryAITemporalChangeV3Tool()

    print()
    print(
        "Checkpoint:",
        tool.checkpoint_path,
    )

    print(
        "Checkpoint epoch:",
        tool.checkpoint_epoch,
    )

    print(
        "Parameter count:",
        f"{tool.parameter_count:,}",
    )

    if tool.parameter_count != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Temporal V3 parameter count verification failed."
        )

    print()
    print(
        "Temporal V3 tool initialization: PASS"
    )