# =============================================================================
# SATQUERYAI — RESPONSE FORMATTER
# =============================================================================
#
# Converts the raw agent result into a compact, frontend/backend-friendly
# response without changing the agent core or specialist contracts.
#
# This layer:
#   - extracts the answer from specialist evidence
#   - preserves answer confidence when available
#   - preserves task/routing information
#   - preserves execution trace
#   - exposes useful evidence metadata
#   - avoids serializing huge internal tensors directly
#
# It does NOT:
#   - run models
#   - modify checkpoints
#   - change routing
#   - change specialist behavior
# =============================================================================

from math import prod
from typing import Any, Dict


# =============================================================================
# KEYS WHOSE VALUES ARE USEFUL TO EXPOSE DIRECTLY
# =============================================================================

SMALL_NUMERIC_KEYS = {
    "probabilities",
    "logits",
    "bbox",
    "point",
    "pred_box",
    "predicted_box",
    "cx",
    "cy",
    "std_x",
    "std_y",
    "peak_probability",
    "evidence_confidence",
    "spatial_entropy",
}


# =============================================================================
# LARGE INTERNAL EVIDENCE KEYS
# =============================================================================

LARGE_EVIDENCE_KEYS = {
    "s1_to_s2_attention",
    "s2_to_s1_attention",
    "cross_attention_s1_to_s2",
    "cross_attention_s2_to_s1",
    "query_attention",
    "spatial_query_attention",
    "fused_tokens",
    "s1_tokens",
    "s2_tokens",
    "native_output",
}


def _shape_of(value):
    """
    Return a JSON-safe shape when the object exposes one.
    """

    shape = getattr(value, "shape", None)

    if shape is None:
        return None

    try:
        return [int(x) for x in shape]
    except Exception:
        return None


def _numel_of(value):
    """
    Return the number of elements when available.
    """

    try:
        if hasattr(value, "numel"):
            return int(value.numel())

        shape = _shape_of(value)

        if shape is not None:
            return int(prod(shape))

    except Exception:
        pass

    return None


def _small_value(value):
    """
    Convert a small tensor/array/scalar to an ordinary Python value.
    """

    try:

        if hasattr(value, "detach"):
            value = (
                value.detach()
                .cpu()
                .tolist()
            )

        elif hasattr(value, "tolist"):
            value = value.tolist()

        elif hasattr(value, "item"):
            value = value.item()

    except Exception:

        return None

    return value


def _summarize_value(
    value,
    key=None,
):
    """
    Convert arbitrary specialist output into JSON-friendly data.

    Small prediction-related arrays are preserved as values.
    Large model internals are summarized by type/shape instead.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):

        result = {}

        for child_key, child_value in value.items():

            if child_key == "native_output":

                result[child_key] = {
                    "available": True,
                    "description": (
                        "Native specialist output retained "
                        "internally and omitted from the API payload."
                    ),
                }

                continue

            if child_key in LARGE_EVIDENCE_KEYS:

                shape = _shape_of(child_value)
                numel = _numel_of(child_value)

                result[child_key] = {
                    "type": type(child_value).__name__,
                    "shape": shape,
                    "num_elements": numel,
                }

                continue

            result[child_key] = _summarize_value(
                child_value,
                key=child_key,
            )

        return result

    if isinstance(
        value,
        (list, tuple),
    ):

        # Keep small lists completely.
        if len(value) <= 32:

            return [
                _summarize_value(item)
                for item in value
            ]

        return {
            "type": type(value).__name__,
            "length": len(value),
        }

    shape = _shape_of(value)
    numel = _numel_of(value)

    # Small numeric arrays/tensors.
    if (
        key in SMALL_NUMERIC_KEYS
        and numel is not None
        and numel <= 64
    ):

        converted = _small_value(value)

        if converted is not None:
            return converted

    # Generic scalar-like objects.
    if numel == 1:

        converted = _small_value(value)

        if converted is not None:
            return converted

    # Larger arrays/tensors.
    return {
        "type": type(value).__name__,
        "shape": shape,
        "num_elements": numel,
    }


# =============================================================================
# RESULT KIND
# =============================================================================

def _result_kind(
    task,
    evidence,
):
    """
    Identify the application-level result family.

    This deliberately uses task/routing metadata rather than exposing the
    underlying model class as the primary frontend concept.
    """

    if task == "grounding":
        return "grounding"

    if task == "temporal_change":
        return "temporal_change"

    if task == "cross_modal":
        route = None

        if isinstance(evidence, dict):
            route = evidence.get(
                "cross_modal_vqa_route"
            )

        if route == "mcq":
            return "cross_modal_mcq_vqa"

        return "cross_modal_vqa"

    if task == "single_image_vqa":

        route = None

        if isinstance(evidence, dict):
            route = evidence.get(
                "single_image_vqa_route"
            )

        if route == "mcq":
            return "single_image_mcq_vqa"

        return "single_image_binary_vqa"

    return task or "unknown"


# =============================================================================
# PUBLIC FORMATTER
# =============================================================================

def format_agent_response(
    agent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a raw SatQueryAI agent result into the stable application response.

    Expected raw structure:

        {
            "answer_status": ...,
            "analysis": ...,
            "evidence": ...,
            "execution_trace": ...
        }
    """

    if not isinstance(
        agent_result,
        dict,
    ):
        raise TypeError(
            "agent_result must be a dictionary."
        )

    status = agent_result.get(
        "answer_status"
    )

    analysis = agent_result.get(
        "analysis"
    )

    evidence = agent_result.get(
        "evidence"
    )

    trace = agent_result.get(
        "execution_trace"
    )

    if not isinstance(
        analysis,
        dict,
    ):
        analysis = {}

    if not isinstance(
        trace,
        dict,
    ):
        trace = {}

    if not isinstance(
        evidence,
        dict,
    ):
        evidence = {}

    # -------------------------------------------------------------------------
    # Answer extraction
    # -------------------------------------------------------------------------

    answer = evidence.get(
        "answer"
    )

    answer_confidence = evidence.get(
        "answer_confidence"
    )

    if answer_confidence is None:
        answer_confidence = evidence.get(
            "answer_probability"
        )

    # -------------------------------------------------------------------------
    # Cross-modal evidence is currently nested inside "evidence".
    # Grounding may already use a direct standardized evidence dictionary.
    # -------------------------------------------------------------------------

    nested_evidence = evidence.get(
        "evidence"
    )

    evidence_type = evidence.get(
        "evidence_type"
    )

    if (
        evidence_type is None
        and isinstance(
            nested_evidence,
            dict,
        )
    ):
        evidence_type = nested_evidence.get(
            "evidence_type"
        )

    # -------------------------------------------------------------------------
    # Result metadata
    # -------------------------------------------------------------------------

    specialist = evidence.get(
        "specialist"
    )

    specialist_selected = evidence.get(
        "specialist_selected"
    )

    route = evidence.get(
        "cross_modal_vqa_route"
    )

    if route is None:
        route = evidence.get(
            "single_image_vqa_route"
        )

    task = analysis.get(
        "task"
    )

    result_kind = _result_kind(
        task,
        evidence,
    )

    # -------------------------------------------------------------------------
    # Build stable API response
    # -------------------------------------------------------------------------

    formatted = {
        "status": status,

        "answer": answer,

        "answer_confidence": (
            _summarize_value(
                answer_confidence,
                key="answer_confidence",
            )
        ),

        "analysis": {
            "task": task,
            "grounding_mode": (
                analysis.get(
                    "grounding_mode"
                )
            ),
            "routing_confidence": (
                analysis.get(
                    "confidence"
                )
            ),
            "matched_signals": (
                analysis.get(
                    "matched_signals",
                    [],
                )
            ),
            "normalized_query": (
                analysis.get(
                    "normalized_query"
                )
            ),
        },

        "result": {
            "kind": result_kind,
            "route": route,
            "specialist": specialist,
            "specialist_selected": specialist_selected,
        },

        "evidence": {
            "available": bool(evidence),
            "type": evidence_type,
            "data": _summarize_value(
                evidence
            ),
        },

        "trace": {
            "status": trace.get(
                "status"
            ),
            "selected_tool": trace.get(
                "selected_tool"
            ),
            "detected_task": trace.get(
                "detected_task"
            ),
            "routing_confidence": trace.get(
                "routing_confidence"
            ),
            "matched_signals": trace.get(
                "matched_signals",
                [],
            ),
            "validation_ms": trace.get(
                "validation_ms"
            ),
            "execution_ms": trace.get(
                "execution_ms"
            ),
            "total_ms": trace.get(
                "total_ms"
            ),
        },
    }

    return formatted


# =============================================================================
# CONVENIENCE ALIAS
# =============================================================================

normalize_agent_response = format_agent_response