# =============================================================================
# SATQUERYAI — RESPONSE FORMATTER SMOKE TEST
# =============================================================================

from agent.response_formatter import format_agent_response


def check_case(name, raw_result, expected_kind):
    formatted = format_agent_response(raw_result)

    if formatted["status"] != "success":
        raise RuntimeError(
            f"{name}: unexpected status: {formatted['status']}"
        )

    if formatted["result"]["kind"] != expected_kind:
        raise RuntimeError(
            f"{name}: expected kind={expected_kind}, "
            f"got={formatted['result']['kind']}"
        )

    if formatted["answer"] is None:
        raise RuntimeError(
            f"{name}: answer was not preserved."
        )

    if formatted["trace"]["status"] != "success":
        raise RuntimeError(
            f"{name}: trace status was not preserved."
        )

    print(f"{name}: PASS")


def main():
    print("=" * 80)
    print("SATQUERYAI — RESPONSE FORMATTER SMOKE TEST")
    print("=" * 80)

    common_analysis = {
        "grounding_mode": None,
        "confidence": 1.0,
        "matched_signals": ["test"],
        "normalized_query": "test query",
    }

    # -------------------------------------------------------------------------
    # 1. Single-image Binary VQA
    # -------------------------------------------------------------------------
    check_case(
        "Single-image Binary VQA",
        {
            "answer_status": "success",
            "analysis": {
                **common_analysis,
                "task": "single_image_vqa",
            },
            "evidence": {
                "answer": "yes",
                "answer_confidence": 0.91,
                "specialist": "SingleModalityS1BinaryVQA",
                "single_image_vqa_route": "binary",
                "logits": [0.1, 2.4],
                "visual_question_attention": [0.2, 0.8],
            },
            "execution_trace": {
                "status": "success",
                "selected_tool": "SingleModalityVQATool",
                "detected_task": "single_image_vqa",
                "routing_confidence": 1.0,
                "matched_signals": ["question"],
                "validation_ms": 1.0,
                "execution_ms": 10.0,
                "total_ms": 11.0,
            },
        },
        "single_image_binary_vqa",
    )

    # -------------------------------------------------------------------------
    # 2. Single-image MCQ
    # -------------------------------------------------------------------------
    check_case(
        "Single-image MCQ",
        {
            "answer_status": "success",
            "analysis": {
                **common_analysis,
                "task": "single_image_vqa",
            },
            "evidence": {
                "answer": "d",
                "answer_probability": 0.66,
                "specialist": "SingleModalityS2MCQVQA",
                "single_image_vqa_route": "mcq",
                "probabilities": [0.1, 0.2, 0.03, 0.67],
            },
            "execution_trace": {
                "status": "success",
                "selected_tool": "SingleModalityVQATool",
                "detected_task": "single_image_vqa",
                "routing_confidence": 1.0,
                "matched_signals": ["options"],
                "validation_ms": 1.0,
                "execution_ms": 10.0,
                "total_ms": 11.0,
            },
        },
        "single_image_mcq_vqa",
    )

    # -------------------------------------------------------------------------
    # 3. Cross-modal Binary VQA
    # -------------------------------------------------------------------------
    check_case(
        "Cross-modal Binary VQA",
        {
            "answer_status": "success",
            "analysis": {
                **common_analysis,
                "task": "cross_modal",
            },
            "evidence": {
                "answer": "yes",
                "answer_confidence": 0.79,
                "specialist": "CrossModalBinaryVQA",
                "cross_modal_vqa_route": "binary",
                "s1_to_s2_attention": [[0.1]],
                "s2_to_s1_attention": [[0.2]],
            },
            "execution_trace": {
                "status": "success",
                "selected_tool": "CrossModalVQARouter",
                "detected_task": "cross_modal",
                "routing_confidence": 1.0,
                "matched_signals": ["s1", "s2"],
                "validation_ms": 1.0,
                "execution_ms": 10.0,
                "total_ms": 11.0,
            },
        },
        "cross_modal_vqa",
    )

    # -------------------------------------------------------------------------
    # 4. Cross-modal MCQ
    # -------------------------------------------------------------------------
    check_case(
        "Cross-modal MCQ",
        {
            "answer_status": "success",
            "analysis": {
                **common_analysis,
                "task": "cross_modal",
            },
            "evidence": {
                "answer": "b",
                "answer_probability": 0.73,
                "specialist": "CrossModalMCQVQA",
                "cross_modal_vqa_route": "mcq",
                "probabilities": [0.12, 0.73, 0.05, 0.10],
                "cross_attention_s1_to_s2": [[0.1]],
                "cross_attention_s2_to_s1": [[0.2]],
            },
            "execution_trace": {
                "status": "success",
                "selected_tool": "CrossModalVQARouter",
                "detected_task": "cross_modal",
                "routing_confidence": 1.0,
                "matched_signals": ["s1", "s2", "options"],
                "validation_ms": 1.0,
                "execution_ms": 10.0,
                "total_ms": 11.0,
            },
        },
        "cross_modal_mcq_vqa",
    )

    # -------------------------------------------------------------------------
    # 5. Grounding
    # -------------------------------------------------------------------------
    grounding_result = {
        "answer_status": "success",
        "analysis": {
            **common_analysis,
            "task": "grounding",
            "grounding_mode": "reference",
        },
        "evidence": {
            "answer": "grounded",
            "evidence_confidence": 0.84,
            "specialist": "C3QueryConditionedGrounding",
            "bbox": [0.1, 0.2, 0.8, 0.9],
            "peak_probability": 0.91,
            "native_output": {
                "large_tensor": True,
            },
        },
        "execution_trace": {
            "status": "success",
            "selected_tool": "GroundingModalityDispatcher",
            "detected_task": "grounding",
            "routing_confidence": 1.0,
            "matched_signals": ["grounding"],
            "validation_ms": 1.0,
            "execution_ms": 10.0,
            "total_ms": 11.0,
        },
    }

    formatted_grounding = format_agent_response(
        grounding_result
    )

    if formatted_grounding["result"]["kind"] != "grounding":
        raise RuntimeError(
            "Grounding: incorrect result kind."
        )

    if formatted_grounding["evidence"]["data"]["bbox"] != [
        0.1,
        0.2,
        0.8,
        0.9,
    ]:
        raise RuntimeError(
            "Grounding: bbox was not preserved."
        )

    native = formatted_grounding["evidence"]["data"]["native_output"]

    if not isinstance(native, dict):
        raise RuntimeError(
            "Grounding: native_output was not summarized."
        )

    print("Grounding: PASS")

    # -------------------------------------------------------------------------
    # 6. Temporal
    # -------------------------------------------------------------------------
    temporal_result = {
        "answer_status": "success",
        "analysis": {
            **common_analysis,
            "task": "temporal_change",
        },
        "evidence": {
            "answer": "yes",
            "answer_probability": 0.96,
            "question_type": "change_or_not",
            "selected_head": "binary",
            "specialist": "TemporalChangeVQAv3OrdinalLoss",
            "checkpoint_epoch": 10,
            "parameter_count": 1679133,
            "query_attention": [[0.1]],
            "change_map": [[0.2]],
        },
        "execution_trace": {
            "status": "success",
            "selected_tool": "TemporalChangeV3",
            "detected_task": "temporal_change",
            "routing_confidence": 1.0,
            "matched_signals": ["change"],
            "validation_ms": 1.0,
            "execution_ms": 10.0,
            "total_ms": 11.0,
        },
    }

    formatted_temporal = format_agent_response(
        temporal_result
    )

    if formatted_temporal["result"]["kind"] != "temporal_change":
        raise RuntimeError(
            "Temporal: incorrect result kind."
        )

    if formatted_temporal["answer"] != "yes":
        raise RuntimeError(
            "Temporal: answer was not preserved."
        )

    print("Temporal: PASS")

    print()
    print("=" * 80)
    print("RESPONSE FORMATTER SMOKE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()