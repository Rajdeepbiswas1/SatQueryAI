from __future__ import annotations

from pathlib import Path

from backend.temporal_inference_service import (
    SatQueryAITemporalInferenceService,
)


PROJECT_ROOT = Path(r"D:\SatQueryAI")

T1_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_raw"
    / "im1"
    / "00003.png"
)

T2_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_raw"
    / "im2"
    / "00003.png"
)

QUERY = (
    "Did the regions of non-vegetated ground surface change?"
)

EXPECTED_ANSWER = "yes"


def main() -> None:

    print("=" * 80)
    print(
        "SATQUERYAI — REAL TEMPORAL SERVICE SMOKE TEST"
    )
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Validate real input files.
    # ------------------------------------------------------------------

    if not T1_PATH.is_file():
        raise FileNotFoundError(
            f"T1 image not found: {T1_PATH}"
        )

    if not T2_PATH.is_file():
        raise FileNotFoundError(
            f"T2 image not found: {T2_PATH}"
        )

    print("T1:", T1_PATH)
    print("T2:", T2_PATH)
    print("Input files: PASS")
    print()

    # ------------------------------------------------------------------
    # 2. Build the application service.
    # ------------------------------------------------------------------

    print(
        "Building temporal inference service..."
    )

    service = (
        SatQueryAITemporalInferenceService(
            device="cpu"
        )
    )

    print("Service construction: PASS")
    print()

    # ------------------------------------------------------------------
    # 3. Execute the complete raw-image pipeline.
    # ------------------------------------------------------------------

    print(
        "Executing real temporal service..."
    )
    print()

    result = service.execute(
        query=QUERY,
        paths=[
            T1_PATH,
            T2_PATH,
        ],
    )

    # ------------------------------------------------------------------
    # 4. Validate top-level application result.
    # ------------------------------------------------------------------

    if result.get("status") != "success":
        raise AssertionError(
            f"Unexpected service status: "
            f"{result.get('status')}"
        )

    agent_result = result.get(
        "agent_result"
    )

    if not isinstance(
        agent_result,
        dict,
    ):
        raise AssertionError(
            "agent_result is missing."
        )

    # ------------------------------------------------------------------
    # 5. Validate routing.
    # ------------------------------------------------------------------

    analysis = agent_result.get(
        "analysis",
        {},
    )

    if analysis.get("task") != "temporal_change":
        raise AssertionError(
            "Incorrect detected task: "
            f"{analysis.get('task')}"
        )

    # ------------------------------------------------------------------
    # 6. Validate execution.
    # ------------------------------------------------------------------

    execution_trace = agent_result.get(
        "execution_trace",
        {},
    )

    if execution_trace.get("status") != "success":
        raise AssertionError(
            "Temporal agent execution did not "
            "complete successfully."
        )

    if execution_trace.get(
        "selected_tool"
    ) != "TemporalChangeV3":
        raise AssertionError(
            "Unexpected temporal tool: "
            f"{execution_trace.get('selected_tool')}"
        )

    # ------------------------------------------------------------------
    # 7. Validate feature shapes.
    # ------------------------------------------------------------------

    feature_info = result.get(
        "feature_extraction",
        {},
    )

    expected_shape = [
        1,
        196,
        768,
    ]

    if feature_info.get(
        "t1_shape"
    ) != expected_shape:
        raise AssertionError(
            "Unexpected T1 feature shape: "
            f"{feature_info.get('t1_shape')}"
        )

    if feature_info.get(
        "t2_shape"
    ) != expected_shape:
        raise AssertionError(
            "Unexpected T2 feature shape: "
            f"{feature_info.get('t2_shape')}"
        )

    if feature_info.get(
        "diff_shape"
    ) != expected_shape:
        raise AssertionError(
            "Unexpected Diff feature shape: "
            f"{feature_info.get('diff_shape')}"
        )

    # ------------------------------------------------------------------
    # 8. Extract specialist answer.
    # ------------------------------------------------------------------

    evidence = agent_result.get(
        "evidence",
        {},
    )

    answer = None

    if isinstance(
        evidence,
        dict,
    ):
        answer = evidence.get(
            "answer"
        )

    # Some formatted responses may keep
    # the answer at the top level.
    if answer is None:
        answer = agent_result.get(
            "answer"
        )

    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print()

    print(
        "Service status       :",
        result.get("status"),
    )

    print(
        "Detected task        :",
        analysis.get("task"),
    )

    print(
        "Selected tool        :",
        execution_trace.get(
            "selected_tool"
        ),
    )

    print(
        "T1 feature shape     :",
        feature_info.get(
            "t1_shape"
        ),
    )

    print(
        "T2 feature shape     :",
        feature_info.get(
            "t2_shape"
        ),
    )

    print(
        "Diff feature shape   :",
        feature_info.get(
            "diff_shape"
        ),
    )

    print(
        "Model answer         :",
        answer,
    )

    print(
        "Expected answer      :",
        EXPECTED_ANSWER,
    )

    print()

    # This is the same real pair/question already
    # validated directly through Temporal V3.
    if str(answer).strip().lower() != EXPECTED_ANSWER:
        raise AssertionError(
            "Temporal service answer mismatch: "
            f"expected {EXPECTED_ANSWER!r}, "
            f"got {answer!r}"
        )

    print(
        "REAL TEMPORAL SERVICE INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()