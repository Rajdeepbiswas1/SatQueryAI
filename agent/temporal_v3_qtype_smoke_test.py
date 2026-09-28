# =============================================================================
# SATQUERYAI — TEMPORAL V3 QUESTION-TYPE SMOKE TEST
# =============================================================================
#
# Verifies all 8 temporal question types through the REAL Agent path:
#
#   Test manifest
#       ↓
#   real temporal feature pair
#       ↓
#   real question embedding
#       ↓
#   SatQueryAIAgent.execute()
#       ↓
#   TemporalChangeV3
#       ↓
#   question-type detection
#       ↓
#   correct answer head
#
# No training.
# No checkpoint modification.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "test_manifest.csv"
)


# =============================================================================
# EXPECTED QUESTION-TYPE → HEAD MAPPING
# =============================================================================

EXPECTED_HEADS = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio_types",
}

EXPECTED_QTYPES = [
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "smallest_change",
    "largest_change",
    "change_to_what",
    "change_ratio",
    "change_ratio_types",
]


# =============================================================================
# MANIFEST
# =============================================================================

def read_test_manifest() -> pd.DataFrame:

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Test manifest not found: {TEST_MANIFEST}"
        )

    df = pd.read_csv(
        TEST_MANIFEST
    )

    if df.empty:
        raise RuntimeError(
            "Temporal Test manifest is empty."
        )

    if "qtype" not in df.columns:
        if "question_type" in df.columns:
            df["qtype"] = df["question_type"]
        else:
            raise RuntimeError(
                "Test manifest has no qtype/question_type column."
            )

    required = {
        "pair_id",
        "question_id",
        "qtype",
        "question",
        "answer",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Test manifest missing columns: {missing}"
        )

    df["pair_id"] = (
        df["pair_id"]
        .astype(str)
        .str.strip()
    )

    # Numeric pair IDs must remain zero-padded.
    def normalize_pair_id(value):
        text = str(value).strip()
        if text.isdigit():
            return text.zfill(5)
        return text

    df["pair_id"] = (
        df["pair_id"].map(normalize_pair_id)
    )

    df["question_id"] = (
        df["question_id"].astype(int)
    )

    df["qtype"] = (
        df["qtype"]
        .astype(str)
        .str.strip()
    )

    df["question"] = (
        df["question"]
        .astype(str)
        .str.strip()
    )

    df["answer"] = (
        df["answer"]
        .astype(str)
        .str.strip()
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "SATQUERYAI — TEMPORAL V3 QUESTION-TYPE SMOKE TEST"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Imports
    # -------------------------------------------------------------------------

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    from agent.bootstrap import (
        create_runtime_agent,
    )

    from agent.temporal_change_vqa_tool import (
        _detect_question_type,
    )

    # Reuse the already-verified real-data shard helpers.
    from agent.temporal_v3_agent_smoke_test import (
        find_feature_pair,
        find_question_embedding,
    )

    # -------------------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------------------

    manifest = read_test_manifest()

    print()
    print(
        f"Test rows available: {len(manifest):,}"
    )

    available_qtypes = set(
        manifest["qtype"]
    )

    missing_qtypes = sorted(
        set(EXPECTED_QTYPES)
        - available_qtypes
    )

    if missing_qtypes:
        raise RuntimeError(
            "Test manifest missing question types: "
            f"{missing_qtypes}"
        )

    print(
        "All 8 temporal question types present: PASS"
    )

    # -------------------------------------------------------------------------
    # Build runtime ONCE
    # -------------------------------------------------------------------------

    print()
    print(
        "Building runtime agent..."
    )

    agent, grounding_tool, binary_vqa_tool = (
        create_runtime_agent()
    )

    del grounding_tool
    del binary_vqa_tool

    if not agent.registry.available(
        "temporal_change"
    ):
        raise RuntimeError(
            "Temporal capability is not registered."
        )

    temporal_entry = agent.registry.get(
        "temporal_change"
    )

    if temporal_entry["tool_name"] != (
        "TemporalChangeV3"
    ):
        raise RuntimeError(
            "Unexpected temporal tool: "
            f"{temporal_entry['tool_name']}"
        )

    print(
        "Temporal capability registration: PASS"
    )

    # -------------------------------------------------------------------------
    # Run one real row for each question type
    # -------------------------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print(
        "RUNNING 8 REAL TEMPORAL QUESTION-TYPE CASES"
    )
    print("=" * 80)

    total_start = time.perf_counter()

    for index, expected_qtype in enumerate(
        EXPECTED_QTYPES,
        start=1,
    ):

        subset = manifest[
            manifest["qtype"]
            == expected_qtype
        ]

        if subset.empty:
            raise RuntimeError(
                f"No test row available for {expected_qtype}"
            )

        row = subset.iloc[0]

        pair_id = str(
            row["pair_id"]
        )

        question_id = int(
            row["question_id"]
        )

        question = str(
            row["question"]
        )

        dataset_answer = str(
            row["answer"]
        )

        print()
        print("-" * 80)
        print(
            f"[{index}/8] {expected_qtype}"
        )
        print("-" * 80)

        print(
            "Question:",
            question,
        )

        print(
            "Dataset answer:",
            dataset_answer,
        )

        # ---------------------------------------------------------------------
        # Verify deterministic detector before model execution.
        # ---------------------------------------------------------------------

        detected_qtype = _detect_question_type(
            question
        )

        print(
            "Detected question type:",
            detected_qtype,
        )

        if detected_qtype != expected_qtype:
            raise RuntimeError(
                f"Question-type detection mismatch for "
                f"{expected_qtype}: detected {detected_qtype}\n"
                f"Question: {question}"
            )

        print(
            "Question-type detection: PASS"
        )

        # ---------------------------------------------------------------------
        # Load real features.
        # ---------------------------------------------------------------------

        t1, t2, diff, feature_path = (
            find_feature_pair(
                pair_id
            )
        )

        text_embedding, question_path = (
            find_question_embedding(
                question_id
            )
        )

        print(
            "Feature source:",
            feature_path.name,
        )

        print(
            "Question source:",
            question_path.name,
        )

        # ---------------------------------------------------------------------
        # Execute REAL Agent.
        # ---------------------------------------------------------------------

        start = time.perf_counter()

        result = agent.execute(
            query=question,
            t1=t1,
            t2=t2,
            diff=diff,
            text_embedding=text_embedding,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # ---------------------------------------------------------------------
        # Verify top-level result.
        # ---------------------------------------------------------------------

        if result.get(
            "answer_status"
        ) != "success":

            raise RuntimeError(
                f"Agent execution failed for {expected_qtype}:\n"
                f"{result}"
            )

        analysis = result[
            "analysis"
        ]

        evidence = result[
            "evidence"
        ]

        outer_trace = result[
            "execution_trace"
        ]

        if analysis["task"] != (
            "temporal_change"
        ):
            raise RuntimeError(
                f"Wrong task for {expected_qtype}: "
                f"{analysis['task']}"
            )

        # ---------------------------------------------------------------------
        # Verify question type and head.
        # ---------------------------------------------------------------------

        actual_qtype = evidence[
            "question_type"
        ]

        actual_head = evidence[
            "selected_head"
        ]

        expected_head = EXPECTED_HEADS[
            expected_qtype
        ]

        print(
            "Model-selected question type:",
            actual_qtype,
        )

        print(
            "Selected answer head:",
            actual_head,
        )

        print(
            "Expected answer head:",
            expected_head,
        )

        if actual_qtype != expected_qtype:
            raise RuntimeError(
                f"Evidence question type mismatch: "
                f"expected {expected_qtype}, "
                f"got {actual_qtype}"
            )

        if actual_head != expected_head:
            raise RuntimeError(
                f"Answer-head mismatch for {expected_qtype}: "
                f"expected {expected_head}, "
                f"got {actual_head}"
            )

        # ---------------------------------------------------------------------
        # Verify specialist identity and provenance.
        # ---------------------------------------------------------------------

        if evidence[
            "specialist"
        ] != "TemporalChangeVQAv3OrdinalLoss":

            raise RuntimeError(
                "Unexpected specialist: "
                f"{evidence['specialist']}"
            )

        if evidence[
            "checkpoint_epoch"
        ] != 10:

            raise RuntimeError(
                "Unexpected checkpoint epoch: "
                f"{evidence['checkpoint_epoch']}"
            )

        if evidence[
            "parameter_count"
        ] != 1_679_133:

            raise RuntimeError(
                "Unexpected parameter count: "
                f"{evidence['parameter_count']}"
            )

        # ---------------------------------------------------------------------
        # Verify evidence.
        # ---------------------------------------------------------------------

        for key in (
            "t1_evidence",
            "t2_evidence",
            "difference_evidence",
            "query_attention",
            "change_map",
            "execution_trace",
        ):

            if key not in evidence:
                raise RuntimeError(
                    f"Missing evidence key: {key}"
                )

        if len(
            evidence["query_attention"]
        ) != 14:

            raise RuntimeError(
                "query_attention is not 14x14."
            )

        if len(
            evidence["change_map"]
        ) != 14:

            raise RuntimeError(
                "change_map is not 14x14."
            )

        # ---------------------------------------------------------------------
        # Verify inner specialist trace.
        # ---------------------------------------------------------------------

        inner_trace = evidence[
            "execution_trace"
        ]

        if inner_trace[
            "status"
        ] != "success":

            raise RuntimeError(
                "Temporal specialist trace failed."
            )

        if inner_trace[
            "selected_specialist"
        ] != "TemporalChangeV3":

            raise RuntimeError(
                "Wrong selected specialist in inner trace."
            )

        # ---------------------------------------------------------------------
        # Verify outer Agent trace.
        # ---------------------------------------------------------------------

        if outer_trace[
            "selected_tool"
        ] != "TemporalChangeV3":

            raise RuntimeError(
                "Wrong selected tool in outer trace."
            )

        if outer_trace[
            "detected_task"
        ] != "temporal_change":

            raise RuntimeError(
                "Wrong detected task in outer trace."
            )

        if outer_trace[
            "status"
        ] != "success":

            raise RuntimeError(
                "Outer AgentExecutionTrace failed."
            )

        # ---------------------------------------------------------------------
        # Case passed.
        # ---------------------------------------------------------------------

        print(
            "Predicted answer:",
            evidence["answer"],
        )

        print(
            "Answer probability:",
            f"{evidence['answer_probability']:.6f}",
        )

        print(
            "Execution time:",
            f"{elapsed_ms:.2f} ms",
        )

        print(
            f"{expected_qtype}: PASS"
        )

        results.append(
            {
                "question_type": expected_qtype,
                "expected_head": expected_head,
                "selected_head": actual_head,
                "answer": evidence["answer"],
                "dataset_answer": dataset_answer,
                "answer_probability": evidence[
                    "answer_probability"
                ],
                "execution_ms": elapsed_ms,
            }
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    ) * 1000.0

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TEMPORAL V3 QUESTION-TYPE TEST SUMMARY"
    )
    print("=" * 80)

    for item in results:

        print(
            f"{item['question_type']:22s} "
            f"head={item['selected_head']:18s} "
            f"answer={item['answer']:12s} "
            f"dataset={item['dataset_answer']:12s} "
            f"PASS"
        )

    print()
    print(
        f"Cases passed: {len(results)}/8"
    )

    print(
        f"Total execution time: "
        f"{total_elapsed / 1000.0:.2f} s"
    )

    if len(results) != 8:
        raise RuntimeError(
            "Not all 8 temporal question types passed."
        )

    print()
    print(
        "ALL 8 TEMPORAL QUESTION TYPES: PASS"
    )

    print(
        "TEMPORAL V3 AGENT QUESTION-TYPE INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()