# =============================================================================
# SATQUERYAI — TEMPORAL V3 AGENT SMOKE TEST
# =============================================================================
#
# Real-data integration test.
#
# Verifies:
#   1. Test manifest can be read.
#   2. A real temporal feature pair can be located.
#   3. The matching real question embedding can be located by question_id.
#   4. SatQueryAIAgent.execute() routes to temporal_change.
#   5. TemporalChangeV3 executes.
#   6. Structured temporal evidence is returned.
#   7. Specialist execution trace is present.
#   8. Outer AgentExecutionTrace is present.
#
# No model training.
# No checkpoint modification.
# No full-cache RAM loading.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys
import time

import pandas as pd
import torch


# =============================================================================
# PATHS
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

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

QUESTION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_test_question_embeddings"
)


# =============================================================================
# HELPERS
# =============================================================================

def normalize_pair_id(
    value: Any,
) -> str:

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8"
        )

    text = str(value).strip()

    if text.isdigit():
        return text.zfill(5)

    return text


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
            "Temporal test manifest is empty."
        )

    # Normalize common column-name variants.
    if "qtype" not in df.columns:
        for candidate in (
            "question_type",
            "type",
        ):
            if candidate in df.columns:
                df["qtype"] = df[candidate]
                break

    if "question_id" not in df.columns:
        for candidate in (
            "question_index",
            "question_embedding_index",
            "qid",
        ):
            if candidate in df.columns:
                df["question_id"] = df[candidate]
                break

    if "pair_id" not in df.columns:
        for candidate in (
            "pair",
            "pairID",
            "image_pair_id",
            "image_id",
        ):
            if candidate in df.columns:
                df["pair_id"] = df[candidate]
                break

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
            "Test manifest is missing columns: "
            f"{missing}"
        )

    df["pair_id"] = (
        df["pair_id"].map(
            normalize_pair_id
        )
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


def find_feature_pair(
    pair_id: str,
):
    """
    Load only the feature shard containing pair_id.
    """

    files = sorted(
        FEATURE_ROOT.glob(
            "shard_*.pt"
        )
    )

    if not files:
        files = sorted(
            FEATURE_ROOT.glob(
                "*.pt"
            )
        )

    if not files:
        raise FileNotFoundError(
            f"No temporal feature shards found under {FEATURE_ROOT}"
        )

    for path in files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids = data.get(
            "pair_ids"
        )

        t1 = data.get("t1")
        if t1 is None:
            t1 = data.get("t1_features")

        t2 = data.get("t2")
        if t2 is None:
            t2 = data.get("t2_features")

        diff = data.get("diff")
        if diff is None:
            diff = data.get("diff_features")

        if (
            pair_ids is None
            or t1 is None
            or t2 is None
            or diff is None
        ):
            raise RuntimeError(
                f"{path}: expected pair_ids plus temporal feature tensors "
                f"(t1/t2/diff or t1_features/t2_features/diff_features). "
                f"Found: {sorted(data.keys())}"
            )

        normalized_ids = [
            normalize_pair_id(x)
            for x in pair_ids
        ]

        if pair_id not in normalized_ids:
            del data
            continue

        index = normalized_ids.index(
            pair_id
        )

        return (
            t1[index],
            t2[index],
            diff[index],
            path,
        )

    raise KeyError(
        f"Temporal feature pair not found: {pair_id}"
    )


def find_question_embedding(
    question_id: int,
):
    """
    Load only the question-embedding shard containing question_id.
    """

    files = sorted(
        QUESTION_ROOT.glob(
            "shard_*.pt"
        )
    )

    if not files:
        files = sorted(
            QUESTION_ROOT.glob(
                "*.pt"
            )
        )

    if not files:
        raise FileNotFoundError(
            f"No question embedding shards found under {QUESTION_ROOT}"
        )

    for path in files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        question_ids = data.get(
            "question_ids"
        )

        if question_ids is None:
            question_ids = data.get(
                "ids"
            )

        embeddings = data.get(
            "question_embeddings"
        )

        if embeddings is None:
            embeddings = data.get(
                "embeddings"
            )

        if (
            question_ids is None
            or embeddings is None
        ):
            raise RuntimeError(
                f"{path}: expected question IDs and embeddings."
            )

        ids = [
            int(x)
            for x in question_ids
        ]

        if question_id not in ids:
            del data
            continue

        index = ids.index(
            question_id
        )

        return (
            embeddings[index],
            path,
        )

    raise KeyError(
        "Question embedding not found for "
        f"question_id={question_id}"
    )


def print_trace(
    label: str,
    trace: dict,
) -> None:

    print()
    print(label)

    for key, value in trace.items():
        print(
            f"  {key}: {value}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "SATQUERYAI — TEMPORAL V3 REAL AGENT SMOKE TEST"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Import runtime
    # -------------------------------------------------------------------------

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    from agent.bootstrap import (
        create_runtime_agent,
    )

    # -------------------------------------------------------------------------
    # 2. Read test manifest
    # -------------------------------------------------------------------------

    manifest = read_test_manifest()

    print()
    print(
        f"Test rows available: {len(manifest):,}"
    )

    # Prefer a general change question so the first smoke test is unambiguous.
    preferred = manifest[
        manifest["qtype"]
        == "change_or_not"
    ]

    if preferred.empty:
        row = manifest.iloc[0]

    else:
        row = preferred.iloc[0]

    pair_id = normalize_pair_id(
        row["pair_id"]
    )

    question_id = int(
        row["question_id"]
    )

    qtype = str(
        row["qtype"]
    )

    question = str(
        row["question"]
    )

    expected_answer = str(
        row["answer"]
    )

    print()
    print("SELECTED REAL TEST ROW")
    print(
        "  pair_id:",
        pair_id,
    )
    print(
        "  question_id:",
        question_id,
    )
    print(
        "  question_type:",
        qtype,
    )
    print(
        "  question:",
        question,
    )
    print(
        "  dataset_answer:",
        expected_answer,
    )

    # -------------------------------------------------------------------------
    # 3. Load one real feature pair
    # -------------------------------------------------------------------------

    t1, t2, diff, feature_path = (
        find_feature_pair(
            pair_id
        )
    )

    print()
    print("TEMPORAL FEATURES")
    print(
        "  source:",
        feature_path,
    )
    print(
        "  T1:",
        tuple(t1.shape),
    )
    print(
        "  T2:",
        tuple(t2.shape),
    )
    print(
        "  Diff:",
        tuple(diff.shape),
    )
    print(
        "  dtype:",
        t1.dtype,
    )

    if tuple(t1.shape) != (
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected T1 shape: {tuple(t1.shape)}"
        )

    if tuple(t2.shape) != (
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected T2 shape: {tuple(t2.shape)}"
        )

    if tuple(diff.shape) != (
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected Diff shape: {tuple(diff.shape)}"
        )

    # -------------------------------------------------------------------------
    # 4. Load exact matching question embedding
    # -------------------------------------------------------------------------

    text_embedding, question_path = (
        find_question_embedding(
            question_id
        )
    )

    print()
    print("QUESTION EMBEDDING")
    print(
        "  source:",
        question_path,
    )
    print(
        "  shape:",
        tuple(text_embedding.shape),
    )
    print(
        "  dtype:",
        text_embedding.dtype,
    )

    if tuple(text_embedding.shape) != (
        768,
    ):
        raise RuntimeError(
            "Unexpected question embedding shape: "
            f"{tuple(text_embedding.shape)}"
        )

    # -------------------------------------------------------------------------
    # 5. Finite-value checks
    # -------------------------------------------------------------------------

    for name, tensor in (
        ("T1", t1),
        ("T2", t2),
        ("Diff", diff),
        ("Text", text_embedding),
    ):

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains non-finite values."
            )

    print()
    print(
        "REAL-DATA INPUT VALIDATION: PASS"
    )

    # -------------------------------------------------------------------------
    # 6. Build complete runtime
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
            "Temporal change capability is not registered."
        )

    print(
        "Temporal capability registration: PASS"
    )

    temporal_entry = agent.registry.get(
        "temporal_change"
    )

    if temporal_entry is None:
        raise RuntimeError(
            "Temporal registry entry missing."
        )

    print(
        "Registered temporal tool:",
        temporal_entry["tool_name"],
    )

    if temporal_entry["tool_name"] != (
        "TemporalChangeV3"
    ):
        raise RuntimeError(
            "Unexpected temporal tool name: "
            f"{temporal_entry['tool_name']}"
        )

    # -------------------------------------------------------------------------
    # 7. REAL agent.execute()
    # -------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("REAL AGENT EXECUTION")
    print("-" * 80)

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

    # -------------------------------------------------------------------------
    # 8. Top-level result verification
    # -------------------------------------------------------------------------

    if result.get(
        "answer_status"
    ) != "success":

        raise RuntimeError(
            "Temporal Agent execution failed:\n"
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
            "Agent routed to the wrong task: "
            f"{analysis['task']}"
        )

    if evidence is None:
        raise RuntimeError(
            "Temporal evidence is missing."
        )

    # -------------------------------------------------------------------------
    # 9. Evidence verification
    # -------------------------------------------------------------------------

    required_evidence = {
        "task",
        "answer",
        "answer_probability",
        "question_type",
        "selected_head",
        "t1_evidence",
        "t2_evidence",
        "difference_evidence",
        "query_attention",
        "change_map",
        "specialist",
        "checkpoint_epoch",
        "parameter_count",
        "execution_trace",
    }

    missing_evidence = sorted(
        required_evidence
        - set(evidence.keys())
    )

    if missing_evidence:
        raise RuntimeError(
            "Temporal evidence missing keys: "
            f"{missing_evidence}"
        )

    if evidence[
        "parameter_count"
    ] != 1_679_133:
        raise RuntimeError(
            "Unexpected temporal parameter count: "
            f"{evidence['parameter_count']}"
        )

    if evidence[
        "checkpoint_epoch"
    ] != 10:
        raise RuntimeError(
            "Unexpected temporal checkpoint epoch: "
            f"{evidence['checkpoint_epoch']}"
        )

    query_attention = evidence[
        "query_attention"
    ]

    change_map = evidence[
        "change_map"
    ]

    if len(query_attention) != 14:
        raise RuntimeError(
            "Query attention is not 14x14."
        )

    if len(query_attention[0]) != 14:
        raise RuntimeError(
            "Query attention is not 14x14."
        )

    if len(change_map) != 14:
        raise RuntimeError(
            "Change map is not 14x14."
        )

    if len(change_map[0]) != 14:
        raise RuntimeError(
            "Change map is not 14x14."
        )

    print()
    print(
        "STANDARDIZED TEMPORAL EVIDENCE: PASS"
    )

    # -------------------------------------------------------------------------
    # 10. Specialist trace verification
    # -------------------------------------------------------------------------

    inner_trace = evidence[
        "execution_trace"
    ]

    required_inner_trace = {
        "task",
        "selected_specialist",
        "question_type",
        "selected_head",
        "input_shapes",
        "checkpoint_epoch",
        "parameter_count",
        "device",
        "execution_time_ms",
        "status",
    }

    missing_inner_trace = sorted(
        required_inner_trace
        - set(inner_trace.keys())
    )

    if missing_inner_trace:
        raise RuntimeError(
            "Temporal specialist trace missing keys: "
            f"{missing_inner_trace}"
        )

    if inner_trace[
        "status"
    ] != "success":
        raise RuntimeError(
            "Temporal specialist trace is not successful."
        )

    if inner_trace[
        "selected_specialist"
    ] != "TemporalChangeV3":
        raise RuntimeError(
            "Unexpected temporal specialist name: "
            f"{inner_trace['selected_specialist']}"
        )

    # -------------------------------------------------------------------------
    # 11. Outer agent trace verification
    # -------------------------------------------------------------------------

    if outer_trace[
        "detected_task"
    ] != "temporal_change":
        raise RuntimeError(
            "Outer execution trace has wrong task."
        )

    if outer_trace[
        "selected_tool"
    ] != "TemporalChangeV3":
        raise RuntimeError(
            "Outer execution trace selected wrong tool: "
            f"{outer_trace['selected_tool']}"
        )

    if outer_trace[
        "status"
    ] != "success":
        raise RuntimeError(
            "Outer AgentExecutionTrace is not successful."
        )

    print()
    print(
        "SPECIALIST EXECUTION TRACE: PASS"
    )

    print()
    print(
        "OUTER AGENT EXECUTION TRACE: PASS"
    )

    # -------------------------------------------------------------------------
    # 12. Report actual execution
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TEMPORAL V3 AGENT RESULT"
    )
    print("=" * 80)

    print(
        "Answer:",
        evidence["answer"],
    )

    print(
        "Answer probability:",
        f"{evidence['answer_probability']:.6f}",
    )

    print(
        "Question type:",
        evidence["question_type"],
    )

    print(
        "Selected head:",
        evidence["selected_head"],
    )

    print(
        "Specialist:",
        evidence["specialist"],
    )

    print(
        "Dataset answer:",
        expected_answer,
    )

    print(
        "End-to-end elapsed:",
        f"{elapsed_ms:.2f} ms",
    )

    print_trace(
        "TEMPORAL SPECIALIST TRACE",
        inner_trace,
    )

    print_trace(
        "OUTER AGENT TRACE",
        outer_trace,
    )

    print()
    print("=" * 80)
    print(
        "TEMPORAL V3 REAL AGENT INTEGRATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()