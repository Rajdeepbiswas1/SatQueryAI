# =============================================================================
# SATQUERYAI — REAL-DATA CROSS-MODAL MCQ VQA SMOKE TEST
# =============================================================================
#
# Verifies:
#   S1 + S2 + question + 4 options
#       ->
#   Agent
#       ->
#   cross_modal
#       ->
#   CrossModalVQARouter
#       ->
#   CrossModalMCQVQA
#
# No training.
# No checkpoint modification.
# =============================================================================

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from agent.bootstrap import create_runtime_agent


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"D:\SatQueryAI")

MCQ_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "mcq_vqa_split.parquet"
)

MCQ_Q_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mcq_question_embeddings"
)

MCQ_OPTION_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings"
)

DOFA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

EXPECTED_SPECIALIST = "CrossModalMCQVQA"
EXPECTED_PARAMS = 2_104_065
EXPECTED_EPOCH = 9


# =============================================================================
# HELPERS
# =============================================================================

def require_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )
        
def normalize_text(value) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def find_dofa_features(patch_id: str):
    patch_id = str(patch_id)

    files = sorted(DOFA_DIR.glob("*.pt"))

    if not files:
        raise FileNotFoundError(
            f"No DOFA shards found under:\n{DOFA_DIR}"
        )

    for path in files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        patch_ids = [
            str(
                x.item()
                if torch.is_tensor(x)
                else x
            )
            for x in data["patch_ids"]
        ]

        for index, candidate in enumerate(
            patch_ids
        ):

            if candidate == patch_id:

                s1 = (
                    data["s1"][index]
                    .float()
                    .cpu()
                )

                s2 = (
                    data["s2"][index]
                    .float()
                    .cpu()
                )

                return s1, s2

    raise KeyError(
        f"Patch not found in DOFA cache: {patch_id}"
    )


def load_mcq_cache():

    qfiles = sorted(
        MCQ_Q_DIR.glob("shard_*.pt")
    )

    if not qfiles:
        qfiles = sorted(
            MCQ_Q_DIR.glob("*.pt")
        )

    ofiles = sorted(
        MCQ_OPTION_DIR.glob("shard_*.pt")
    )

    if not ofiles:
        ofiles = sorted(
            MCQ_OPTION_DIR.glob("*.pt")
        )

    if not qfiles:
        raise FileNotFoundError(
            "No MCQ question shards found."
        )

    if not ofiles:
        raise FileNotFoundError(
            "No MCQ option shards found."
        )

    # -------------------------------------------------------------------------
    # Question cache
    # -------------------------------------------------------------------------

    question_cache = {}
    question_by_text = {}

    for path in qfiles:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        indices = data["question_indices"]
        embeddings = data["question_embeddings"]

        questions = data.get("question")

        if questions is None:
            questions = data.get("questions")

        if questions is None:
            raise RuntimeError(
                f"MCQ question shard does not contain original question text:\n"
                f"{path}\n"
                f"Available keys: {sorted(data.keys())}"
            )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Invalid question shape in {path}: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"Invalid question embedding width in {path}: "
                f"{embeddings.shape[1]}"
            )

        if len(indices) != embeddings.shape[0]:
            raise RuntimeError(
                f"Question index/embedding mismatch in {path}"
            )

        if len(questions) != embeddings.shape[0]:
            raise RuntimeError(
                f"Question text/embedding mismatch in {path}"
            )

        for i, qindex in enumerate(indices):

            qindex = int(qindex)

            question_cache[qindex] = (
                embeddings[i]
                .float()
                .cpu()
            )

            normalized_question = normalize_text(
                questions[i]
            )

            if normalized_question in question_by_text:
                existing = question_by_text[
                    normalized_question
                ]

                if existing != qindex:
                    raise RuntimeError(
                        "Duplicate normalized MCQ question text maps "
                        f"to multiple indices: {existing}, {qindex}"
                    )

            question_by_text[
                normalized_question
            ] = qindex

    # -------------------------------------------------------------------------
    # Option cache
    # -------------------------------------------------------------------------

    option_cache = {}

    for path in ofiles:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        indices = data["question_indices"]

        option_names = [
            "option_a_embeddings",
            "option_b_embeddings",
            "option_c_embeddings",
            "option_d_embeddings",
        ]

        for i, qindex in enumerate(indices):

            qindex = int(qindex)

            option_cache[qindex] = torch.stack(
                [
                    data[name][i]
                    .float()
                    .cpu()
                    for name in option_names
                ],
                dim=0,
            )

    # -------------------------------------------------------------------------
    # Exact question/option index agreement
    # -------------------------------------------------------------------------

    if set(
        question_cache.keys()
    ) != set(
        option_cache.keys()
    ):
        raise RuntimeError(
            "Question/option cache index sets differ."
        )

    return (
        question_cache,
        option_cache,
        question_by_text,
    )


def parse_options(query: str):

    pattern = re.compile(
        r"(?:^|,\s*|\s+)"
        r"([abcd])\)"
        r"\s*"
        r"(.*?)"
        r"(?=,\s*[abcd]\)|$)",
        re.IGNORECASE,
    )

    matches = pattern.findall(
        str(query)
    )

    parsed = {}

    for label, value in matches:
        parsed[
            label.lower()
        ] = value.strip()

    if set(parsed) != {
        "a",
        "b",
        "c",
        "d",
    }:
        raise RuntimeError(
            "Could not parse exactly four MCQ options.\n"
            f"Query:\n{query}\n"
            f"Parsed:\n{parsed}"
        )

    return [
        parsed["a"],
        parsed["b"],
        parsed["c"],
        parsed["d"],
    ]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "SATQUERYAI — REAL-DATA CROSS-MODAL MCQ VQA SMOKE TEST"
    )
    print("=" * 80)

    print("\nDEVICE: CPU")
    print("TRAINING: NOT STARTED")
    print("CHECKPOINTS: FROZEN")

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------

    print("\nPATH CHECK")

    require_path(
        MCQ_PARQUET,
        "MCQ validation data",
    )

    require_path(
        MCQ_Q_DIR,
        "MCQ question cache",
    )

    require_path(
        MCQ_OPTION_DIR,
        "MCQ option cache",
    )

    require_path(
        DOFA_DIR,
        "DOFA cache",
    )

    print("  MCQ validation: PASS")
    print("  MCQ question cache: PASS")
    print("  MCQ option cache: PASS")
    print("  DOFA cache: PASS")

    # -------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BUILDING RUNTIME AGENT")
    print("=" * 80)

    agent, _, _ = create_runtime_agent()

    capabilities = agent.registry.capabilities()

    print(
        "Registered capabilities:",
        capabilities,
    )

    if "cross_modal" not in capabilities:
        raise RuntimeError(
            "cross_modal capability is not registered."
        )

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    mcq = pd.read_parquet(
        MCQ_PARQUET
    )

    val = mcq[
        mcq["model_split"] == "val"
    ].copy()

    if val.empty:
        raise RuntimeError(
            "MCQ validation split is empty."
        )

    row = val.iloc[0]

    patch_id = str(
        row["patch_id"]
    )

    query = str(
        row["input"]
    )

    ground_truth = (
        str(row["output"])
        .strip()
        .lower()
    )

    if ground_truth not in {
        "a",
        "b",
        "c",
        "d",
    }:
        raise RuntimeError(
            f"Unexpected MCQ ground truth: {ground_truth}"
        )
        
    (
        question_cache,
        option_cache,
        question_by_text,
    ) = load_mcq_cache()

    normalized_query = normalize_text(
        query
    )

    if normalized_query not in question_by_text:
        raise RuntimeError(
            "Could not map MCQ annotation question text "
            "to the question-embedding cache.\n"
            f"Question:\n{query}"
        )

    qindex = int(
        question_by_text[
            normalized_query
        ]
    )

    print("\nVALIDATION SAMPLE")

    print(
        "  ID:",
        row["ID"],
    )

    print(
        "  Patch ID:",
        patch_id,
    )

    print(
        "  Question index:",
        qindex,
    )
    
    print(
        "  Question text mapping: PASS"
    )

    print(
        "  Question:",
        query,
    )

    print(
        "  Ground truth:",
        ground_truth,
    )

    # -------------------------------------------------------------------------
    # CACHE
    # -------------------------------------------------------------------------

    if qindex not in question_cache:
        raise RuntimeError(
            f"Question index {qindex} is missing."
        )

    if qindex not in option_cache:
        raise RuntimeError(
            f"Option index {qindex} is missing."
        )

    question = (
        question_cache[qindex]
        .unsqueeze(0)
    )

    options = (
        option_cache[qindex]
        .unsqueeze(0)
    )

    s1, s2 = find_dofa_features(
        patch_id
    )

    s1 = s1.unsqueeze(0)
    s2 = s2.unsqueeze(0)

    option_text = parse_options(
        query
    )

    # -------------------------------------------------------------------------
    # SHAPES
    # -------------------------------------------------------------------------

    print("\nINPUT SHAPES")

    print(
        "  S1:",
        tuple(s1.shape),
    )

    print(
        "  S2:",
        tuple(s2.shape),
    )

    print(
        "  Text:",
        tuple(question.shape),
    )

    print(
        "  Options:",
        tuple(options.shape),
    )

    if tuple(s1.shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "Invalid S1 shape."
        )

    if tuple(s2.shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "Invalid S2 shape."
        )

    if tuple(question.shape) != (
        1,
        768,
    ):
        raise RuntimeError(
            "Invalid question shape."
        )

    if tuple(options.shape) != (
        1,
        4,
        768,
    ):
        raise RuntimeError(
            "Invalid option shape."
        )

    # -------------------------------------------------------------------------
    # AGENT EXECUTION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "REAL CROSS-MODAL MCQ VQA — AGENT EXECUTION"
    )
    print("=" * 80)

    print(
        "Options:",
        option_text,
    )

    start = time.perf_counter()

    result = agent.execute(
        query=query,
        input_modalities=[
            "s1",
            "s2",
        ],
        s1=s1,
        s2=s2,
        text_embedding=question,
        options=options,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    # -------------------------------------------------------------------------
    # OUTER RESULT
    # -------------------------------------------------------------------------

    if result.get(
        "answer_status"
    ) != "success":
        raise RuntimeError(
            "Agent execution failed:\n"
            f"{result}"
        )

    analysis = result[
        "analysis"
    ]

    evidence = result[
        "evidence"
    ]

    trace = result[
        "execution_trace"
    ]

    if analysis[
        "task"
    ] != "cross_modal":
        raise RuntimeError(
            "Expected cross_modal analysis task, got "
            f"{analysis['task']}"
        )

    if evidence.get(
        "specialist"
    ) != EXPECTED_SPECIALIST:
        raise RuntimeError(
            "Unexpected cross-modal MCQ specialist:\n"
            f"expected={EXPECTED_SPECIALIST}\n"
            f"actual={evidence.get('specialist')}"
        )

    if evidence.get(
        "parameter_count"
    ) != EXPECTED_PARAMS:
        raise RuntimeError(
            "Unexpected parameter count:\n"
            f"expected={EXPECTED_PARAMS}\n"
            f"actual={evidence.get('parameter_count')}"
        )

    if int(
        evidence.get(
            "checkpoint_epoch"
        )
    ) != EXPECTED_EPOCH:
        raise RuntimeError(
            "Unexpected checkpoint epoch:\n"
            f"expected={EXPECTED_EPOCH}\n"
            f"actual={evidence.get('checkpoint_epoch')}"
        )

    if trace.get(
        "status"
    ) != "success":
        raise RuntimeError(
            "Outer execution trace is not successful."
        )

    answer = str(
        evidence.get(
            "answer"
        )
    ).strip().lower()

    if answer not in {
        "a",
        "b",
        "c",
        "d",
    }:
        raise RuntimeError(
            f"Invalid MCQ answer returned: {answer}"
        )

    correct = (
        answer == ground_truth
    )

    print("\nRESULT")

    print(
        "  answer:",
        answer,
    )

    print(
        "  ground_truth:",
        ground_truth,
    )

    print(
        "  correct_for_sample:",
        correct,
    )

    print(
        "  specialist:",
        evidence.get(
            "specialist"
        ),
    )

    print(
        "  checkpoint_epoch:",
        evidence.get(
            "checkpoint_epoch"
        ),
    )

    print(
        "  parameter_count:",
        evidence.get(
            "parameter_count"
        ),
    )

    print(
        "  cross_modal_vqa_route:",
        evidence.get(
            "cross_modal_vqa_route"
        ),
    )

    print(
        "  specialist_selected:",
        evidence.get(
            "specialist_selected"
        ),
    )

    print("\nEXECUTION TRACE")

    print(
        "  selected_tool:",
        trace.get(
            "selected_tool"
        ),
    )

    print(
        "  detected_task:",
        trace.get(
            "detected_task"
        ),
    )

    print(
        "  status:",
        trace.get(
            "status"
        ),
    )

    print(
        "  routing_confidence:",
        trace.get(
            "routing_confidence"
        ),
    )

    print(
        "  execution_ms:",
        trace.get(
            "execution_ms"
        ),
    )

    print(
        "  total_ms:",
        trace.get(
            "total_ms"
        ),
    )

    print(
        f"\nExternal wall-clock execution time: "
        f"{elapsed:.2f} ms"
    )

    print(
        "\nREAL CROSS-MODAL MCQ VQA: PASS"
    )


if __name__ == "__main__":
    main()