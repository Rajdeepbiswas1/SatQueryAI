# =============================================================================
# SATQUERYAI — REAL-DATA SINGLE-MODALITY VQA SMOKE TEST
# =============================================================================
#
# Purpose:
#   Verify the real one-modality VQA paths:
#
#       S1 + question -> SingleModalityS1BinaryVQA / MCQVQA
#       S2 + question -> SingleModalityS2BinaryVQA / MCQVQA
#
# Covers:
#   - S1 Binary
#   - S2 Binary
#   - S1 MCQ
#   - S2 MCQ
#
# No training.
# No checkpoint modification.
# No model modification.
#
# Real cached validation data only.
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

PROJECT_ROOT = Path(r"D:\SatQueryAI")

BINARY_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

MCQ_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mcq_vqa_split.parquet"
)

DOFA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

BINARY_Q_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

MCQ_Q_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mcq_question_embeddings"
)

MCQ_OPTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings"
)


# =============================================================================
# EXPECTED SPECIALISTS
# =============================================================================

EXPECTED_BINARY = {
    "s1": "SingleModalityS1BinaryVQA",
    "s2": "SingleModalityS2BinaryVQA",
}

EXPECTED_MCQ = {
    "s1": "SingleModalityS1MCQVQA",
    "s2": "SingleModalityS2MCQVQA",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_path(path: Path, label: str) -> None:
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


def to_float_tensor(value) -> torch.Tensor:
    """
    Convert a cached / dataframe value into float32 tensor.
    """
    if torch.is_tensor(value):
        return value.detach().float().cpu()

    if isinstance(value, np.ndarray):
        return torch.from_numpy(
            np.array(
                value,
                copy=True,
            )
        ).float()

    if isinstance(value, (list, tuple)):
        return torch.tensor(
            value,
            dtype=torch.float32,
        )

    numbers = re.findall(
        r"[-+]?(?:\d*\.\d+|\d+\.?)",
        str(value),
    )

    if not numbers:
        raise ValueError(
            f"Could not parse numeric value:\n{value}"
        )

    return torch.tensor(
        [float(x) for x in numbers],
        dtype=torch.float32,
    )


# =============================================================================
# DOFA CACHE
# =============================================================================

def find_dofa_features(patch_id: str):
    """
    Locate one patch in the verified DOFA shards.
    """
    patch_id = str(patch_id)

    files = sorted(
        DOFA_DIR.glob("*.pt")
    )

    if not files:
        raise FileNotFoundError(
            f"No DOFA shards found in:\n{DOFA_DIR}"
        )

    print(
        f"Searching DOFA cache for patch:\n  {patch_id}"
    )

    for shard_index, path in enumerate(
        files,
        start=1,
    ):
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

        for row_index, candidate in enumerate(
            patch_ids
        ):
            if candidate == patch_id:

                print(
                    f"DOFA match: shard "
                    f"{shard_index}/{len(files)}"
                )

                s1 = (
                    data["s1"][row_index]
                    .float()
                    .cpu()
                )

                s2 = (
                    data["s2"][row_index]
                    .float()
                    .cpu()
                )

                return s1, s2

    raise KeyError(
        f"Patch not found in DOFA cache:\n{patch_id}"
    )


# =============================================================================
# BINARY QUESTION CACHE
# =============================================================================

def load_binary_question_lookup():
    """
    Build:
        annotation ID -> [768] question embedding

    The verified binary VQA training code uses the annotation ID directly
    against the cached binary question IDs.
    """
    files = sorted(
        BINARY_Q_DIR.glob("shard_*.pt")
    )

    if not files:
        files = sorted(
            BINARY_Q_DIR.glob("*.pt")
        )

    if not files:
        raise FileNotFoundError(
            f"No binary question cache shards found:\n{BINARY_Q_DIR}"
        )

    lookup = {}

    print("\nBINARY QUESTION CACHE")

    for shard_index, path in enumerate(
        files,
        start=1,
    ):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        ids = data.get(
            "ids"
        )

        embeddings = data.get(
            "question_embeddings"
        )

        if ids is None:
            ids = data.get(
                "question_ids"
            )

        if embeddings is None:
            embeddings = data.get(
                "embeddings"
            )

        if ids is None or embeddings is None:
            raise RuntimeError(
                f"Binary question shard has unexpected keys:\n"
                f"{path}\n"
                f"{sorted(data.keys())}"
            )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Unexpected binary question shape: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"Binary question embedding width must be 768, "
                f"got {embeddings.shape[1]}"
            )

        if len(ids) != embeddings.shape[0]:
            raise RuntimeError(
                "Binary question ID/embedding length mismatch."
            )

        for index, qid in enumerate(ids):
            lookup[str(qid)] = (
                embeddings[index]
                .float()
                .cpu()
            )

        print(
            f"  shard {shard_index}/{len(files)}: "
            f"{len(ids)} rows"
        )

    print(
        "  total question embeddings:",
        len(lookup),
    )

    return lookup


# =============================================================================
# MCQ QUESTION + OPTION CACHE
# =============================================================================

def load_mcq_question_option_lookup():
    """
    Build:
        normalized question text
            ->
        {
            question_embedding_index,
            question_embedding,
            option_embeddings [4,768]
        }

    The MCQ cache is handled by the verified question-index mapping rather
    than inventing a numerical relationship from annotation ID.
    """
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
            f"No MCQ question shards found:\n{MCQ_Q_DIR}"
        )

    if not ofiles:
        raise FileNotFoundError(
            f"No MCQ option shards found:\n{MCQ_OPTION_DIR}"
        )

    # -------------------------------------------------------------------------
    # Question cache
    # -------------------------------------------------------------------------

    question_by_index = {}
    question_by_text = {}

    print("\nMCQ QUESTION CACHE")

    for shard_index, path in enumerate(
        qfiles,
        start=1,
    ):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        indices = data.get(
            "question_indices"
        )

        embeddings = data.get(
            "question_embeddings"
        )

        questions = data.get(
            "question"
        )

        if questions is None:
            questions = data.get(
                "questions"
            )

        if indices is None or embeddings is None:
            raise RuntimeError(
                f"MCQ question shard has unexpected keys:\n"
                f"{path}\n"
                f"{sorted(data.keys())}"
            )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Unexpected MCQ question shape: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"MCQ question embedding width must be 768, "
                f"got {embeddings.shape[1]}"
            )

        if len(indices) != embeddings.shape[0]:
            raise RuntimeError(
                "MCQ question index/embedding length mismatch."
            )

        for local_index, qindex in enumerate(
            indices
        ):
            qindex = int(
                qindex
            )

            embedding = (
                embeddings[local_index]
                .float()
                .cpu()
            )

            question_by_index[qindex] = {
                "embedding": embedding,
            }

            if questions is not None:
                question_by_index[qindex][
                    "question"
                ] = str(
                    questions[local_index]
                )

                question_by_text[
                    normalize_text(
                        questions[local_index]
                    )
                ] = qindex

        print(
            f"  question shard "
            f"{shard_index}/{len(qfiles)}: "
            f"{len(indices)} rows"
        )

    # -------------------------------------------------------------------------
    # Option cache
    # -------------------------------------------------------------------------

    print("\nMCQ OPTION CACHE")

    option_by_index = {}

    for shard_index, path in enumerate(
        ofiles,
        start=1,
    ):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        indices = data.get(
            "question_indices"
        )

        if indices is None:
            indices = data.get(
                "question_ids"
            )

        if indices is None:
            raise RuntimeError(
                f"MCQ option shard has no question index field:\n"
                f"{path}\n"
                f"{sorted(data.keys())}"
            )

        option_names = [
            "option_a_embeddings",
            "option_b_embeddings",
            "option_c_embeddings",
            "option_d_embeddings",
        ]

        missing = [
            name
            for name in option_names
            if name not in data
        ]

        if missing:
            raise RuntimeError(
                "MCQ option shard is missing fields:\n"
                f"{missing}\n"
                f"Available keys:\n{sorted(data.keys())}"
            )

        for local_index, qindex in enumerate(
            indices
        ):
            qindex = int(
                qindex
            )

            options = torch.stack(
                [
                    data[name][local_index]
                    .float()
                    .cpu()
                    for name in option_names
                ],
                dim=0,
            )

            option_by_index[qindex] = (
                options
            )

        print(
            f"  option shard "
            f"{shard_index}/{len(ofiles)}: "
            f"{len(indices)} rows"
        )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    combined = {}

    for qindex, record in question_by_index.items():

        if qindex not in option_by_index:
            raise RuntimeError(
                f"Missing option cache for question index {qindex}"
            )

        combined[qindex] = {
            "question_embedding": record[
                "embedding"
            ],
            "options_embedding": option_by_index[
                qindex
            ],
            "question": record.get(
                "question"
            ),
        }

    print(
        "\n  question indices:",
        len(question_by_index),
    )

    print(
        "  option indices:",
        len(option_by_index),
    )

    print(
        "  combined MCQ records:",
        len(combined),
    )

    return (
        combined,
        question_by_text,
    )


# =============================================================================
# OPTION PARSING
# =============================================================================

def parse_four_options(query: str):
    """
    Parse the four A-D options from the standard MCQ text representation.

    Example:
        ... a) 50 to 70%, b) 10 to 20%, c) 90 to 100%, d) 70 to 80%
    """
    pattern = re.compile(
        r"(?:^|,\s*|\s+)"
        r"([abcd])\)"
        r"\s*"
        r"(.*?)"
        r"(?=,\s*[abcd]\)|$)",
        flags=re.IGNORECASE,
    )

    matches = pattern.findall(
        str(query)
    )

    result = {}

    for label, value in matches:
        result[
            label.lower()
        ] = value.strip()

    if set(result.keys()) != {
        "a",
        "b",
        "c",
        "d",
    }:
        raise RuntimeError(
            "Could not parse exactly four MCQ options.\n"
            f"Query:\n{query}\n"
            f"Parsed:\n{result}"
        )

    return [
        result["a"],
        result["b"],
        result["c"],
        result["d"],
    ]


# =============================================================================
# RESULT CHECKING
# =============================================================================

def verify_common_result(
    result,
    expected_specialist,
    expected_modality,
    expected_mode,
):
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
    ] != "single_image_vqa":
        raise RuntimeError(
            f"Expected single_image_vqa, got "
            f"{analysis['task']}"
        )

    if evidence.get(
        "specialist"
    ) != expected_specialist:
        raise RuntimeError(
            "Unexpected VQA specialist:\n"
            f"expected={expected_specialist}\n"
            f"actual={evidence.get('specialist')}"
        )

    if evidence.get(
        "modality"
    ) != expected_modality:
        raise RuntimeError(
            "Unexpected VQA modality:\n"
            f"expected={expected_modality}\n"
            f"actual={evidence.get('modality')}"
        )

    if trace.get(
        "status"
    ) != "success":
        raise RuntimeError(
            "Outer agent trace does not report success."
        )

    if not isinstance(
        evidence,
        dict,
    ):
        raise RuntimeError(
            "VQA evidence is not a dict."
        )

    return (
        analysis,
        evidence,
        trace,
    )


# =============================================================================
# BINARY TEST
# =============================================================================

def run_binary_test(
    agent,
    row,
    modality,
    question_lookup,
    s1,
    s2,
):
    expected_specialist = (
        EXPECTED_BINARY[
            modality
        ]
    )

    qid = str(
        row["ID"]
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
        "yes",
        "no",
    }:
        raise RuntimeError(
            f"Unexpected binary label: {ground_truth}"
        )

    if qid not in question_lookup:
        raise RuntimeError(
            f"Missing binary question embedding for ID {qid}"
        )

    question_embedding = (
        question_lookup[qid]
    )

    if modality == "s1":
        visual = s1
    else:
        visual = s2

    visual = visual.unsqueeze(
        0
    )

    question_embedding = (
        question_embedding
        .unsqueeze(0)
    )

    print()
    print("=" * 80)
    print(
        f"REAL {modality.upper()} BINARY VQA — AGENT EXECUTION"
    )
    print("=" * 80)

    print(
        "Dataset ID:",
        qid,
    )

    print(
        "Patch ID:",
        row["patch_id"],
    )

    print(
        "Question:",
        query,
    )

    print(
        "Ground truth:",
        ground_truth,
    )

    print("\nINPUT SHAPES")

    print(
        f"  {modality.upper()}:",
        tuple(visual.shape),
    )

    print(
        "  Text:",
        tuple(
            question_embedding.shape
        ),
    )

    if tuple(
        visual.shape
    ) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "Invalid visual shape."
        )

    if tuple(
        question_embedding.shape
    ) != (
        1,
        768,
    ):
        raise RuntimeError(
            "Invalid question shape."
        )

    start = time.perf_counter()

    kwargs = {
        "query": query,
        "input_modalities": [
            modality
        ],
        "text_embedding": question_embedding,
        "point": None,
        "reference": None,
        "patch_id": str(
            row["patch_id"]
        ),
    }

    if modality == "s1":
        kwargs["s1"] = visual
    else:
        kwargs["s2"] = visual

    result = agent.execute(
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    (
        analysis,
        evidence,
        trace,
    ) = verify_common_result(
        result,
        expected_specialist,
        modality,
        "binary",
    )

    answer = str(
        evidence.get(
            "answer"
        )
    ).strip().lower()

    if answer not in {
        "yes",
        "no",
    }:
        raise RuntimeError(
            f"Binary specialist returned invalid answer: {answer}"
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
        "  modality:",
        evidence.get(
            "modality"
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
        "  answer_confidence:",
        evidence.get(
            "answer_confidence"
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
        "  status:",
        trace.get(
            "status"
        ),
    )

    print(
        "  validation_ms:",
        trace.get(
            "validation_ms"
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
        f"\nREAL {modality.upper()} BINARY VQA: PASS"
    )

    return {
        "modality": modality,
        "specialist": evidence.get(
            "specialist"
        ),
        "answer": answer,
        "ground_truth": ground_truth,
        "correct": correct,
        "parameter_count": evidence.get(
            "parameter_count"
        ),
        "checkpoint_epoch": evidence.get(
            "checkpoint_epoch"
        ),
    }


# =============================================================================
# MCQ TEST
# =============================================================================

def run_mcq_test(
    agent,
    row,
    modality,
    mcq_lookup,
    question_by_text,
    s1,
    s2,
):
    expected_specialist = (
        EXPECTED_MCQ[
            modality
        ]
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
            f"Unexpected MCQ label: {ground_truth}"
        )

    # -------------------------------------------------------------------------
    # Prefer the verified explicit question_embedding_index when available.
    # Otherwise recover the index from the actual question text in the cache.
    # -------------------------------------------------------------------------

    if (
        "question_embedding_index"
        in row.index
    ):
        qindex = int(
            row["question_embedding_index"]
        )

        if qindex not in mcq_lookup:
            raise RuntimeError(
                f"MCQ question_embedding_index {qindex} "
                "is missing from the cache."
            )

    else:
        normalized = normalize_text(
            query
        )

        if normalized not in question_by_text:
            raise RuntimeError(
                "Could not map MCQ question text to "
                "question embedding cache."
            )

        qindex = question_by_text[
            normalized
        ]

    record = mcq_lookup[
        qindex
    ]

    question_embedding = (
        record[
            "question_embedding"
        ]
    )

    option_embeddings = (
        record[
            "options_embedding"
        ]
    )

    options = parse_four_options(
        query
    )

    if tuple(
        option_embeddings.shape
    ) != (
        4,
        768,
    ):
        raise RuntimeError(
            "Unexpected MCQ option embedding shape: "
            f"{tuple(option_embeddings.shape)}"
        )

    if modality == "s1":
        visual = s1
    else:
        visual = s2

    visual = visual.unsqueeze(
        0
    )

    question_embedding = (
        question_embedding
        .unsqueeze(0)
    )

    option_embeddings = (
        option_embeddings
        .unsqueeze(0)
    )

    print()
    print("=" * 80)
    print(
        f"REAL {modality.upper()} MCQ VQA — AGENT EXECUTION"
    )
    print("=" * 80)

    print(
        "Patch ID:",
        row["patch_id"],
    )

    print(
        "Question index:",
        qindex,
    )

    print(
        "Question:",
        query,
    )

    print(
        "Options:",
        options,
    )

    print(
        "Ground truth:",
        ground_truth,
    )

    print("\nINPUT SHAPES")

    print(
        f"  {modality.upper()}:",
        tuple(
            visual.shape
        ),
    )

    print(
        "  Text:",
        tuple(
            question_embedding.shape
        ),
    )

    print(
        "  Options:",
        tuple(
            option_embeddings.shape
        ),
    )

    if tuple(
        visual.shape
    ) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "Invalid visual shape."
        )

    if tuple(
        question_embedding.shape
    ) != (
        1,
        768,
    ):
        raise RuntimeError(
            "Invalid question shape."
        )

    if tuple(
        option_embeddings.shape
    ) != (
        1,
        4,
        768,
    ):
        raise RuntimeError(
            "Invalid option shape."
        )

    start = time.perf_counter()

    kwargs = {
        "query": query,
        "input_modalities": [
            modality
        ],
        "text_embedding": question_embedding,
        "options": option_embeddings,
        "point": None,
        "reference": None,
        "patch_id": str(
            row["patch_id"]
        ),
    }

    if modality == "s1":
        kwargs["s1"] = visual
    else:
        kwargs["s2"] = visual

    result = agent.execute(
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    (
        analysis,
        evidence,
        trace,
    ) = verify_common_result(
        result,
        expected_specialist,
        modality,
        "mcq",
    )

    answer_raw = evidence.get(
        "answer"
    )

    answer = str(
        answer_raw
    ).strip().lower()

    if answer.startswith(
        "("
    ):
        answer = answer.strip(
            "()"
        )

    if answer not in {
        "a",
        "b",
        "c",
        "d",
    }:
        raise RuntimeError(
            f"MCQ specialist returned invalid answer: "
            f"{answer_raw}"
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
        "  modality:",
        evidence.get(
            "modality"
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

    if "option_probabilities" in evidence:
        print(
            "  option_probabilities:",
            evidence[
                "option_probabilities"
            ],
        )

    if "probabilities" in evidence:
        print(
            "  probabilities:",
            evidence[
                "probabilities"
            ],
        )

    print("\nEXECUTION TRACE")

    print(
        "  selected_tool:",
        trace.get(
            "selected_tool"
        ),
    )

    print(
        "  status:",
        trace.get(
            "status"
        ),
    )

    print(
        "  validation_ms:",
        trace.get(
            "validation_ms"
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
        f"\nREAL {modality.upper()} MCQ VQA: PASS"
    )

    return {
        "modality": modality,
        "specialist": evidence.get(
            "specialist"
        ),
        "answer": answer,
        "ground_truth": ground_truth,
        "correct": correct,
        "parameter_count": evidence.get(
            "parameter_count"
        ),
        "checkpoint_epoch": evidence.get(
            "checkpoint_epoch"
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print(
        "SATQUERYAI — REAL-DATA SINGLE-MODALITY VQA SMOKE TEST"
    )
    print("=" * 80)

    print("\nDEVICE: CPU")
    print("TRAINING: NOT STARTED")
    print("CHECKPOINTS: FROZEN")

    # -------------------------------------------------------------------------
    # PATH CHECK
    # -------------------------------------------------------------------------

    print("\nPATH CHECK")

    require_path(
        BINARY_PARQUET,
        "Binary VQA validation data",
    )

    require_path(
        MCQ_PARQUET,
        "MCQ VQA validation data",
    )

    require_path(
        DOFA_DIR,
        "DOFA cache",
    )

    require_path(
        BINARY_Q_DIR,
        "Binary question cache",
    )

    require_path(
        MCQ_Q_DIR,
        "MCQ question cache",
    )

    require_path(
        MCQ_OPTION_DIR,
        "MCQ option cache",
    )

    print(
        "  Binary validation: PASS"
    )

    print(
        "  MCQ validation: PASS"
    )

    print(
        "  DOFA cache: PASS"
    )

    print(
        "  Binary question cache: PASS"
    )

    print(
        "  MCQ question cache: PASS"
    )

    print(
        "  MCQ option cache: PASS"
    )

    # -------------------------------------------------------------------------
    # BUILD AGENT
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BUILDING RUNTIME AGENT")
    print("=" * 80)

    agent, _, _ = create_runtime_agent()

    print(
        "Registered capabilities:",
        agent.registry.capabilities(),
    )

    # -------------------------------------------------------------------------
    # LOAD CACHES
    # -------------------------------------------------------------------------

    binary_question_lookup = (
        load_binary_question_lookup()
    )

    (
        mcq_lookup,
        question_by_text,
    ) = load_mcq_question_option_lookup()

    # -------------------------------------------------------------------------
    # LOAD REAL VALIDATION TABLES
    # -------------------------------------------------------------------------

    binary_df = pd.read_parquet(
        BINARY_PARQUET
    )

    mcq_df = pd.read_parquet(
        MCQ_PARQUET
    )

    binary_val = binary_df[
        binary_df["model_split"] == "val"
    ].copy()

    mcq_val = mcq_df[
        mcq_df["model_split"] == "val"
    ].copy()

    if binary_val.empty:
        raise RuntimeError(
            "Binary validation split is empty."
        )

    if mcq_val.empty:
        raise RuntimeError(
            "MCQ validation split is empty."
        )

    # Use the first verified validation sample.
    binary_row = binary_val.iloc[0]
    mcq_row = mcq_val.iloc[0]

    print("\nVALIDATION SAMPLE SELECTION")

    print(
        "  Binary ID:",
        binary_row["ID"],
    )

    print(
        "  Binary patch:",
        binary_row["patch_id"],
    )

    print(
        "  MCQ ID:",
        mcq_row["ID"],
    )

    print(
        "  MCQ patch:",
        mcq_row["patch_id"],
    )

    # -------------------------------------------------------------------------
    # LOAD DOFA FEATURES
    #
    # Use one patch at a time so the CPU test stays lightweight.
    # -------------------------------------------------------------------------

    binary_s1, binary_s2 = find_dofa_features(
        str(
            binary_row["patch_id"]
        )
    )

    if str(
        mcq_row["patch_id"]
    ) == str(
        binary_row["patch_id"]
    ):
        mcq_s1 = binary_s1
        mcq_s2 = binary_s2
    else:
        mcq_s1, mcq_s2 = find_dofa_features(
            str(
                mcq_row["patch_id"]
            )
        )

    # -------------------------------------------------------------------------
    # BINARY: S1
    # -------------------------------------------------------------------------

    binary_s1_result = run_binary_test(
        agent=agent,
        row=binary_row,
        modality="s1",
        question_lookup=binary_question_lookup,
        s1=binary_s1,
        s2=binary_s2,
    )

    # -------------------------------------------------------------------------
    # BINARY: S2
    # -------------------------------------------------------------------------

    binary_s2_result = run_binary_test(
        agent=agent,
        row=binary_row,
        modality="s2",
        question_lookup=binary_question_lookup,
        s1=binary_s1,
        s2=binary_s2,
    )

    # -------------------------------------------------------------------------
    # MCQ: S1
    # -------------------------------------------------------------------------

    mcq_s1_result = run_mcq_test(
        agent=agent,
        row=mcq_row,
        modality="s1",
        mcq_lookup=mcq_lookup,
        question_by_text=question_by_text,
        s1=mcq_s1,
        s2=mcq_s2,
    )

    # -------------------------------------------------------------------------
    # MCQ: S2
    # -------------------------------------------------------------------------

    mcq_s2_result = run_mcq_test(
        agent=agent,
        row=mcq_row,
        modality="s2",
        mcq_lookup=mcq_lookup,
        question_by_text=question_by_text,
        s1=mcq_s1,
        s2=mcq_s2,
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    results = [
        binary_s1_result,
        binary_s2_result,
        mcq_s1_result,
        mcq_s2_result,
    ]

    for result in results:

        if result[
            "parameter_count"
        ] not in (
            558722,
            920065,
        ):
            raise RuntimeError(
                "Unexpected single-modality VQA parameter count."
            )

    print()
    print("=" * 80)
    print(
        "SINGLE-MODALITY VQA RESULTS"
    )
    print("=" * 80)

    print(
        "  S1 Binary:",
        binary_s1_result["specialist"],
        "| answer:",
        binary_s1_result["answer"],
        "| sample_correct:",
        binary_s1_result["correct"],
    )

    print(
        "  S2 Binary:",
        binary_s2_result["specialist"],
        "| answer:",
        binary_s2_result["answer"],
        "| sample_correct:",
        binary_s2_result["correct"],
    )

    print(
        "  S1 MCQ:",
        mcq_s1_result["specialist"],
        "| answer:",
        mcq_s1_result["answer"],
        "| sample_correct:",
        mcq_s1_result["correct"],
    )

    print(
        "  S2 MCQ:",
        mcq_s2_result["specialist"],
        "| answer:",
        mcq_s2_result["answer"],
        "| sample_correct:",
        mcq_s2_result["correct"],
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  'sample_correct' is only correctness on the single "
        "smoke-test example."
    )

    print(
        "  It is NOT a new validation-accuracy measurement."
    )

    print(
        "\nSATQUERYAI REAL-DATA SINGLE-MODALITY "
        "VQA SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()