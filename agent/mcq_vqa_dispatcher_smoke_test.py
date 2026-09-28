# =============================================================================
# SATQUERYAI — REAL MCQ VQA DISPATCHER SMOKE TEST
# =============================================================================
#
# Purpose:
#   Verify the PERMANENT runtime path:
#
#       Agent
#          ↓
#       single_image_vqa
#          ↓
#       SingleImageVQARouter
#          ↓
#       MCQOptionAwareVQA
#
# This test uses:
#   - real validation annotation
#   - real DOFA cached features
#   - real option-aware question/option cache
#   - real frozen MCQ checkpoint through the runtime agent
#
# It does NOT:
#   - train anything
#   - modify checkpoints
#   - replace the registry locally
#   - modify specialist architecture
# =============================================================================

from pathlib import Path
import re
import time

import pandas as pd
import torch

from agent.bootstrap import create_runtime_agent


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

ANNOTATIONS = (
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

OPTION_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings"
)


# =============================================================================
# VERIFIED REAL-DATA SAMPLE
# =============================================================================

EXPECTED_ID = 6479

EXPECTED_PATCH = (
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90"
)

EXPECTED_ANSWER = "A"


# =============================================================================
# HELPERS
# =============================================================================

def find_feature_tensors(blob):
    """
    Find the S1/S2 feature tensors without depending on exact cache key names.

    Expected tensor shape:
        [N, 196, 768]
    """
    candidates = []

    for key, value in blob.items():
        if (
            torch.is_tensor(value)
            and value.ndim == 3
            and tuple(value.shape[1:]) == (196, 768)
        ):
            candidates.append((str(key), value))

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not find both S1 and S2 feature tensors."
        )

    s1 = None
    s2 = None

    for key, value in candidates:
        key_lower = key.lower()

        if s1 is None and (
            "s1" in key_lower
            or "sentinel1" in key_lower
            or "sar" in key_lower
        ):
            s1 = value

        if s2 is None and (
            "s2" in key_lower
            or "sentinel2" in key_lower
            or "optical" in key_lower
        ):
            s2 = value

    if s1 is None or s2 is None:
        if len(candidates) == 2:
            s1 = candidates[0][1]
            s2 = candidates[1][1]
        else:
            raise RuntimeError(
                "Found multiple feature tensors but could not "
                "identify S1 and S2."
            )

    return s1, s2


def parse_mcq(text):
    """
    Parse:
        question a) ... b) ... c) ... d) ...

    Returns:
        question: str
        options: list[str] with exactly 4 entries
    """

    match = re.match(
        r"^(?P<question>.*?)\s+"
        r"a\)\s*(?P<a>.*?)\s*,\s*"
        r"b\)\s*(?P<b>.*?)\s*,\s*"
        r"c\)\s*(?P<c>.*?)\s*,\s*"
        r"d\)\s*(?P<d>.*?)\s*$",
        text.strip(),
        flags=re.IGNORECASE,
    )

    if match is None:
        raise RuntimeError(
            "Could not parse the MCQ input into a question and "
            "four options."
        )

    question = match.group("question").strip()

    options = [
        match.group("a").strip(),
        match.group("b").strip(),
        match.group("c").strip(),
        match.group("d").strip(),
    ]

    return question, options


def find_option_cache_sample(question_index):
    """
    Find the cached question/options row by question index.
    """

    shards = sorted(
        OPTION_CACHE_DIR.glob("shard_*.pt")
    )

    if not shards:
        raise RuntimeError(
            f"No MCQ option cache shards found in {OPTION_CACHE_DIR}"
        )

    for shard_path in shards:
        blob = torch.load(
            shard_path,
            map_location="cpu",
        )

        question_indices = blob.get(
            "question_indices"
        )

        if question_indices is None:
            continue

        for position, value in enumerate(question_indices):
            if int(value) == question_index:
                return shard_path, blob, position

    raise RuntimeError(
        f"Question index {question_index} was not found "
        "in the MCQ option cache."
    )


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("SATQUERYAI — REAL MCQ VQA DISPATCHER SMOKE TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Validate required files/directories
    # -------------------------------------------------------------------------

    if not ANNOTATIONS.exists():
        raise RuntimeError(
            f"Missing annotations: {ANNOTATIONS}"
        )

    print(
        "Annotations         : PASS ->",
        ANNOTATIONS,
    )

    if not DOFA_DIR.exists():
        raise RuntimeError(
            f"Missing DOFA cache: {DOFA_DIR}"
        )

    print(
        "DOFA cache           : PASS ->",
        DOFA_DIR,
    )

    if not OPTION_CACHE_DIR.exists():
        raise RuntimeError(
            f"Missing MCQ option cache: {OPTION_CACHE_DIR}"
        )

    print(
        "Option cache         : PASS ->",
        OPTION_CACHE_DIR,
    )

    # -------------------------------------------------------------------------
    # Load the verified real validation sample
    # -------------------------------------------------------------------------

    print()
    print("Loading validation annotations...")

    df = pd.read_parquet(
        ANNOTATIONS
    )

    required_columns = {
        "ID",
        "input",
        "output",
        "type",
        "category",
        "patch_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required annotation columns: {sorted(missing)}"
        )

    rows = df[df["ID"] == EXPECTED_ID]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row for ID {EXPECTED_ID}, "
            f"found {len(rows)}."
        )

    row = rows.iloc[0]

    patch_id = str(row["patch_id"])
    mcq_input = str(row["input"])
    stored_answer = str(row["output"]).strip().upper()

    print()
    print("Selected validation sample:")
    print("  ID:", int(row["ID"]))
    print("  Patch:", patch_id)
    print("  Type:", row["type"])
    print("  Category:", row["category"])
    print("  Query:", mcq_input)
    print("  Expected answer:", stored_answer)

    if patch_id != EXPECTED_PATCH:
        raise RuntimeError(
            "The verified validation sample changed unexpectedly."
        )

    if stored_answer != EXPECTED_ANSWER:
        raise RuntimeError(
            "The stored answer for the verified sample changed "
            f"from {EXPECTED_ANSWER} to {stored_answer}."
        )

    # -------------------------------------------------------------------------
    # Parse question + options
    # -------------------------------------------------------------------------

    question, option_texts = parse_mcq(
        mcq_input
    )

    print()
    print("Parsed MCQ:")
    print("  Question:", question)

    for index, option in enumerate(option_texts):
        print(
            f"  {chr(65 + index)}: {option}"
        )

    if len(option_texts) != 4:
        raise RuntimeError(
            "MCQ parser did not produce exactly four options."
        )

    print(
        "MCQ parsing: PASS"
    )

    # -------------------------------------------------------------------------
    # Load real option-aware cache
    # -------------------------------------------------------------------------

    print()
    print("Searching MCQ option cache...")

    option_shard_path, option_blob, option_position = (
        find_option_cache_sample(
            EXPECTED_ID
        )
    )

    print(
        "Selected option shard:",
        option_shard_path,
    )

    print(
        "Selected position:",
        option_position,
    )

    question_embedding = option_blob[
        "question_embeddings"
    ][option_position]

    options = torch.stack(
        [
            option_blob["option_a_embeddings"][option_position],
            option_blob["option_b_embeddings"][option_position],
            option_blob["option_c_embeddings"][option_position],
            option_blob["option_d_embeddings"][option_position],
        ],
        dim=0,
    )

    print(
        "Question embedding shape:",
        tuple(question_embedding.shape),
    )

    print(
        "Options embedding shape:",
        tuple(options.shape),
    )

    # -------------------------------------------------------------------------
    # Load real DOFA cached features
    # -------------------------------------------------------------------------

    print()
    print("Searching DOFA cache...")

    dofa_shards = sorted(
        DOFA_DIR.glob("shard_*.pt")
    )

    if not dofa_shards:
        raise RuntimeError(
            "No DOFA cache shards found."
        )

    selected_dofa = None
    selected_dofa_blob = None
    selected_dofa_position = None

    for shard_path in dofa_shards:

        blob = torch.load(
            shard_path,
            map_location="cpu",
        )

        # Try to find patch IDs using common cache key names.
        patch_values = None

        for key in (
            "patch_ids",
            "patch_id",
            "patch_names",
            "ids",
            "id",
        ):
            if key in blob:
                patch_values = blob[key]
                break

        if patch_values is None:
            continue

        for position, value in enumerate(patch_values):

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            if str(value) == patch_id:
                selected_dofa = shard_path
                selected_dofa_blob = blob
                selected_dofa_position = position
                break

        if selected_dofa is not None:
            break

    # The verified real sample is known to be shard 0000, row 0.
    # Use that location if the cache does not expose a patch-ID field.
    if selected_dofa is None:

        fallback = DOFA_DIR / "shard_0000.pt"

        if not fallback.exists():
            raise RuntimeError(
                "Could not locate verified DOFA shard_0000.pt."
            )

        selected_dofa = fallback
        selected_dofa_blob = torch.load(
            fallback,
            map_location="cpu",
        )
        selected_dofa_position = 0

    print(
        "Selected DOFA shard:",
        selected_dofa,
    )

    print(
        "Selected DOFA position:",
        selected_dofa_position,
    )

    s1_all, s2_all = find_feature_tensors(
        selected_dofa_blob
    )

    s1 = s1_all[
        selected_dofa_position
    ]

    s2 = s2_all[
        selected_dofa_position
    ]

    print()
    print("Real cached feature shapes:")
    print("  S1:", tuple(s1.shape))
    print("  S2:", tuple(s2.shape))
    print("  Q :", tuple(question_embedding.shape))
    print("  O :", tuple(options.shape))

    if tuple(s1.shape) != (196, 768):
        raise RuntimeError(
            f"Unexpected S1 shape: {tuple(s1.shape)}"
        )

    if tuple(s2.shape) != (196, 768):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(s2.shape)}"
        )

    if tuple(question_embedding.shape) != (768,):
        raise RuntimeError(
            "Unexpected question embedding shape."
        )

    if tuple(options.shape) != (4, 768):
        raise RuntimeError(
            f"Unexpected options shape: {tuple(options.shape)}"
        )

    tensors = {
        "s1": s1,
        "s2": s2,
        "question_embedding": question_embedding,
        "options": options,
    }

    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite values detected in {name}."
            )

    print(
        "Real input shape/finiteness checks: PASS"
    )

    # -------------------------------------------------------------------------
    # Build the ACTUAL runtime agent
    # -------------------------------------------------------------------------

    print()
    print("Building runtime agent...")

    runtime_start = time.perf_counter()

    agent, grounding_tool, binary_vqa_tool = (
        create_runtime_agent()
    )

    build_time = (
        time.perf_counter() - runtime_start
    ) * 1000.0

    capabilities = (
        agent.registry.capabilities()
    )

    print(
        "Registered capabilities:",
        capabilities,
    )

    if "single_image_vqa" not in capabilities:
        raise RuntimeError(
            "single_image_vqa capability is not registered."
        )

    registry_entry = (
        agent.registry.get(
            "single_image_vqa"
        )
    )

    if registry_entry is None:
        raise RuntimeError(
            "single_image_vqa registry entry is missing."
        )

    print(
        "VQA registry tool:",
        registry_entry["tool_name"],
    )

    if registry_entry["tool_name"] != "SingleImageVQARouter":
        raise RuntimeError(
            "Permanent single-image VQA dispatcher is not registered."
        )

    print(
        "Permanent VQA dispatcher registry: PASS"
    )

    # -------------------------------------------------------------------------
    # Execute through the REAL agent
    # -------------------------------------------------------------------------

    print()
    print("Executing SatQueryAI agent through permanent dispatcher...")

    execution_start = time.perf_counter()

    result = agent.execute(
        query=mcq_input,
        s1=s1.unsqueeze(0),
        s2=s2.unsqueeze(0),
        text_embedding=question_embedding.unsqueeze(0),
        options=options.unsqueeze(0),
        patch_id=patch_id,
    )

    local_execution_time = (
        time.perf_counter() - execution_start
    ) * 1000.0

    print()
    print("Agent result:")
    print(
        "  answer_status:",
        result.get("answer_status"),
    )

    print(
        "  analysis:",
        result.get("analysis"),
    )

    print(
        "  execution_trace:",
        result.get("execution_trace"),
    )

    print(
        "  evidence keys:",
        list(
            result.get(
                "evidence",
                {}
            ).keys()
        ),
    )

    # -------------------------------------------------------------------------
    # Verify outer execution
    # -------------------------------------------------------------------------

    if result.get("answer_status") != "success":
        raise RuntimeError(
            "Agent execution did not succeed."
        )

    trace = result.get(
        "execution_trace",
        {}
    )

    if trace.get("selected_tool") != "SingleImageVQARouter":
        raise RuntimeError(
            "Agent did not select SingleImageVQARouter."
        )

    if trace.get("status") != "success":
        raise RuntimeError(
            "Agent execution trace does not report success."
        )

    evidence = result.get(
        "evidence",
        {}
    )

    # -------------------------------------------------------------------------
    # Verify dispatcher routing
    # -------------------------------------------------------------------------

    if evidence.get("vqa_route") != "mcq":
        raise RuntimeError(
            "Dispatcher did not route this query to MCQ VQA."
        )

    if evidence.get("specialist_selected") != "MCQOptionAwareVQA":
        raise RuntimeError(
            "Dispatcher did not select MCQOptionAwareVQA."
        )

    print(
        "MCQ dispatcher routing: PASS"
    )

    # -------------------------------------------------------------------------
    # Verify final prediction
    # -------------------------------------------------------------------------

    predicted_answer = (
        str(evidence.get("answer", ""))
        .strip()
        .upper()
    )

    answer_confidence = float(
        evidence.get(
            "answer_confidence",
            0.0,
        )
    )

    print()
    print("=" * 80)
    print("MCQ VQA PERMANENT DISPATCHER EXECUTION RESULT")
    print("=" * 80)

    print(
        "Predicted answer   :",
        predicted_answer,
    )

    print(
        "Expected answer    :",
        stored_answer,
    )

    print(
        "Answer confidence  :",
        f"{answer_confidence:.4f}",
    )

    print(
        "Specialist         :",
        evidence.get("specialist"),
    )

    print(
        "Selected tool      :",
        trace.get("selected_tool"),
    )

    print(
        "VQA route          :",
        evidence.get("vqa_route"),
    )

    print(
        "Checkpoint epoch   :",
        evidence.get("checkpoint_epoch"),
    )

    print(
        "Parameters         :",
        evidence.get("parameter_count"),
    )

    print(
        "Agent execution ms :",
        f"{trace.get('execution_ms', local_execution_time):.2f}",
    )

    print(
        "Local measured ms  :",
        f"{local_execution_time:.2f}",
    )

    print(
        "Bootstrap build ms :",
        f"{build_time:.2f}",
    )

    if predicted_answer != stored_answer:
        raise RuntimeError(
            f"Prediction mismatch: predicted={predicted_answer!r}, "
            f"expected={stored_answer!r}"
        )

    print()
    print(
        "Prediction matches stored answer: PASS"
    )

    print()
    print("=" * 80)
    print("SATQUERYAI REAL MCQ VQA PERMANENT DISPATCHER SMOKE TEST: PASS")
    print("=" * 80)