from pathlib import Path

import pandas as pd
import torch

from agent.bootstrap import create_runtime_agent


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

ANNOTATIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

QUESTION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

DOFA_CACHE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)


# ============================================================
# HELPERS
# ============================================================

def load_question_lookup():

    question_ids = []
    question_embeddings = []

    files = sorted(
        QUESTION_CACHE.glob("shard_*.pt")
    )

    if not files:
        raise RuntimeError(
            "No binary question embedding shards found."
        )

    for file in files:

        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        question_ids.extend(
            str(x)
            for x in data["ids"]
        )

        question_embeddings.append(
            data["question_embeddings"]
            .float()
        )

    embeddings = torch.cat(
        question_embeddings,
        dim=0,
    )

    return {
        qid: embeddings[i]
        for i, qid in enumerate(question_ids)
    }


def load_dofa_index():

    files = sorted(
        DOFA_CACHE.glob("shard_*.pt")
    )

    if not files:
        raise RuntimeError(
            "No DOFA feature shards found."
        )

    patch_to_shard = {}
    patch_to_position = {}

    for shard_index, file in enumerate(files):

        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        for position, patch_id in enumerate(
            data["patch_ids"]
        ):

            patch_id = str(patch_id)

            patch_to_shard[patch_id] = shard_index
            patch_to_position[patch_id] = position

        del data

    return files, patch_to_shard, patch_to_position


def find_question_column(df):

    candidates = [
        "input",
        "question",
        "query",
        "question_text",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        "Could not find a question-text column. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("SATQUERYAI — REAL BINARY VQA AGENT SMOKE TEST")
print("=" * 80)


# ------------------------------------------------------------
# Path checks
# ------------------------------------------------------------

for name, path in [
    ("Annotations", ANNOTATIONS),
    ("Question cache", QUESTION_CACHE),
    ("DOFA cache", DOFA_CACHE),
]:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    print(
        f"{name:20s}: PASS -> {path}"
    )


# ------------------------------------------------------------
# Load validation data
# ------------------------------------------------------------

print("\nLoading validation annotations...")

df = pd.read_parquet(
    ANNOTATIONS
)

df["answer_clean"] = (
    df["output"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[
    df["model_split"] == "val"
]

df = df[
    df["answer_clean"].isin(
        ["yes", "no"]
    )
].copy()

if len(df) == 0:
    raise RuntimeError(
        "No binary validation rows found."
    )

question_column = find_question_column(df)

row = df.iloc[0]

qid = str(row["ID"])
patch_id = str(row["patch_id"])
expected_answer = str(
    row["answer_clean"]
)

query = str(
    row[question_column]
).strip()

if not query:
    raise RuntimeError(
        "Selected validation row has an empty query."
    )


print("\nSelected validation sample:")
print("  ID:", qid)
print("  Patch:", patch_id)
print("  Query:", query)
print("  Expected answer:", expected_answer)


# ------------------------------------------------------------
# Question embedding
# ------------------------------------------------------------

print("\nLoading question embedding...")

question_lookup = load_question_lookup()

if qid not in question_lookup:
    raise RuntimeError(
        f"Missing question embedding: {qid}"
    )

text_embedding = (
    question_lookup[qid]
    .float()
)

if text_embedding.shape != (768,):
    raise RuntimeError(
        "Unexpected question embedding shape: "
        f"{tuple(text_embedding.shape)}"
    )

if not torch.isfinite(
    text_embedding
).all():

    raise RuntimeError(
        "Question embedding contains non-finite values."
    )

print(
    "Question embedding shape:",
    tuple(text_embedding.shape),
)


# ------------------------------------------------------------
# DOFA lookup
# ------------------------------------------------------------

print("\nIndexing DOFA shards...")

(
    dofa_files,
    patch_to_shard,
    patch_to_position,
) = load_dofa_index()

if patch_id not in patch_to_shard:
    raise RuntimeError(
        f"DOFA patch not found: {patch_id}"
    )

shard_index = patch_to_shard[
    patch_id
]

position = patch_to_position[
    patch_id
]

print(
    "DOFA shards:",
    len(dofa_files),
)

print(
    "Selected shard:",
    shard_index + 1,
)

print(
    "Selected position:",
    position,
)


# ------------------------------------------------------------
# Load selected DOFA shard
# ------------------------------------------------------------

data = torch.load(
    dofa_files[shard_index],
    map_location="cpu",
    weights_only=False,
)

s1 = (
    data["s1"][position]
    .float()
    .unsqueeze(0)
)

s2 = (
    data["s2"][position]
    .float()
    .unsqueeze(0)
)

del data


# ------------------------------------------------------------
# Input validation
# ------------------------------------------------------------

print("\nReal input shapes:")

print(
    "  S1:",
    tuple(s1.shape),
)

print(
    "  S2:",
    tuple(s2.shape),
)

print(
    "  Q :",
    tuple(text_embedding.unsqueeze(0).shape),
)

assert s1.shape == (
    1,
    196,
    768,
)

assert s2.shape == (
    1,
    196,
    768,
)

assert text_embedding.shape == (
    768,
)

assert torch.isfinite(s1).all()
assert torch.isfinite(s2).all()
assert torch.isfinite(text_embedding).all()

print(
    "Real input shape/finiteness checks: PASS"
)


# ------------------------------------------------------------
# Build runtime agent
# ------------------------------------------------------------

print("\nBuilding runtime agent...")

agent, grounding_tool, binary_vqa_tool = (
    create_runtime_agent()
)

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

print(
    "Binary VQA registry lookup: PASS"
)


# ------------------------------------------------------------
# Execute through the REAL agent path
# ------------------------------------------------------------

print("\nExecuting SatQueryAI agent...")

result = agent.execute(
    query=query,
    s1=s1,
    s2=s2,
    text_embedding=text_embedding,
)


# ------------------------------------------------------------
# Inspect result
# ------------------------------------------------------------

print("\nAgent result:")
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
        result["evidence"].keys()
    )
)


# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------

if result.get("answer_status") != "success":
    raise RuntimeError(
        "Agent execution did not return success."
    )

analysis = result["analysis"]

if analysis["task"] != "single_image_vqa":
    raise RuntimeError(
        "Agent selected the wrong task: "
        f"{analysis['task']}"
    )

evidence = result["evidence"]

if evidence["specialist"] != "CachedBinaryVQA":
    raise RuntimeError(
        "Wrong specialist selected: "
        f"{evidence['specialist']}"
    )

answer = evidence["answer"]

if answer not in {
    "yes",
    "no",
}:
    raise RuntimeError(
        f"Unexpected Binary VQA answer: {answer}"
    )

trace = result["execution_trace"]

if trace["selected_tool"] != "SingleImageVQARouter":
    raise RuntimeError(
        "Unexpected selected tool: "
        f"{trace['selected_tool']}"
    )

if trace["status"] != "success":
    raise RuntimeError(
        "Agent execution trace is not successful."
    )

if not trace["capability_available"]:
    raise RuntimeError(
        "Binary VQA capability was not available."
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

confidence = float(
    evidence["answer_confidence"]
)

print("\n" + "=" * 80)
print("BINARY VQA REAL AGENT EXECUTION RESULT")
print("=" * 80)

print(
    "Predicted answer :",
    answer,
)

print(
    "Expected answer  :",
    expected_answer,
)

print(
    "Answer confidence:",
    f"{confidence:.4f}",
)

print(
    "Specialist       :",
    evidence["specialist"],
)

print(
    "Checkpoint epoch :",
    evidence["checkpoint_epoch"],
)

print(
    "Parameters       :",
    evidence["parameter_count"],
)

print(
    "Agent status     :",
    trace["status"],
)

print(
    "Selected tool    :",
    trace["selected_tool"],
)

print(
    "Execution time   :",
    f"{trace['execution_ms']:.2f} ms",
)

print(
    "Total time       :",
    f"{trace['total_ms']:.2f} ms",
)

print("\n" + "=" * 80)
print("SATQUERYAI REAL BINARY VQA AGENT SMOKE TEST: PASS")
print("=" * 80)