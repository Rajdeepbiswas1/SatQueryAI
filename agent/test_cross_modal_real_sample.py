from pathlib import Path

import pandas as pd
import torch

from agent.bootstrap import create_runtime_agent

from agent.response_formatter import (
    format_agent_response,
)

PROJECT_ROOT = Path(r"D:\SatQueryAI")

DOFA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

QUESTION_EMBED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def find_vqa_parquet():
    candidates = sorted(PROCESSED_DIR.glob("*.parquet"))

    for path in candidates:
        try:
            df = pd.read_parquet(path)

            required = {"ID", "patch_id", "input", "output"}
            if required.issubset(df.columns):
                return path, df

        except Exception:
            continue

    raise FileNotFoundError(
        "Could not find a parquet file containing "
        "ID, patch_id, input, output."
    )


def find_question_embedding(question_id):
    shards = sorted(QUESTION_EMBED_DIR.glob("*.pt"))

    if not shards:
        raise FileNotFoundError(
            f"No question embedding shards found in {QUESTION_EMBED_DIR}"
        )

    target = str(question_id)

    for shard_path in shards:
        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        ids = shard["ids"]

        for i, value in enumerate(ids):
            if str(value) == target:
                return (
                    shard_path,
                    shard["patch_ids"][i],
                    shard["question_embeddings"][i:i + 1].float(),
                )

    raise KeyError(
        f"Question ID {question_id} was not found in the "
        "question embedding cache."
    )


def find_dofa_pair(patch_id):
    shards = sorted(DOFA_DIR.glob("*.pt"))

    if not shards:
        raise FileNotFoundError(
            f"No DOFA shards found in {DOFA_DIR}"
        )

    target = str(patch_id)

    for shard_path in shards:
        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        patch_ids = shard["patch_ids"]

        for i, value in enumerate(patch_ids):
            if str(value) == target:
                return (
                    shard_path,
                    shard["s1"][i:i + 1].float(),
                    shard["s2"][i:i + 1].float(),
                )

    raise KeyError(
        f"Patch ID {patch_id} was not found in the DOFA cache."
    )


def main():
    print("=" * 80)
    print("SATQUERYAI — REAL CROSS-MODAL SAMPLE VERIFICATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load the actual cross-modal VQA dataset
    # ------------------------------------------------------------------
    dataset_path, df = find_vqa_parquet()

    row = df.iloc[0]

    question_id = row["ID"]
    patch_id_from_dataset = row["patch_id"]
    question = str(row["input"])
    expected_output = str(row["output"])

    print()
    print("Dataset:")
    print(" ", dataset_path)

    print()
    print("Question ID:")
    print(" ", question_id)

    print("Dataset patch ID:")
    print(" ", patch_id_from_dataset)

    print("Question:")
    print(" ", question)

    print("Expected dataset output:")
    print(" ", expected_output)

    # ------------------------------------------------------------------
    # 2. Find the exact question embedding
    # ------------------------------------------------------------------
    (
        question_shard,
        patch_id_from_embedding,
        text_embedding,
    ) = find_question_embedding(question_id)

    print()
    print("Question embedding shard:")
    print(" ", question_shard)

    print("Embedding patch ID:")
    print(" ", patch_id_from_embedding)

    print("Question embedding shape:")
    print(" ", tuple(text_embedding.shape))

    if str(patch_id_from_dataset) != str(patch_id_from_embedding):
        raise RuntimeError(
            "Dataset patch ID and question-cache patch ID do not match."
        )

    if text_embedding.shape != (1, 768):
        raise RuntimeError(
            f"Unexpected question embedding shape: "
            f"{tuple(text_embedding.shape)}"
        )

    # ------------------------------------------------------------------
    # 3. Find the exact matching DOFA S1 + S2 pair
    # ------------------------------------------------------------------
    dofa_shard, s1, s2 = find_dofa_pair(patch_id_from_dataset)

    print()
    print("DOFA shard:")
    print(" ", dofa_shard)

    print("S1 shape:")
    print(" ", tuple(s1.shape))

    print("S2 shape:")
    print(" ", tuple(s2.shape))

    if s1.shape != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S1 shape: {tuple(s1.shape)}"
        )

    if s2.shape != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(s2.shape)}"
        )

    if not torch.isfinite(s1).all():
        raise RuntimeError("S1 contains non-finite values.")

    if not torch.isfinite(s2).all():
        raise RuntimeError("S2 contains non-finite values.")

    if not torch.isfinite(text_embedding).all():
        raise RuntimeError(
            "Question embedding contains non-finite values."
        )

    # ------------------------------------------------------------------
    # 4. Build the actual runtime agent
    # ------------------------------------------------------------------
    runtime_agent, _, _ = create_runtime_agent()

    print()
    print("Registered capabilities:")
    print(" ", runtime_agent.registry.capabilities())

    if not runtime_agent.registry.available("cross_modal"):
        raise RuntimeError(
            "Cross-modal capability is unavailable."
        )

    # ------------------------------------------------------------------
    # 5. Execute the REAL dataset question
    # ------------------------------------------------------------------
    
    print()
    print("Agent query:")
    print(" ", question)

    result = runtime_agent.execute(
        question,
        input_modalities=["s1", "s2"],
        s1=s1,
        s2=s2,
        text_embedding=text_embedding,
        patch_id=str(patch_id_from_dataset),
    )
    
    formatted = format_agent_response(
        result
    )

    print()
    print("=" * 80)
    print("NORMALIZED APPLICATION RESPONSE")
    print("=" * 80)
    print(formatted)

    # ------------------------------------------------------------------
    # 6. Print the important result
    # ------------------------------------------------------------------
    print()
    print("-" * 80)
    print("MODEL RESULT")
    print("-" * 80)
    print("Top-level answer:")
    print(" ", result.get("answer"))

    print()
    print("Answer status:")
    print(" ", result.get("answer_status"))

    evidence = result.get("evidence")

    if isinstance(evidence, dict):
        print()
        print("Specialist answer:")
        print(" ", evidence.get("answer"))

        print("Specialist confidence:")
        print(" ", evidence.get("answer_confidence"))

        print("Specialist:")
        print(" ", evidence.get("specialist"))

        print("Route:")
        print(" ", evidence.get("cross_modal_vqa_route"))

        print("Evidence type:")
        print(" ", evidence.get("evidence_type"))

    print()
    print("Execution trace:")
    print(" ", result.get("execution_trace"))

    print()
    print("=" * 80)
    print("REAL CROSS-MODAL SAMPLE VERIFICATION: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()