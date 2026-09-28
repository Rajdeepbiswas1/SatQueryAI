from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import torch

from agent.temporal_upload_adapter import TemporalUploadAdapter
from agent.temporal_change_vqa_tool import (
    SatQueryAITemporalChangeV3Tool,
)


ROOT = Path(r"D:\SatQueryAI")

PAIR_ID = "00003"

RAW_T1 = (
    ROOT
    / "data"
    / "processed"
    / "temporal_raw"
    / "im1"
    / f"{PAIR_ID}.png"
)

RAW_T2 = (
    ROOT
    / "data"
    / "processed"
    / "temporal_raw"
    / "im2"
    / f"{PAIR_ID}.png"
)

FEATURE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

MANIFESTS = [
    ROOT / "data" / "processed" / "temporal_split" / "train_manifest.csv",
    ROOT / "data" / "processed" / "temporal_split" / "val_manifest.csv",
    ROOT / "data" / "processed" / "temporal_split" / "test_manifest.csv",
]

QUESTION_DIRS = [
    ROOT
    / "data"
    / "processed"
    / "temporal_question_embeddings",

    ROOT
    / "data"
    / "processed"
    / "temporal_val_question_embeddings",

    ROOT
    / "data"
    / "processed"
    / "temporal_test_question_embeddings",
]


def normalize_pair_id(value) -> str:
    return str(int(value)).zfill(5)


def find_question_row() -> dict:
    for manifest_path in MANIFESTS:

        if not manifest_path.is_file():
            continue

        df = pd.read_csv(manifest_path)

        if "pair_id" not in df.columns:
            continue

        matches = df[
            df["pair_id"].map(normalize_pair_id) == PAIR_ID
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        return {
            "question_id": int(row["question_id"]),
            "question": str(row["question"]).strip(),
            "question_type": str(row["question_type"]).strip(),
            "answer": str(row["answer"]).strip(),
            "manifest": str(manifest_path),
        }

    raise RuntimeError(
        f"No manifest row found for pair_id={PAIR_ID}"
    )


def find_cached_features():

    shard_paths = sorted(
        FEATURE_ROOT.glob("shard_*.pt")
    )

    if not shard_paths:
        raise FileNotFoundError(
            f"No feature shards found in {FEATURE_ROOT}"
        )

    for path in shard_paths:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids = [
            normalize_pair_id(x)
            for x in data["pair_ids"]
        ]

        if PAIR_ID not in pair_ids:
            continue

        index = pair_ids.index(PAIR_ID)

        return (
            data["t1_features"][index].float().unsqueeze(0),
            data["t2_features"][index].float().unsqueeze(0),
            data["diff_features"][index].float().unsqueeze(0),
            path,
        )

    raise KeyError(
        f"Cached pair {PAIR_ID} not found."
    )


def find_question_embedding(question_id: int):

    for directory in QUESTION_DIRS:

        if not directory.is_dir():
            continue

        for path in sorted(
            directory.glob("*.pt")
        ):

            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            ids = data.get("question_ids")

            if ids is None:
                ids = data.get("ids")

            embeddings = data.get(
                "embeddings"
            )

            if embeddings is None:
                embeddings = data.get(
                    "question_embeddings"
                )

            if ids is None or embeddings is None:
                continue

            for index, raw_id in enumerate(ids):

                if int(raw_id) == int(question_id):

                    embedding = (
                        embeddings[index]
                        .float()
                        .unsqueeze(0)
                        .contiguous()
                    )

                    return embedding, path

    raise KeyError(
        f"Question embedding not found for question_id={question_id}"
    )


def run():

    print("=" * 78)
    print("SATQUERYAI — TEMPORAL CACHE VS RAW ANSWER TEST")
    print("=" * 78)

    # ------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------

    if not RAW_T1.is_file():
        raise FileNotFoundError(
            f"Raw T1 not found: {RAW_T1}"
        )

    if not RAW_T2.is_file():
        raise FileNotFoundError(
            f"Raw T2 not found: {RAW_T2}"
        )

    print("Raw T1: PASS")
    print("Raw T2: PASS")

    # ------------------------------------------------------------
    # Same question for both paths
    # ------------------------------------------------------------

    metadata = find_question_row()

    question_id = metadata["question_id"]
    question = metadata["question"]
    dataset_answer = metadata["answer"]

    print()
    print("PAIR:", PAIR_ID)
    print("QUESTION ID:", question_id)
    print("QUESTION TYPE:", metadata["question_type"])
    print("QUESTION:", question)
    print("DATASET ANSWER:", dataset_answer)
    print("MANIFEST:", metadata["manifest"])

    text_embedding, question_path = (
        find_question_embedding(question_id)
    )

    print("QUESTION EMBEDDING:", question_path)
    print(
        "QUESTION EMBEDDING SHAPE:",
        tuple(text_embedding.shape),
    )

    if tuple(text_embedding.shape) != (1, 768):
        raise RuntimeError(
            "Invalid question embedding shape: "
            f"{tuple(text_embedding.shape)}"
        )

    # ------------------------------------------------------------
    # Historical cached features
    # ------------------------------------------------------------

    (
        cached_t1,
        cached_t2,
        cached_diff,
        feature_path,
    ) = find_cached_features()

    print()
    print("CACHE:", feature_path)
    print("CACHE T1:", tuple(cached_t1.shape))
    print("CACHE T2:", tuple(cached_t2.shape))
    print("CACHE DIFF:", tuple(cached_diff.shape))

    # ------------------------------------------------------------
    # Raw-image adapter
    # ------------------------------------------------------------

    print()
    print("Building raw temporal adapter...")

    adapter = TemporalUploadAdapter()

    raw = adapter.transform_pair(
        RAW_T1,
        RAW_T2,
    )

    raw_t1 = raw["t1"]
    raw_t2 = raw["t2"]
    raw_diff = raw["diff"]

    print("RAW T1:", tuple(raw_t1.shape))
    print("RAW T2:", tuple(raw_t2.shape))
    print("RAW DIFF:", tuple(raw_diff.shape))

    # ------------------------------------------------------------
    # Feature comparison
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("FEATURE DIFFERENCE")
    print("-" * 78)

    for name, cached, current in (
        ("T1", cached_t1, raw_t1),
        ("T2", cached_t2, raw_t2),
        ("Diff", cached_diff, raw_diff),
    ):

        delta = (
            current.float()
            - cached.float()
        )

        print(
            f"{name}: "
            f"max_abs={delta.abs().max().item():.8f} "
            f"mean_abs={delta.abs().mean().item():.8f}"
        )

    # ------------------------------------------------------------
    # Same Temporal V3 specialist
    # ------------------------------------------------------------

    print()
    print("Building Temporal V3 specialist...")

    tool = SatQueryAITemporalChangeV3Tool()

    print(
        "Checkpoint epoch:",
        tool.checkpoint_epoch,
    )

    print(
        "Parameter count:",
        tool.parameter_count,
    )

    # ------------------------------------------------------------
    # Cached inference
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("CACHED FEATURE INFERENCE")
    print("-" * 78)

    cached_result = tool(
        query=question,
        analysis=type(
            "Analysis",
            (),
            {"task": "temporal_change", "confidence": 1.0},
        )(),
        t1=cached_t1,
        t2=cached_t2,
        diff=cached_diff,
        text_embedding=text_embedding,
    )

    # ------------------------------------------------------------
    # Raw inference
    # ------------------------------------------------------------

    print()
    print("-" * 78)
    print("RAW ADAPTER INFERENCE")
    print("-" * 78)

    raw_result = tool(
        query=question,
        analysis=type(
            "Analysis",
            (),
            {"task": "temporal_change", "confidence": 1.0},
        )(),
        t1=raw_t1,
        t2=raw_t2,
        diff=raw_diff,
        text_embedding=text_embedding,
    )

    # ------------------------------------------------------------
    # Extract answer information
    # ------------------------------------------------------------

    cached_answer = cached_result["answer"]
    raw_answer = raw_result["answer"]

    cached_probability = float(
        cached_result["answer_probability"]
    )

    raw_probability = float(
        raw_result["answer_probability"]
    )

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print("Dataset answer :", dataset_answer)
    print("Cached answer  :", cached_answer)
    print("Raw answer     :", raw_answer)

    print(
        "Cached probability:",
        f"{cached_probability:.6f}",
    )

    print(
        "Raw probability   :",
        f"{raw_probability:.6f}",
    )

    print(
        "Same answer:",
        cached_answer == raw_answer,
    )

    print(
        "Same as dataset (cached):",
        cached_answer == dataset_answer,
    )

    print(
        "Same as dataset (raw):",
        raw_answer == dataset_answer,
    )

    print()

    if cached_answer == raw_answer:
        print(
            "CACHE VS RAW ANSWER CONSISTENCY: PASS"
        )
    else:
        print(
            "CACHE VS RAW ANSWER CONSISTENCY: DIFFERENT"
        )


if __name__ == "__main__":
    run()