from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# SATQUERYAI — TEMPORAL TEST QUESTION EMBEDDING GENERATOR
# =============================================================================
#
# Purpose:
#   Generate the held-out temporal Test question embeddings required for
#   evaluating the preserved V3 checkpoint.
#
# Established temporal embedding configuration:
#   model      = distilbert-base-uncased
#   max_length = 64
#   pooling    = masked_mean
#   dimension  = 768
#   dtype      = float32
#
# Input:
#   D:\SatQueryAI\data\processed\temporal_split\test_manifest.csv
#
# Output:
#   D:\SatQueryAI\data\processed\temporal_test_question_embeddings
#
# Important:
#   - Does NOT modify the Test manifest.
#   - Does NOT modify temporal DOFA features.
#   - Does NOT train anything.
#   - Does NOT modify the V3 checkpoint.
#   - Question IDs are preserved exactly from test_manifest.csv.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "test_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_test_question_embeddings"
)


# =============================================================================
# FIXED CONFIGURATION
# =============================================================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
POOLING = "masked_mean"
EMBEDDING_DIM = 768
DTYPE_NAME = "float32"

BATCH_SIZE = 32
SHARD_SIZE = 4000


# =============================================================================
# HELPERS
# =============================================================================

def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} not found: {path}"
        )


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings using the attention mask.

    Input:
        last_hidden_state [B, L, H]
        attention_mask    [B, L]

    Output:
        embeddings        [B, H]
    """

    mask = attention_mask.unsqueeze(-1).to(
        dtype=last_hidden_state.dtype
    )

    masked = last_hidden_state * mask

    summed = masked.sum(dim=1)

    counts = mask.sum(dim=1).clamp_min(1.0)

    return summed / counts


def save_shard(
    output_path: Path,
    question_ids: list[int],
    embeddings: torch.Tensor,
) -> None:

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embeddings, got {tuple(embeddings.shape)}"
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"expected {EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}"
        )

    if len(question_ids) != embeddings.shape[0]:
        raise RuntimeError(
            "Question ID count does not match embedding rows."
        )

    embeddings = (
        embeddings
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    if not torch.isfinite(embeddings).all():
        raise RuntimeError(
            f"Non-finite embedding values detected in {output_path}"
        )

    payload = {
        "question_ids": torch.tensor(
            question_ids,
            dtype=torch.long,
        ),
        "embeddings": embeddings,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "pooling": POOLING,
        "dtype": DTYPE_NAME,
    }

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    print("=" * 80)
    print("SATQUERYAI — TEMPORAL TEST QUESTION EMBEDDING GENERATOR")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------

    require_path(
        TEST_MANIFEST,
        "Test manifest",
    )

    print(
        "Test manifest:",
        TEST_MANIFEST,
    )

    # -------------------------------------------------------------------------
    # Load Test manifest
    # -------------------------------------------------------------------------

    print()
    print("Loading Test manifest...")

    df = pd.read_csv(
        TEST_MANIFEST
    )

    required_columns = {
        "question_id",
        "question",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Test manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise RuntimeError(
            "Test manifest is empty."
        )

    question_ids = (
        df["question_id"]
        .astype(int)
        .tolist()
    )

    questions = (
        df["question"]
        .astype(str)
        .tolist()
    )

    # -------------------------------------------------------------------------
    # Verify exact Test indexing
    # -------------------------------------------------------------------------

    expected_ids = list(
        range(len(df))
    )

    if question_ids != expected_ids:
        raise RuntimeError(
            "Test question IDs are not exactly continuous "
            "from 0 to N-1 in manifest order."
        )

    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError(
            "Duplicate Test question IDs detected."
        )

    if len(questions) != 39686:
        raise RuntimeError(
            f"Expected 39,686 Test questions, "
            f"found {len(questions)}."
        )

    print(
        f"Test rows:            {len(df):,}"
    )

    print(
        f"Question ID range:    "
        f"{question_ids[0]} -> {question_ids[-1]}"
    )

    print(
        f"Unique questions:     {len(set(questions)):,}"
    )

    print(
        "Question ID ordering: PASS"
    )

    # -------------------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Output directory:     ",
        OUTPUT_ROOT,
    )

    # -------------------------------------------------------------------------
    # Load tokenizer/model
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING DISTILBERT")
    print("=" * 80)

    print(
        "Model:      ",
        MODEL_NAME,
    )

    print(
        "Max length: ",
        MAX_LENGTH,
    )

    print(
        "Pooling:    ",
        POOLING,
    )

    print(
        "Embedding:  ",
        EMBEDDING_DIM,
    )

    print(
        "Dtype:      ",
        DTYPE_NAME,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:     ",
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
    )

    model = model.to(device)
    model.eval()

    actual_hidden_size = (
        int(model.config.hidden_size)
    )

    print(
        "Hidden size:",
        actual_hidden_size,
    )

    if actual_hidden_size != EMBEDDING_DIM:
        raise RuntimeError(
            "Unexpected DistilBERT hidden size: "
            f"expected {EMBEDDING_DIM}, "
            f"got {actual_hidden_size}"
        )

    print(
        "Model configuration: PASS"
    )

    # -------------------------------------------------------------------------
    # Existing output safety check
    # -------------------------------------------------------------------------

    existing_shards = sorted(
        OUTPUT_ROOT.glob("shard_*.pt")
    )

    if existing_shards:
        print()
        print(
            "Existing Test embedding shards detected:",
            len(existing_shards),
        )

        raise RuntimeError(
            "Output directory already contains Test embedding shards. "
            "Remove the existing shards only after verifying they are "
            "not the intended final Test cache."
        )

    # -------------------------------------------------------------------------
    # Generate embeddings
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATING TEST EMBEDDINGS")
    print("=" * 80)

    all_embedding_parts: list[torch.Tensor] = []

    start_time = time.time()

    total_batches = (
        len(questions)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    batch_counter = 0

    with torch.no_grad():

        for start in range(
            0,
            len(questions),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(questions),
            )

            batch_questions = questions[
                start:end
            ]

            encoded = tokenizer(
                batch_questions,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            embeddings = masked_mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings = (
                embeddings
                .detach()
                .cpu()
                .float()
                .contiguous()
            )

            if tuple(embeddings.shape) != (
                end - start,
                EMBEDDING_DIM,
            ):
                raise RuntimeError(
                    "Unexpected batch embedding shape: "
                    f"{tuple(embeddings.shape)}"
                )

            if not torch.isfinite(
                embeddings
            ).all():
                raise RuntimeError(
                    f"Non-finite embeddings detected "
                    f"for rows {start}:{end}"
                )

            all_embedding_parts.append(
                embeddings
            )

            batch_counter += 1

            if (
                batch_counter == 1
                or batch_counter % 25 == 0
                or end == len(questions)
            ):
                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"Batch {batch_counter:4d}/{total_batches:4d} | "
                    f"rows {end:6d}/{len(questions):6d} | "
                    f"elapsed {elapsed:8.1f}s"
                )

    # -------------------------------------------------------------------------
    # Combine embeddings
    # -------------------------------------------------------------------------

    embeddings_all = torch.cat(
        all_embedding_parts,
        dim=0,
    )

    if tuple(embeddings_all.shape) != (
        len(questions),
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Final embedding matrix shape mismatch: "
            f"{tuple(embeddings_all.shape)}"
        )

    if embeddings_all.dtype != torch.float32:
        embeddings_all = embeddings_all.float()

    if not torch.isfinite(
        embeddings_all
    ).all():
        raise RuntimeError(
            "Non-finite values detected in final embedding matrix."
        )

    print()
    print(
        "Final embedding matrix:",
        tuple(embeddings_all.shape),
    )

    print(
        "Final embedding dtype:",
        embeddings_all.dtype,
    )

    print(
        "Final embedding finiteness: PASS"
    )

    # -------------------------------------------------------------------------
    # Save shards
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVING TEST EMBEDDING SHARDS")
    print("=" * 80)

    shard_count = 0

    for start in range(
        0,
        len(question_ids),
        SHARD_SIZE,
    ):

        end = min(
            start + SHARD_SIZE,
            len(question_ids),
        )

        shard_ids = question_ids[
            start:end
        ]

        shard_embeddings = embeddings_all[
            start:end
        ]

        shard_path = (
            OUTPUT_ROOT
            / f"shard_{shard_count:03d}.pt"
        )

        save_shard(
            shard_path,
            shard_ids,
            shard_embeddings,
        )

        print(
            f"Shard {shard_count:02d}: "
            f"rows={start:6d}:{end:6d} "
            f"shape={tuple(shard_embeddings.shape)} "
            f"-> {shard_path.name}"
        )

        shard_count += 1

    # -------------------------------------------------------------------------
    # Reload and verify every shard
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VERIFYING SAVED TEST EMBEDDING CACHE")
    print("=" * 80)

    saved_shards = sorted(
        OUTPUT_ROOT.glob("shard_*.pt")
    )

    if len(saved_shards) != shard_count:
        raise RuntimeError(
            f"Expected {shard_count} saved shards, "
            f"found {len(saved_shards)}"
        )

    verified_ids: list[int] = []
    verified_rows = 0

    for shard_path in saved_shards:

        payload = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        required_keys = {
            "question_ids",
            "embeddings",
            "model_name",
            "max_length",
            "pooling",
            "dtype",
        }

        missing_keys = (
            required_keys
            - set(payload.keys())
        )

        if missing_keys:
            raise RuntimeError(
                f"{shard_path.name} missing keys: "
                f"{sorted(missing_keys)}"
            )

        shard_ids = [
            int(x)
            for x in payload["question_ids"].tolist()
        ]

        shard_embeddings = payload[
            "embeddings"
        ]

        if shard_embeddings.ndim != 2:
            raise RuntimeError(
                f"{shard_path.name}: invalid embedding rank."
            )

        if tuple(shard_embeddings.shape[1:]) != (
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                f"{shard_path.name}: invalid shape "
                f"{tuple(shard_embeddings.shape)}"
            )

        if len(shard_ids) != shard_embeddings.shape[0]:
            raise RuntimeError(
                f"{shard_path.name}: ID/embedding count mismatch."
            )

        if payload["model_name"] != MODEL_NAME:
            raise RuntimeError(
                f"{shard_path.name}: model metadata mismatch."
            )

        if int(payload["max_length"]) != MAX_LENGTH:
            raise RuntimeError(
                f"{shard_path.name}: max_length metadata mismatch."
            )

        if payload["pooling"] != POOLING:
            raise RuntimeError(
                f"{shard_path.name}: pooling metadata mismatch."
            )

        if payload["dtype"] != DTYPE_NAME:
            raise RuntimeError(
                f"{shard_path.name}: dtype metadata mismatch."
            )

        if shard_embeddings.dtype != torch.float32:
            raise RuntimeError(
                f"{shard_path.name}: embeddings are not float32."
            )

        if not torch.isfinite(
            shard_embeddings
        ).all():
            raise RuntimeError(
                f"{shard_path.name}: non-finite values detected."
            )

        verified_ids.extend(
            shard_ids
        )

        verified_rows += (
            shard_embeddings.shape[0]
        )

    if verified_rows != len(question_ids):
        raise RuntimeError(
            f"Verified {verified_rows:,} rows, "
            f"expected {len(question_ids):,}"
        )

    if verified_ids != question_ids:
        raise RuntimeError(
            "Saved Test question IDs do not exactly match "
            "the original Test manifest order."
        )

    print(
        "Shard count:",
        len(saved_shards),
    )

    print(
        "Verified rows:",
        f"{verified_rows:,}",
    )

    print(
        "Question ID/order preservation: PASS"
    )

    print(
        "Metadata verification: PASS"
    )

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * 80)
    print("SATQUERYAI TEMPORAL TEST QUESTION EMBEDDING GENERATION: PASS")
    print("=" * 80)

    print(
        "Test questions :",
        f"{len(questions):,}",
    )

    print(
        "Embedding shape:",
        tuple(embeddings_all.shape),
    )

    print(
        "Embedding dtype:",
        embeddings_all.dtype,
    )

    print(
        "Shards         :",
        len(saved_shards),
    )

    print(
        "Elapsed seconds:",
        f"{total_time:.2f}",
    )

    print(
        "Output         :",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()