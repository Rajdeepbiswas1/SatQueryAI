# Extracted from Kaggle notebook: notebookf960a15c75.ipynb, cell 4
# Preserved from the notebook without changing the embedding-generation logic.
# Kaggle input/output paths are intentionally preserved for now.

import os
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 128
MAX_LENGTH = 64
SHARD_SIZE = 4096

OUTPUT_DIR = Path(
    "/kaggle/working/mcq_question_embeddings"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

annotations_path = Path(
    "/kaggle/input/datasets/rajdeepbiswas1/"
    "satqueryai-mcq-annotations/"
    "training_annotations.parquet"
)

df = pd.read_parquet(
    annotations_path
)

mcq = df[
    df["type"] == "mcq"
].copy()

# Use unique question strings.
unique_questions = (
    mcq["input"]
    .astype(str)
    .drop_duplicates()
    .reset_index(drop=True)
)

print("\nMCQ rows:", len(mcq))
print(
    "Unique questions:",
    len(unique_questions)
)


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading DistilBERT...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# MASKED MEAN POOLING
# ============================================================

def masked_mean_pool(
    hidden_states,
    attention_mask,
):

    mask = attention_mask.unsqueeze(
        -1
    ).to(hidden_states.dtype)

    summed = (
        hidden_states * mask
    ).sum(dim=1)

    counts = mask.sum(dim=1).clamp(
        min=1e-9
    )

    return summed / counts


# ============================================================
# EMBEDDING GENERATION
# ============================================================

total = len(unique_questions)
num_shards = math.ceil(
    total / SHARD_SIZE
)

print("\n" + "=" * 60)
print("MCQ QUESTION EMBEDDING CACHE")
print("=" * 60)

print("Questions:", total)
print("Batch size:", BATCH_SIZE)
print("Max length:", MAX_LENGTH)
print("Shard size:", SHARD_SIZE)
print("Shards:", num_shards)
print("Output:", OUTPUT_DIR)
print("=" * 60)


with torch.inference_mode():

    for shard_index in range(
        num_shards
    ):

        start = (
            shard_index
            * SHARD_SIZE
        )

        end = min(
            start + SHARD_SIZE,
            total
        )

        output_file = (
            OUTPUT_DIR
            / f"shard_{shard_index:03d}.pt"
        )

        # Resume support
        if output_file.exists():

            print(
                f"[{shard_index + 1}/"
                f"{num_shards}] "
                f"Already exists → "
                f"{output_file.name}"
            )

            continue

        texts = unique_questions[
            start:end
        ].tolist()

        embeddings = []

        for batch_start in range(
            0,
            len(texts),
            BATCH_SIZE
        ):

            batch_texts = texts[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            pooled = masked_mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings.append(
                pooled.cpu().float()
            )

        embeddings = torch.cat(
            embeddings,
            dim=0
        )

        shard_data = {
            "question_indices": torch.arange(
                start,
                end,
                dtype=torch.long
            ),
            "questions": texts,
            "question_embeddings": embeddings,

            "model_name": MODEL_NAME,
            "hidden_size": 768,
            "dtype": "float32",
            "max_length": MAX_LENGTH,
            "pooling": "masked_mean",
            "start_index": start,
            "end_index": end,
        }

        torch.save(
            shard_data,
            output_file
        )

        print(
            f"[{shard_index + 1}/"
            f"{num_shards}] "
            f"Saved {len(texts)} rows → "
            f"{output_file.name}"
        )


print("\n" + "=" * 60)
print("MCQ QUESTION EMBEDDING CACHE COMPLETE")
print("=" * 60)

print(
    "Questions:",
    total
)

print(
    "Shards:",
    num_shards
)

print(
    "Output:",
    OUTPUT_DIR
)