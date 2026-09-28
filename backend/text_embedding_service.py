from pathlib import Path
from typing import Any

import pandas as pd
import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
)


# -------------------------------------------------------------------
# Canonical SatQueryAI text embedding configuration.
#
# These values are taken directly from the existing question
# embedding pipeline.
# -------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(r"D:\SatQueryAI")

BINARY_ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

BINARY_EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)


class TextEmbeddingError(Exception):
    """Raised when query text embedding fails."""


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Canonical masked-mean pooling used by SatQueryAI.

    Input:
        hidden_states  [B, sequence_length, 768]
        attention_mask [B, sequence_length]

    Output:
        [B, 768]
    """

    mask = attention_mask.unsqueeze(-1).to(
        hidden_states.dtype
    )

    masked_hidden = hidden_states * mask

    summed = masked_hidden.sum(
        dim=1
    )

    lengths = mask.sum(
        dim=1
    ).clamp(
        min=1
    )

    return summed / lengths


class SatQueryAITextEmbeddingService:
    """
    Application-level wrapper around the frozen DistilBERT
    question encoder.

    Input:
        natural-language query

    Output:
        [1,768] float32 embedding
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = torch.device(
            device
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
            .to(self.device)
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def embed(
        self,
        query: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:

        if not isinstance(
            query,
            str,
        ):
            raise TextEmbeddingError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise TextEmbeddingError(
                "Query cannot be empty."
            )

        encoded = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        output = self.model(
            input_ids=encoded[
                "input_ids"
            ].to(self.device),
            attention_mask=encoded[
                "attention_mask"
            ].to(self.device),
        )

        embedding = masked_mean_pool(
            output.last_hidden_state,
            encoded[
                "attention_mask"
            ].to(self.device),
        )

        embedding = embedding.float().cpu()

        expected_shape = (
            1,
            768,
        )

        if tuple(embedding.shape) != expected_shape:
            raise TextEmbeddingError(
                "Unexpected embedding shape: "
                f"{tuple(embedding.shape)}"
            )

        if not torch.isfinite(
            embedding
        ).all().item():
            raise TextEmbeddingError(
                "Text embedding contains NaN or Inf."
            )

        metadata = {
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "pooling": "masked_mean",
            "shape": list(
                embedding.shape
            ),
            "dtype": str(
                embedding.dtype
            ),
            "device": str(
                self.device
            ),
        }

        return (
            embedding,
            metadata,
        )


def load_first_cached_binary_embedding():
    """
    Load the first cached binary question embedding and its
    corresponding question from the same dataframe ordering used
    by the original cache-generation script.
    """

    if not BINARY_ANNOTATIONS_PATH.is_file():
        raise FileNotFoundError(
            "Missing binary VQA annotations: "
            f"{BINARY_ANNOTATIONS_PATH}"
        )

    shard_path = (
        BINARY_EMBEDDING_DIR
        / "shard_000.pt"
    )

    if not shard_path.is_file():
        raise FileNotFoundError(
            "Missing binary question embedding shard: "
            f"{shard_path}"
        )

    df = pd.read_parquet(
        BINARY_ANNOTATIONS_PATH
    )

    df = df[
        df["type"] == "binary"
    ].copy()

    df = df.reset_index(
        drop=True
    )

    cached = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    question = str(
        df.iloc[0]["input"]
    )

    cached_id = int(
        cached["ids"][0]
    )

    cached_embedding = (
        cached[
            "question_embeddings"
        ][0:1]
        .float()
        .cpu()
    )

    return (
        question,
        cached_id,
        cached_embedding,
    )


if __name__ == "__main__":

    print(
        "SATQUERYAI — TEXT EMBEDDING SERVICE TEST"
    )
    print("=" * 80)
    print()

    (
        question,
        cached_id,
        cached_embedding,
    ) = load_first_cached_binary_embedding()

    print(
        f"Question ID: {cached_id}"
    )

    print(
        f"Question: {question}"
    )

    print()
    print(
        "Loading canonical DistilBERT encoder..."
    )

    service = (
        SatQueryAITextEmbeddingService(
            device="cpu"
        )
    )

    embedding, metadata = (
        service.embed(
            question
        )
    )

    max_difference = (
        embedding - cached_embedding
    ).abs().max().item()

    print()
    print(
        "OUTPUT"
    )
    print(
        f"  shape: {metadata['shape']}"
    )
    print(
        f"  dtype: {metadata['dtype']}"
    )
    print(
        f"  model: {metadata['model_name']}"
    )
    print(
        f"  max length: {metadata['max_length']}"
    )
    print(
        f"  pooling: {metadata['pooling']}"
    )
    print(
        f"  device: {metadata['device']}"
    )

    print()
    print(
        "CACHE REGRESSION TEST"
    )

    print(
        f"  max absolute difference: "
        f"{max_difference}"
    )

    cosine_similarity = (
        torch.nn.functional.cosine_similarity(
            embedding,
            cached_embedding,
            dim=1,
        ).item()
    )

    print(
        f"  cosine similarity: "
        f"{cosine_similarity}"
    )

    if max_difference > 1e-6:
        raise RuntimeError(
            "Generated embedding differs too much from "
            "the existing cached embedding: "
            f"max_difference={max_difference}"
        )

    if cosine_similarity < 0.999999:
        raise RuntimeError(
            "Generated embedding has unexpectedly low cosine "
            f"similarity with the cached embedding: "
            f"{cosine_similarity}"
        )

    print()
    print(
        "TEXT EMBEDDING SERVICE TEST: PASS"
    )