# Extracted from Kaggle notebook: satqueryai-grounding-text-embeddings-v2.ipynb
# Cells 0-7 combined into one reproducible preprocessing script.
# This is the row-aligned grounding text-cache pipeline used by the project.
# Kaggle input/output paths are intentionally preserved for now.

# ============================================================
# NOTEBOOK CELL 0
# ============================================================

# ============================================================
# CELL 1 — Configuration and environment
# ============================================================

import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

print("=" * 70)
print("SATQUERYAI — GROUNDING TEXT EMBEDDING PIPELINE")
print("=" * 70)

MODEL_NAME = "distilbert-base-uncased"

POINT_QUERY = "locate the land cover instance"

MAX_LENGTH = 128
BATCH_SIZE = 32
SHARD_SIZE = 256

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working/grounding_text_embedding_shards")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")
print(f"Shard size: {SHARD_SIZE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available.")

print("Configuration complete.")

# ============================================================
# NOTEBOOK CELL 1
# ============================================================

# ============================================================
# CELL 2 — Locate and load annotations
# ============================================================

annotation_candidates = list(
    INPUT_ROOT.rglob("training_annotations.parquet")
)

print("=" * 70)
print("SEARCHING FOR ANNOTATIONS")
print("=" * 70)

for path in annotation_candidates:
    print(path)

if not annotation_candidates:
    raise FileNotFoundError(
        "training_annotations.parquet was not found under /kaggle/input"
    )

ANNOTATION_FILE = annotation_candidates[0]

print(f"\nUsing: {ANNOTATION_FILE}")

df = pd.read_parquet(
    ANNOTATION_FILE
)

grounding = df[
    df["type"] == "bounding box"
].copy()

grounding = grounding.reset_index(
    drop=False
)

grounding.rename(
    columns={"index": "source_row_index"},
    inplace=True,
)

print("\n" + "=" * 70)
print("GROUNDING DATASET")
print("=" * 70)

print(f"Grounding rows: {len(grounding):,}")
print(
    f"Unique patches: "
    f"{grounding['patch_id'].nunique():,}"
)
print(
    f"Point rows: "
    f"{(grounding['category'] == 'point').sum():,}"
)
print(
    f"Reference rows: "
    f"{(grounding['category'] == 'reference').sum():,}"
)

# ============================================================
# NOTEBOOK CELL 2
# ============================================================

# ============================================================
# CELL 3 — Extract semantic grounding queries
# ============================================================

REF_PATTERN = re.compile(
    r"<ref>(.*?)</ref>",
    re.IGNORECASE,
)

POINT_PATTERN = re.compile(
    r"<point>\(\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)"
    r"\s*\)</point>",
    re.IGNORECASE,
)

BOX_PATTERN = re.compile(
    r"\[\s*"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)"
    r"\s*\]"
)


def extract_reference(text):
    match = REF_PATTERN.search(str(text))

    if match is None:
        return None

    return match.group(1).strip()


def parse_point(text):
    match = POINT_PATTERN.search(str(text))

    if match is None:
        return (0.0, 0.0)

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


def parse_box(text):
    match = BOX_PATTERN.fullmatch(
        str(text).strip()
    )

    if match is None:
        raise ValueError(
            f"Invalid bounding box: {text}"
        )

    return tuple(
        float(value)
        for value in match.groups()
    )


def build_semantic_query(row):
    if row["category"] == "point":
        return POINT_QUERY

    if row["category"] == "reference":
        reference = extract_reference(
            row["input"]
        )

        if reference is None:
            raise ValueError(
                f"Missing reference: {row['input']}"
            )

        return reference

    raise ValueError(
        f"Unexpected category: {row['category']}"
    )


grounding["semantic_query"] = grounding.apply(
    build_semantic_query,
    axis=1,
)

queries = (
    grounding["semantic_query"]
    .drop_duplicates()
    .tolist()
)

query_to_index = {
    query: index
    for index, query in enumerate(queries)
}

grounding["embedding_index"] = (
    grounding["semantic_query"]
    .map(query_to_index)
    .astype("int64")
)

print("=" * 70)
print("QUERY ANALYSIS")
print("=" * 70)

print(
    f"Unique semantic queries: "
    f"{len(queries):,}"
)

print("Expected: 244")

assert len(queries) == 244

print("\nFirst queries:")

for i, query in enumerate(
    queries[:20]
):
    print(f"{i:03d} | {query}")

# ============================================================
# NOTEBOOK CELL 3
# ============================================================

# ============================================================
# CELL 4 — Load DistilBERT
# ============================================================

print("=" * 70)
print("LOADING DISTILBERT")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

text_model = AutoModel.from_pretrained(
    MODEL_NAME
)

text_model.to(DEVICE)
text_model.eval()

for parameter in text_model.parameters():
    parameter.requires_grad = False

print("Model loaded successfully.")

# ============================================================
# NOTEBOOK CELL 4
# ============================================================

# ============================================================
# CELL 5 — Small-batch test
# ============================================================

print("=" * 70)
print("SMALL-BATCH EMBEDDING TEST")
print("=" * 70)

test_queries = queries[:8]

encoded = tokenizer(
    test_queries,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

encoded = {
    key: value.to(DEVICE)
    for key, value in encoded.items()
}

with torch.no_grad():

    hidden = text_model(
        **encoded
    ).last_hidden_state

    mask = encoded[
        "attention_mask"
    ].unsqueeze(-1)

    pooled = (
        hidden * mask
    ).sum(dim=1) / mask.sum(
        dim=1
    ).clamp(min=1)

print(
    f"Input examples: {len(test_queries)}"
)

print(
    f"Test embedding shape: "
    f"{tuple(pooled.shape)}"
)

assert pooled.shape == (
    len(test_queries),
    768,
)

assert torch.isfinite(
    pooled
).all()

print("Small-batch test PASSED.")

# ============================================================
# NOTEBOOK CELL 5
# ============================================================

# ============================================================
# CELL 6 — Generate all unique text embeddings
# ============================================================

print("=" * 70)
print("GENERATING ALL UNIQUE TEXT EMBEDDINGS")
print("=" * 70)

unique_embeddings = []

with torch.no_grad():

    for start in range(
        0,
        len(queries),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(queries),
        )

        batch_queries = queries[
            start:end
        ]

        encoded = tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        hidden = text_model(
            **encoded
        ).last_hidden_state

        mask = encoded[
            "attention_mask"
        ].unsqueeze(-1)

        pooled = (
            hidden * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)

        unique_embeddings.append(
            pooled.cpu()
        )

        print(
            f"Processed "
            f"{end:,}/{len(queries):,}"
        )

unique_embeddings = torch.cat(
    unique_embeddings,
    dim=0,
)

print(
    f"\nUnique embedding shape: "
    f"{tuple(unique_embeddings.shape)}"
)

assert unique_embeddings.shape == (
    244,
    768,
)

assert torch.isfinite(
    unique_embeddings
).all()

unique_cache = {
    "queries": queries,
    "embeddings": unique_embeddings,
}

unique_cache_path = (
    Path("/kaggle/working")
    / "grounding_unique_text_embeddings.pt"
)

torch.save(
    unique_cache,
    unique_cache_path,
)

print(
    f"Saved unique cache: "
    f"{unique_cache_path}"
)

# ============================================================
# NOTEBOOK CELL 6
# ============================================================

# ============================================================
# CELL 7 — Create row-aligned shards
# ============================================================

print("=" * 70)
print("CREATING ROW-ALIGNED SHARDS")
print("=" * 70)

# Parse point coordinates.
points = grounding[
    "input"
].apply(parse_point)

grounding["point_x"] = points.apply(
    lambda value: value[0]
)

grounding["point_y"] = points.apply(
    lambda value: value[1]
)

# Parse target boxes.
boxes = grounding[
    "output"
].apply(parse_box)

grounding["x1"] = boxes.apply(
    lambda value: value[0]
)

grounding["y1"] = boxes.apply(
    lambda value: value[1]
)

grounding["x2"] = boxes.apply(
    lambda value: value[2]
)

grounding["y2"] = boxes.apply(
    lambda value: value[3]
)

# Mode:
# 0 = reference
# 1 = point
grounding["mode"] = (
    grounding["category"] == "point"
).astype("int64")

num_rows = len(grounding)

num_shards = (
    num_rows + SHARD_SIZE - 1
) // SHARD_SIZE

print(
    f"Rows: {num_rows:,}"
)

print(
    f"Shards: {num_shards:,}"
)

for shard_id, start in enumerate(
    range(
        0,
        num_rows,
        SHARD_SIZE,
    )
):

    end = min(
        start + SHARD_SIZE,
        num_rows,
    )

    chunk = grounding.iloc[
        start:end
    ]

    embedding_indices = torch.tensor(
        chunk[
            "embedding_index"
        ].tolist(),
        dtype=torch.long,
    )

    text_embeddings = (
        unique_embeddings[
            embedding_indices
        ]
        .contiguous()
    )

    points_tensor = torch.tensor(
        chunk[
            ["point_x", "point_y"]
        ].to_numpy(),
        dtype=torch.float32,
    )

    boxes_tensor = torch.tensor(
        chunk[
            ["x1", "y1", "x2", "y2"]
        ].to_numpy(),
        dtype=torch.float32,
    )

    modes_tensor = torch.tensor(
        chunk[
            "mode"
        ].to_numpy(),
        dtype=torch.long,
    )

    source_rows_tensor = torch.tensor(
        chunk[
            "source_row_index"
        ].tolist(),
        dtype=torch.long,
    )

    shard = {
        "source_row_index": source_rows_tensor,
        "patch_ids": (
            chunk["patch_id"]
            .astype(str)
            .tolist()
        ),
        "text_embeddings": text_embeddings,
        "points": points_tensor,
        "target_boxes": boxes_tensor,
        "modes": modes_tensor,
        "categories": (
            chunk["category"]
            .astype(str)
            .tolist()
        ),
    }

    expected_rows = end - start

    assert text_embeddings.shape == (
        expected_rows,
        768,
    )

    assert points_tensor.shape == (
        expected_rows,
        2,
    )

    assert boxes_tensor.shape == (
        expected_rows,
        4,
    )

    assert modes_tensor.shape == (
        expected_rows,
    )

    assert torch.isfinite(
        text_embeddings
    ).all()

    output_path = (
        OUTPUT_DIR
        / f"grounding_text_shard_{shard_id:03d}.pt"
    )

    torch.save(
        shard,
        output_path,
    )

    print(
        f"Shard {shard_id:03d}: "
        f"{start:,}-{end - 1:,} "
        f"({expected_rows} rows)"
    )

print("\nSharding complete.")

# ============================================================
# NOTEBOOK CELL 7
# ============================================================

# ============================================================
# CELL 8 — Verify every shard
# ============================================================

print("=" * 70)
print("FINAL SHARD VERIFICATION")
print("=" * 70)

shard_files = sorted(
    OUTPUT_DIR.glob(
        "grounding_text_shard_*.pt"
    )
)

expected_shards = (
    num_rows + SHARD_SIZE - 1
) // SHARD_SIZE

print(
    f"Shard files found: "
    f"{len(shard_files):,}"
)

print(
    f"Expected shards: "
    f"{expected_shards:,}"
)

assert len(shard_files) == (
    expected_shards
)

total_rows = 0

seen_rows = set()

for path in shard_files:

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    n = len(
        data["patch_ids"]
    )

    assert data[
        "text_embeddings"
    ].shape == (
        n,
        768,
    )

    assert data[
        "points"
    ].shape == (
        n,
        2,
    )

    assert data[
        "target_boxes"
    ].shape == (
        n,
        4,
    )

    assert data[
        "modes"
    ].shape == (
        n,
    )

    assert torch.isfinite(
        data["text_embeddings"]
    ).all()

    total_rows += n

    for row_id in data[
        "source_row_index"
    ].tolist():

        if row_id in seen_rows:
            raise RuntimeError(
                f"Duplicate source row: "
                f"{row_id}"
            )

        seen_rows.add(row_id)

print(
    f"Total rows: {total_rows:,}"
)

print(
    f"Unique source rows: "
    f"{len(seen_rows):,}"
)

assert total_rows == num_rows

assert len(seen_rows) == num_rows

print(
    "ALL SHARDS VERIFIED SUCCESSFULLY."
)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)

print(
    f"Unique semantic queries: "
    f"{len(queries):,}"
)

print(
    f"Grounding examples: "
    f"{num_rows:,}"
)

print(
    f"Shards: "
    f"{len(shard_files):,}"
)

print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)