# Extracted from Kaggle notebook: notebookf960a15c75.ipynb, cell 6
# Packages the question embedding shards together with aligned metadata.
# Kaggle input/output paths are intentionally preserved for now.

from pathlib import Path
import pandas as pd
import torch
import json
import shutil


# ============================================================
# PATHS
# ============================================================

ANNOTATIONS_PATH = Path(
    "/kaggle/input/datasets/rajdeepbiswas1/"
    "satqueryai-mcq-annotations/"
    "training_annotations.parquet"
)

EMBEDDING_DIR = Path(
    "/kaggle/working/mcq_question_embeddings"
)

PACKAGE_DIR = Path(
    "/kaggle/working/mcq_question_cache_package"
)

ARCHIVE_PATH = Path(
    "/kaggle/working/satqueryai_mcq_question_cache.zip"
)


# ============================================================
# CLEAN / CREATE PACKAGE DIRECTORY
# ============================================================

if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)

PACKAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MCQ ANNOTATIONS
# ============================================================

print("Loading annotations...")

df = pd.read_parquet(
    ANNOTATIONS_PATH
)

mcq = df[
    df["type"] == "mcq"
].copy()

mcq["question"] = (
    mcq["input"]
    .astype(str)
)


# ============================================================
# CREATE UNIQUE QUESTION TABLE
# ============================================================

unique_questions = (
    mcq["question"]
    .drop_duplicates()
    .reset_index(drop=True)
)

question_to_index = {
    question: index
    for index, question
    in enumerate(unique_questions)
}


# Map every MCQ row to the unique question embedding
mcq["question_embedding_index"] = (
    mcq["question"]
    .map(question_to_index)
)


# ============================================================
# VERIFY MAPPING
# ============================================================

print("\n" + "=" * 60)
print("MCQ MAPPING")
print("=" * 60)

print(
    "MCQ rows:",
    len(mcq)
)

print(
    "Unique questions:",
    len(unique_questions)
)

print(
    "Missing mappings:",
    mcq["question_embedding_index"]
    .isna()
    .sum()
)

if (
    mcq["question_embedding_index"]
    .isna()
    .any()
):
    raise RuntimeError(
        "Some MCQ rows could not be mapped "
        "to a question embedding."
    )


mcq["question_embedding_index"] = (
    mcq["question_embedding_index"]
    .astype("int64")
)


# ============================================================
# SAVE ROW MAPPING
# ============================================================

mapping_path = (
    PACKAGE_DIR
    / "mcq_row_mapping.parquet"
)

mcq[
    [
        "ID",
        "patch_id",
        "input",
        "output",
        "question_embedding_index",
    ]
].to_parquet(
    mapping_path,
    index=False
)

print(
    "\nSaved:",
    mapping_path
)


# ============================================================
# SAVE UNIQUE QUESTION TABLE
# ============================================================

questions_path = (
    PACKAGE_DIR
    / "unique_questions.parquet"
)

pd.DataFrame(
    {
        "question_embedding_index":
            range(len(unique_questions)),
        "question":
            unique_questions,
    }
).to_parquet(
    questions_path,
    index=False
)

print(
    "Saved:",
    questions_path
)


# ============================================================
# COPY VERIFIED EMBEDDING SHARDS
# ============================================================

destination_embeddings = (
    PACKAGE_DIR
    / "mcq_question_embeddings"
)

destination_embeddings.mkdir(
    exist_ok=True
)

source_files = sorted(
    EMBEDDING_DIR.glob("shard_*.pt")
)

for source in source_files:

    destination = (
        destination_embeddings
        / source.name
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        "Copied:",
        source.name
    )


# ============================================================
# METADATA
# ============================================================

metadata = {
    "model_name":
        "distilbert-base-uncased",
    "hidden_size":
        768,
    "dtype":
        "float32",
    "max_length":
        64,
    "pooling":
        "masked_mean",
    "mcq_rows":
        len(mcq),
    "unique_questions":
        len(unique_questions),
    "embedding_shards":
        len(source_files),
}

metadata_path = (
    PACKAGE_DIR
    / "metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL PACKAGE CONTENTS
# ============================================================

print("\n" + "=" * 60)
print("PACKAGE CONTENTS")
print("=" * 60)

for path in sorted(
    PACKAGE_DIR.rglob("*")
):

    if path.is_file():

        print(
            path.relative_to(
                PACKAGE_DIR
            )
        )


# ============================================================
# CREATE ZIP
# ============================================================

print("\nCreating ZIP archive...")

if ARCHIVE_PATH.exists():
    ARCHIVE_PATH.unlink()

shutil.make_archive(
    str(ARCHIVE_PATH.with_suffix("")),
    "zip",
    PACKAGE_DIR
)

print(
    "\nArchive created:",
    ARCHIVE_PATH
)

print(
    "Archive size:",
    f"{ARCHIVE_PATH.stat().st_size / (1024**2):.2f} MB"
)

print("\n" + "=" * 60)
print("MCQ CACHE PACKAGE COMPLETE")
print("=" * 60)