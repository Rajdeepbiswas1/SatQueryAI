from pathlib import Path
import json
import sys

import pandas as pd
import torch


# ============================================================
# SATQUERYAI — KAGGLE BUNDLE VALIDATOR
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT
    / "training"
    / "single_modality"
    / "kaggle_bundle_manifest"
)

OUT_MANIFEST = (
    OUT_DIR
    / "manifest.json"
)


# ============================================================
# REQUIRED ASSETS
# ============================================================

REQUIRED_FILES = [
    ROOT / "data" / "processed" / "binary_vqa_split.parquet",
]

REQUIRED_DIRS = [
    ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features",

    ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings",

    ROOT
    / "data"
    / "processed"
    / "mcq_question_embeddings",

    ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings",

    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split",
]


# ============================================================
# HELPERS
# ============================================================

def size_bytes(path: Path) -> int:

    if path.is_file():
        return path.stat().st_size

    total = 0

    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size

    return total


def human_size(value: int) -> str:

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    size = float(value)

    for unit in units:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def load_torch(path: Path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)
print("SATQUERYAI — KAGGLE TRAINING BUNDLE VALIDATOR")
print("=" * 72)

print()
print("Project root:")
print(ROOT)


# ============================================================
# PATH CHECK
# ============================================================

print()
print("=" * 72)
print("1. PATH CHECK")
print("=" * 72)

failures = []


for path in REQUIRED_FILES:

    if path.exists():

        print(
            f"PASS  FILE  {path}"
        )

    else:

        print(
            f"FAIL  FILE  {path}"
        )

        failures.append(
            f"Missing file: {path}"
        )


for path in REQUIRED_DIRS:

    if path.exists() and path.is_dir():

        print(
            f"PASS  DIR   {path}"
        )

    else:

        print(
            f"FAIL  DIR   {path}"
        )

        failures.append(
            f"Missing directory: {path}"
        )


# ============================================================
# SIZE INVENTORY
# ============================================================

print()
print("=" * 72)
print("2. SIZE INVENTORY")
print("=" * 72)

asset_sizes = {}


for path in REQUIRED_FILES + REQUIRED_DIRS:

    if path.exists():

        value = size_bytes(path)

        asset_sizes[
            str(path.relative_to(ROOT))
        ] = value

        print(
            f"{human_size(value):>12}  "
            f"{path.relative_to(ROOT)}"
        )


total_bytes = sum(
    asset_sizes.values()
)

print()
print(
    "Total:",
    human_size(total_bytes)
)


# ============================================================
# DOFA CACHE
# ============================================================

print()
print("=" * 72)
print("3. DOFA CACHE")
print("=" * 72)

dofa_dir = (
    ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

dofa_files = sorted(
    dofa_dir.glob("shard_*.pt")
)

print(
    "Shard count:",
    len(dofa_files)
)

if len(dofa_files) != 40:

    failures.append(
        f"Expected 40 DOFA shards, found "
        f"{len(dofa_files)}"
    )


dofa_sample = None

if dofa_files:

    dofa_sample = load_torch(
        dofa_files[0]
    )

    print(
        "First shard:",
        dofa_files[0].name
    )

    print(
        "Keys:",
        list(dofa_sample.keys())
    )

    for modality in [
        "s1",
        "s2",
    ]:

        if modality not in dofa_sample:

            failures.append(
                f"DOFA shard missing key: {modality}"
            )

            print(
                f"FAIL  missing {modality}"
            )

            continue


        tensor = dofa_sample[
            modality
        ]

        print(
            f"{modality.upper():3s}: "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )


        if tensor.ndim != 3:

            failures.append(
                f"{modality} expected 3D tensor"
            )

        elif tensor.shape[-2:] != (
            196,
            768,
        ):

            failures.append(
                f"{modality} expected "
                f"[N,196,768], got "
                f"{tuple(tensor.shape)}"
            )


# ============================================================
# BINARY VQA
# ============================================================

print()
print("=" * 72)
print("4. BINARY VQA")
print("=" * 72)

binary_path = (
    ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

binary_df = pd.read_parquet(
    binary_path
)

print(
    "Rows:",
    len(binary_df)
)

print(
    "Columns:",
    binary_df.columns.tolist()
)

required_binary_columns = {
    "ID",
    "patch_id",
    "output",
    "category",
    "model_split",
}

missing_binary = (
    required_binary_columns
    - set(binary_df.columns)
)

if missing_binary:

    failures.append(
        "Binary VQA missing columns: "
        + str(
            sorted(missing_binary)
        )
    )

else:

    print(
        "Required columns: PASS"
    )


print(
    "Model split counts:",
    binary_df[
        "model_split"
    ].value_counts().to_dict()
)

print(
    "Answer counts:",
    binary_df[
        "output"
    ]
    .astype(str)
    .str.lower()
    .value_counts()
    .to_dict()
)


# ============================================================
# QUESTION EMBEDDINGS
# ============================================================

print()
print("=" * 72)
print("5. BINARY QUESTION EMBEDDINGS")
print("=" * 72)

binary_question_dir = (
    ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

binary_question_files = sorted(
    binary_question_dir.glob(
        "shard_*.pt"
    )
)

print(
    "Shard count:",
    len(binary_question_files)
)

if len(binary_question_files) != 20:

    failures.append(
        "Expected 20 binary question "
        f"shards, found {len(binary_question_files)}"
    )


if binary_question_files:

    question_sample = load_torch(
        binary_question_files[0]
    )

    print(
        "Keys:",
        list(
            question_sample.keys()
        )
    )

    required_question_keys = {
        "ids",
        "question_embeddings",
    }

    missing = (
        required_question_keys
        - set(
            question_sample.keys()
        )
    )

    if missing:

        failures.append(
            "Binary question cache missing "
            f"keys: {sorted(missing)}"
        )

    else:

        embeddings = (
            question_sample[
                "question_embeddings"
            ]
        )

        print(
            "Embedding shape:",
            tuple(
                embeddings.shape
            )
        )

        print(
            "Embedding dtype:",
            embeddings.dtype
        )


# ============================================================
# MCQ QUESTION CACHE
# ============================================================

print()
print("=" * 72)
print("6. MCQ QUESTION CACHE")
print("=" * 72)

mcq_question_dir = (
    ROOT
    / "data"
    / "processed"
    / "mcq_question_embeddings"
)

mcq_question_files = sorted(
    mcq_question_dir.glob(
        "shard_*.pt"
    )
)

print(
    "Shard count:",
    len(
        mcq_question_files
    )
)

if len(mcq_question_files) != 15:

    failures.append(
        "Expected 15 MCQ question shards, "
        f"found {len(mcq_question_files)}"
    )


mcq_map = (
    mcq_question_dir
    / "mcq_row_mapping.parquet"
)

mcq_unique = (
    mcq_question_dir
    / "unique_questions.parquet"
)

mcq_metadata = (
    mcq_question_dir
    / "metadata.json"
)

for path in [
    mcq_map,
    mcq_unique,
    mcq_metadata,
]:

    if path.exists():

        print(
            "PASS:",
            path.name
        )

    else:

        failures.append(
            f"Missing MCQ asset: {path}"
        )

        print(
            "FAIL:",
            path.name
        )


if mcq_question_files:

    sample = load_torch(
        mcq_question_files[0]
    )

    print(
        "Keys:",
        list(
            sample.keys()
        )
    )


# ============================================================
# MCQ OPTIONS
# ============================================================

print()
print("=" * 72)
print("7. MCQ OPTION EMBEDDINGS")
print("=" * 72)

mcq_option_dir = (
    ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings"
)

mcq_option_files = sorted(
    mcq_option_dir.glob(
        "shard_*.pt"
    )
)

print(
    "Shard count:",
    len(mcq_option_files)
)

if len(mcq_option_files) != 30:

    failures.append(
        "Expected 30 MCQ option shards, "
        f"found {len(mcq_option_files)}"
    )


if mcq_option_files:

    option_sample = load_torch(
        mcq_option_files[0]
    )

    print(
        "Keys:",
        list(
            option_sample.keys()
        )
    )


# ============================================================
# REFERENCE GROUNDING
# ============================================================

print()
print("=" * 72)
print("8. REFERENCE GROUNDING")
print("=" * 72)

grounding_dir = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
)

grounding_files = sorted(
    grounding_dir.glob(
        "*.parquet"
    )
)

print(
    "Files:",
    [
        p.name
        for p in grounding_files
    ]
)

required_grounding_files = {
    "reference_patch_manifest.parquet",
    "reference_train.parquet",
    "reference_val.parquet",
}

actual_grounding_files = {
    p.name
    for p in grounding_files
}

missing_grounding = (
    required_grounding_files
    - actual_grounding_files
)

if missing_grounding:

    failures.append(
        "Missing grounding files: "
        + str(
            sorted(
                missing_grounding
            )
        )
    )


for name in [
    "reference_train.parquet",
    "reference_val.parquet",
]:

    path = (
        grounding_dir
        / name
    )

    if path.exists():

        df = pd.read_parquet(
            path
        )

        print(
            f"{name}: {len(df):,} rows"
        )

        print(
            "  columns:",
            df.columns.tolist()
        )


# ============================================================
# MANIFEST
# ============================================================

print()
print("=" * 72)
print("9. WRITING MANIFEST")
print("=" * 72)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


manifest = {
    "project": "SatQueryAI",
    "purpose": (
        "Kaggle single-modality "
        "specialist training"
    ),
    "root": str(ROOT),
    "assets": {
        key: {
            "bytes": value,
            "human_size":
                human_size(value),
        }
        for key, value
        in asset_sizes.items()
    },
    "total_bytes": total_bytes,
    "total_human_size":
        human_size(total_bytes),
    "dofa": {
        "shards":
            len(dofa_files),
    },
    "binary_vqa": {
        "rows":
            len(binary_df),
        "splits":
            binary_df[
                "model_split"
            ]
            .value_counts()
            .to_dict(),
    },
    "binary_question_embeddings": {
        "shards":
            len(
                binary_question_files
            ),
    },
    "mcq_question_embeddings": {
        "shards":
            len(
                mcq_question_files
            ),
    },
    "mcq_option_embeddings": {
        "shards":
            len(
                mcq_option_files
            ),
    },
    "grounding": {
        "files":
            sorted(
                actual_grounding_files
            ),
    },
    "validation": {
        "status":
            (
                "PASS"
                if not failures
                else "FAIL"
            ),
        "failures":
            failures,
    },
}


with open(
    OUT_MANIFEST,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2,
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 72)

if failures:

    print(
        "KAGGLE BUNDLE VALIDATION: FAIL"
    )

    print()

    for failure in failures:

        print(
            " -",
            failure
        )

    print()

    print(
        "Manifest:",
        OUT_MANIFEST
    )

    sys.exit(1)

else:

    print(
        "KAGGLE BUNDLE VALIDATION: PASS"
    )

    print()
    print(
        "Total asset size:",
        human_size(total_bytes)
    )

    print(
        "Manifest:",
        OUT_MANIFEST
    )

    print(
        "No training was performed."
    )

    print("=" * 72)