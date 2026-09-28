from pathlib import Path
import json
import shutil


# ============================================================
# SATQUERYAI — KAGGLE BUNDLE PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "processed"

BUNDLE = (
    ROOT
    / "training"
    / "single_modality"
    / "kaggle_bundle"
)


# ============================================================
# SOURCE ASSETS
# ============================================================

ASSETS = {
    "dofa_features": (
        SOURCE
        / "dofa_features_download"
        / "dofa_features"
    ),

    "binary_vqa": (
        SOURCE
        / "binary_vqa_split.parquet"
    ),

    "binary_question_embeddings": (
        SOURCE
        / "binary_question_embeddings"
    ),

    "mcq_question_embeddings": (
        SOURCE
        / "mcq_question_embeddings"
    ),

    "mcq_option_embeddings": (
        SOURCE
        / "mcq_option_embeddings"
    ),

    "reference_grounding_split": (
        SOURCE
        / "reference_grounding_split"
    ),
}


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

OUTPUTS = {
    "dofa_features": BUNDLE / "dofa_features",

    "binary_vqa": (
        BUNDLE
        / "binary_vqa"
        / "binary_vqa_split.parquet"
    ),

    "binary_question_embeddings": (
        BUNDLE
        / "binary_vqa"
        / "question_embeddings"
    ),

    "mcq_question_embeddings": (
        BUNDLE
        / "mcq_vqa"
        / "question_embeddings"
    ),

    "mcq_option_embeddings": (
        BUNDLE
        / "mcq_vqa"
        / "option_embeddings"
    ),

    "reference_grounding_split": (
        BUNDLE
        / "grounding"
        / "reference_grounding_split"
    ),
}


# ============================================================
# HELPERS
# ============================================================

def size_bytes(path: Path) -> int:

    if path.is_file():
        return path.stat().st_size

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


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


def copy_asset(
    source: Path,
    destination: Path,
):

    if not source.exists():
        raise FileNotFoundError(
            f"Source asset does not exist:\n{source}"
        )

    if source.is_dir():

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"COPY DIR\n"
            f"  FROM: {source}\n"
            f"  TO:   {destination}"
        )

        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
        )

    else:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"COPY FILE\n"
            f"  FROM: {source}\n"
            f"  TO:   {destination}"
        )

        shutil.copy2(
            source,
            destination,
        )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 72)
print("SATQUERYAI — KAGGLE BUNDLE PREPARATION")
print("=" * 72)

print()
print("Source:")
print(SOURCE)

print()
print("Bundle:")
print(BUNDLE)


# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

if BUNDLE.exists():

    print()
    print(
        "Existing bundle found."
    )

    print(
        "It will be refreshed from the verified source assets."
    )

    shutil.rmtree(
        BUNDLE
    )


BUNDLE.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# COPY ASSETS
# ============================================================

print()
print("=" * 72)
print("1. COPYING VERIFIED ASSETS")
print("=" * 72)

for name, source in ASSETS.items():

    destination = OUTPUTS[
        name
    ]

    copy_asset(
        source,
        destination,
    )


# ============================================================
# COPY VALIDATION MANIFEST
# ============================================================

validator_manifest = (
    ROOT
    / "training"
    / "single_modality"
    / "kaggle_bundle_manifest"
    / "manifest.json"
)

if validator_manifest.exists():

    shutil.copy2(
        validator_manifest,
        BUNDLE
        / "source_validation_manifest.json",
    )


# ============================================================
# CREATE README
# ============================================================

readme = """SatQueryAI — Single Modality Training Bundle

Purpose
-------
Training inputs for the S1-only and S2-only VQA and
grounding specialists.

This bundle contains cached DOFA representations and
task supervision. Raw satellite imagery is intentionally
not included.

Structure
---------

dofa_features/
    shard_0000.pt
    ...
    shard_0039.pt

binary_vqa/
    binary_vqa_split.parquet
    question_embeddings/

mcq_vqa/
    question_embeddings/
    option_embeddings/

grounding/
    reference_grounding_split/
        reference_patch_manifest.parquet
        reference_train.parquet
        reference_val.parquet

Important
---------
The DOFA shards contain:
    patch_ids
    s1
    s2

S1 shape per sample:
    [196, 768]

S2 shape per sample:
    [196, 768]

Training plan
-------------
1. S2 Binary VQA
2. S1 Binary VQA
3. S2 MCQ VQA
4. S1 MCQ VQA
5. S2 Reference Grounding
6. S1 Reference Grounding

Existing model checkpoints are NOT included or modified.

Prepared from verified local assets.
"""

(BUNDLE / "README.txt").write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# FINAL INVENTORY
# ============================================================

print()
print("=" * 72)
print("2. FINAL BUNDLE INVENTORY")
print("=" * 72)

inventory = {}

for item in BUNDLE.rglob("*"):

    if not item.is_file():
        continue

    relative = str(
        item.relative_to(BUNDLE)
    )

    size = item.stat().st_size

    inventory[relative] = size

    print(
        f"{human_size(size):>12}  "
        f"{relative}"
    )


total_size = sum(
    inventory.values()
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "project": "SatQueryAI",
    "purpose": (
        "Single-modality specialist "
        "training bundle"
    ),
    "bundle_path": str(BUNDLE),
    "total_files": len(
        inventory
    ),
    "total_bytes": total_size,
    "total_human_size":
        human_size(total_size),
    "assets": {
        name: {
            "source": str(source),
            "bundle_destination":
                str(
                    OUTPUTS[name]
                ),
        }
        for name, source
        in ASSETS.items()
    },
}


with open(
    BUNDLE
    / "bundle_manifest.json",
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)
print("KAGGLE BUNDLE READY")
print("=" * 72)

print()
print(
    "Files:",
    len(inventory)
)

print(
    "Total:",
    human_size(total_size)
)

print()
print(
    "Bundle:",
    BUNDLE
)

print()
print(
    "Original source data was not modified."
)

print("=" * 72)