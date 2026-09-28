from pathlib import Path


PROJECT = Path(r"D:\SatQueryAI")

CDVQA_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "CDVQA"
)

TEMPORAL_FEATURE_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "temporal_features"
)

TEMPORAL_PREPROCESSED_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "temporal_preprocessed"
)

TEMPORAL_RAW_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "temporal_raw"
)

TEMPORAL_KAGGLE_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "temporal_kaggle_upload"
)

CHECKPOINT_DIR = (
    PROJECT
    / "models"
    / "checkpoints"
)

SOURCE_DIRS = [
    PROJECT / "training",
    PROJECT / "src",
    PROJECT / "preprocessing",
    PROJECT / "kaggle",
]


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def list_files(path, extensions=None):
    print()
    print(f"DIRECTORY: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        return

    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
    )

    if extensions is not None:
        files = [
            p
            for p in files
            if p.suffix.lower() in extensions
        ]

    print(f"FILE COUNT: {len(files)}")

    for file_path in files[:250]:
        size = file_path.stat().st_size

        print(
            f"{file_path.relative_to(path)}"
            f"    "
            f"{size:,} bytes"
        )

    if len(files) > 250:
        print(
            f"... {len(files) - 250} additional files"
        )


def list_directories(path):
    print()
    print(f"DIRECTORIES UNDER: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        return

    directories = sorted(
        p
        for p in path.rglob("*")
        if p.is_dir()
    )

    print(
        f"DIRECTORY COUNT: "
        f"{len(directories)}"
    )

    for directory in directories[:150]:
        print(
            directory.relative_to(path)
        )


# ============================================================
# START
# ============================================================

print_header(
    "SATQUERYAI — TEMPORAL ASSET INSPECTION"
)

print(
    f"PROJECT: {PROJECT}"
)

print(
    "PURPOSE: identify existing CDVQA implementation, "
    "checkpoints, and prepared temporal assets."
)

print(
    "NO TRAINING WILL BE PERFORMED."
)


# ============================================================
# 1. CDVQA SOURCE / DATA
# ============================================================

print_header(
    "1. CDVQA RAW DATA / SOURCE"
)

list_directories(
    CDVQA_DIR
)

list_files(
    CDVQA_DIR,
    extensions={
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".txt",
        ".md",
    },
)


# ============================================================
# 2. TEMPORAL FEATURE CACHE
# ============================================================

print_header(
    "2. TEMPORAL DOFA FEATURE CACHE"
)

list_directories(
    TEMPORAL_FEATURE_DIR
)

list_files(
    TEMPORAL_FEATURE_DIR,
    extensions={
        ".pt",
        ".json",
        ".csv",
        ".txt",
        ".parquet",
    },
)


# ============================================================
# 3. TEMPORAL PREPROCESSING
# ============================================================

print_header(
    "3. TEMPORAL PREPROCESSED DATA"
)

list_directories(
    TEMPORAL_PREPROCESSED_DIR
)

list_files(
    TEMPORAL_PREPROCESSED_DIR,
    extensions={
        ".pt",
        ".json",
        ".csv",
        ".txt",
        ".parquet",
    },
)


# ============================================================
# 4. TEMPORAL RAW DATA
# ============================================================

print_header(
    "4. TEMPORAL RAW DATA"
)

list_directories(
    TEMPORAL_RAW_DIR
)

list_files(
    TEMPORAL_RAW_DIR,
    extensions={
        ".pt",
        ".json",
        ".csv",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    },
)


# ============================================================
# 5. KAGGLE TEMPORAL PACKAGE
# ============================================================

print_header(
    "5. TEMPORAL KAGGLE PACKAGE"
)

list_directories(
    TEMPORAL_KAGGLE_DIR
)

list_files(
    TEMPORAL_KAGGLE_DIR,
    extensions={
        ".pt",
        ".json",
        ".csv",
        ".txt",
        ".parquet",
        ".zip",
    },
)


# ============================================================
# 6. EXISTING CHECKPOINTS
# ============================================================

print_header(
    "6. EXISTING MODEL CHECKPOINTS"
)

list_files(
    CHECKPOINT_DIR,
    extensions={
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
    },
)


# ============================================================
# 7. EXISTING TEMPORAL-RELATED SOURCE
# ============================================================

print_header(
    "7. TEMPORAL-RELATED PYTHON SOURCE"
)

keywords = (
    "temporal",
    "change",
    "cdvqa",
    "second",
)

found = []

for source_dir in SOURCE_DIRS:

    if not source_dir.exists():
        continue

    for path in source_dir.rglob("*.py"):

        name = path.name.lower()

        path_text = str(
            path.relative_to(PROJECT)
        ).lower()

        if any(
            keyword in name
            or keyword in path_text
            for keyword in keywords
        ):
            found.append(
                path
            )

for path in sorted(
    set(found)
)[:250]:

    print(
        path.relative_to(PROJECT)
    )

print()

print(
    f"Temporal-related Python files found: "
    f"{len(set(found))}"
)


# ============================================================
# 8. FINAL
# ============================================================

print_header(
    "TEMPORAL ASSET INSPECTION COMPLETE"
)

print(
    "No model was loaded."
)

print(
    "No checkpoint was modified."
)

print(
    "No training was performed."
)

print("=" * 72)