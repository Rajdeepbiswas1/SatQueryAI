from pathlib import Path
import json
import random

import pandas as pd


ROOT = Path(r"D:\SatQueryAI")

PREPROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_preprocessed"
)

FULL_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "temporal_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_split"
)

SEED = 42
TRAIN_RATIO = 0.80

EXPECTED_PAIRS = 2968


def main() -> None:

    print("=" * 72)
    print("SATQUERYAI — TEMPORAL TRAIN / VALIDATION SPLIT")
    print("=" * 72)

    # ============================================================
    # 1. LOAD FULL MANIFEST
    # ============================================================

    print()
    print("Loading temporal manifest...")

    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{FULL_MANIFEST}"
        )

    manifest = pd.read_csv(
        FULL_MANIFEST
    )

    print(
        "Manifest rows:",
        f"{len(manifest):,}"
    )

    required_columns = [
        "pair_id",
        "file_name",
        "t1_path",
        "t2_path",
        "label1_path",
        "label2_path",
        "question_id",
        "question_type",
        "question",
        "answer",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ============================================================
    # 2. NORMALIZE PAIR IDS
    # ============================================================

    manifest = manifest.copy()

    manifest["_pair_id_normalized"] = (
        manifest["pair_id"]
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    unique_manifest_pairs = set(
        manifest["_pair_id_normalized"]
        .unique()
    )

    print(
        "Unique manifest pairs:",
        len(unique_manifest_pairs)
    )

    if len(unique_manifest_pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} unique manifest pairs, "
            f"found {len(unique_manifest_pairs)}."
        )

    # ============================================================
    # 3. VERIFY PREPROCESSED PAIR POOL
    # ============================================================

    print()
    print("Checking preprocessed temporal pairs...")

    feature_files = sorted(
        PREPROCESSED_DIR.glob("*.pt")
    )

    preprocessed_pairs = {
        path.stem
        for path in feature_files
    }

    print(
        "Preprocessed pair files:",
        len(preprocessed_pairs)
    )

    if len(preprocessed_pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} preprocessed pairs, "
            f"found {len(preprocessed_pairs)}."
        )

    missing_features = (
        unique_manifest_pairs
        - preprocessed_pairs
    )

    extra_features = (
        preprocessed_pairs
        - unique_manifest_pairs
    )

    print(
        "Manifest pairs missing features:",
        len(missing_features)
    )

    print(
        "Extra preprocessed pairs:",
        len(extra_features)
    )

    if missing_features:
        print(
            "Examples:",
            sorted(missing_features)[:20]
        )

        raise RuntimeError(
            "Some manifest pairs do not have "
            "preprocessed feature files."
        )

    if extra_features:
        print(
            "Examples:",
            sorted(extra_features)[:20]
        )

        raise RuntimeError(
            "Preprocessed directory contains pairs "
            "not represented in the manifest."
        )

    # ============================================================
    # 4. CREATE DETERMINISTIC PAIR-LEVEL SPLIT
    # ============================================================

    print()
    print("Creating deterministic pair-level split...")

    all_pairs = sorted(
        unique_manifest_pairs
    )

    rng = random.Random(
        SEED
    )

    shuffled_pairs = all_pairs.copy()

    rng.shuffle(
        shuffled_pairs
    )

    train_count = int(
        EXPECTED_PAIRS
        * TRAIN_RATIO
    )

    val_count = (
        EXPECTED_PAIRS
        - train_count
    )

    train_pairs = sorted(
        shuffled_pairs[:train_count]
    )

    val_pairs = sorted(
        shuffled_pairs[train_count:]
    )

    train_set = set(
        train_pairs
    )

    val_set = set(
        val_pairs
    )

    # ============================================================
    # 5. SPLIT INTEGRITY
    # ============================================================

    print()
    print("Checking split integrity...")

    overlap = (
        train_set
        & val_set
    )

    union = (
        train_set
        | val_set
    )

    print(
        "Train/validation overlap:",
        len(overlap)
    )

    print(
        "Total covered pairs:",
        len(union)
    )

    if overlap:
        raise RuntimeError(
            "PAIR LEAKAGE DETECTED: "
            "training and validation share pair IDs."
        )

    if union != unique_manifest_pairs:
        raise RuntimeError(
            "Coverage failure: "
            "train + validation does not equal "
            "the complete 2,968-pair pool."
        )

    if len(train_pairs) != 2374:
        raise RuntimeError(
            f"Expected 2,374 training pairs, "
            f"found {len(train_pairs)}."
        )

    if len(val_pairs) != 594:
        raise RuntimeError(
            f"Expected 594 validation pairs, "
            f"found {len(val_pairs)}."
        )

    # ============================================================
    # 6. BUILD TRAIN / VALIDATION MANIFESTS
    # ============================================================

    train_manifest = manifest[
        manifest["_pair_id_normalized"].isin(
            train_set
        )
    ].copy()

    val_manifest = manifest[
        manifest["_pair_id_normalized"].isin(
            val_set
        )
    ].copy()

    train_manifest = train_manifest[
        required_columns
    ].reset_index(
        drop=True
    )

    val_manifest = val_manifest[
        required_columns
    ].reset_index(
        drop=True
    )

    # ============================================================
    # 7. VERIFY QA-ROW COVERAGE
    # ============================================================

    print()
    print("QA row distribution:")
    print(
        "Training rows:",
        f"{len(train_manifest):,}"
    )

    print(
        "Validation rows:",
        f"{len(val_manifest):,}"
    )

    total_split_rows = (
        len(train_manifest)
        + len(val_manifest)
    )

    print(
        "Combined split rows:",
        f"{total_split_rows:,}"
    )

    if total_split_rows != len(manifest):
        raise RuntimeError(
            "QA-row coverage failure."
        )

    train_question_ids = set(
        train_manifest["question_id"]
    )

    val_question_ids = set(
        val_manifest["question_id"]
    )

    question_overlap = (
        train_question_ids
        & val_question_ids
    )

    if question_overlap:
        raise RuntimeError(
            "Question-ID overlap detected."
        )

    if (
        len(train_question_ids)
        + len(val_question_ids)
        != len(manifest)
    ):
        raise RuntimeError(
            "Question-ID coverage failure."
        )

    # ============================================================
    # 8. QUESTION-TYPE DISTRIBUTION
    # ============================================================

    print()
    print("QUESTION TYPE DISTRIBUTION")
    print("-" * 72)

    all_type_counts = (
        manifest["question_type"]
        .value_counts()
        .sort_index()
    )

    train_type_counts = (
        train_manifest["question_type"]
        .value_counts()
        .sort_index()
    )

    val_type_counts = (
        val_manifest["question_type"]
        .value_counts()
        .sort_index()
    )

    for question_type in all_type_counts.index:

        total_count = int(
            all_type_counts.get(
                question_type,
                0
            )
        )

        train_count_type = int(
            train_type_counts.get(
                question_type,
                0
            )
        )

        val_count_type = int(
            val_type_counts.get(
                question_type,
                0
            )
        )

        print(
            f"{question_type:22s}"
            f" total={total_count:6d}"
            f" train={train_count_type:6d}"
            f" val={val_count_type:6d}"
        )

    # ============================================================
    # 9. OUTPUT DIRECTORY
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_pairs_path = (
        OUTPUT_DIR
        / "train_pairs.txt"
    )

    val_pairs_path = (
        OUTPUT_DIR
        / "val_pairs.txt"
    )

    train_manifest_path = (
        OUTPUT_DIR
        / "train_manifest.csv"
    )

    val_manifest_path = (
        OUTPUT_DIR
        / "val_manifest.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / "split_metadata.json"
    )

    # ============================================================
    # 10. SAVE PAIR LISTS
    # ============================================================

    train_pairs_path.write_text(
        "\n".join(train_pairs)
        + "\n",
        encoding="utf-8",
    )

    val_pairs_path.write_text(
        "\n".join(val_pairs)
        + "\n",
        encoding="utf-8",
    )

    # ============================================================
    # 11. SAVE MANIFESTS
    # ============================================================

    train_manifest.to_csv(
        train_manifest_path,
        index=False,
    )

    val_manifest.to_csv(
        val_manifest_path,
        index=False,
    )

    # ============================================================
    # 12. SAVE METADATA
    # ============================================================

    metadata = {
        "dataset": "SECOND",
        "source_manifest": str(
            FULL_MANIFEST
        ),
        "source_preprocessed_directory": str(
            PREPROCESSED_DIR
        ),
        "total_pairs": EXPECTED_PAIRS,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "train_ratio": (
            len(train_pairs)
            / EXPECTED_PAIRS
        ),
        "validation_ratio": (
            len(val_pairs)
            / EXPECTED_PAIRS
        ),
        "random_seed": SEED,
        "split_unit": "pair_id",
        "total_manifest_rows": len(manifest),
        "train_manifest_rows": len(
            train_manifest
        ),
        "validation_manifest_rows": len(
            val_manifest
        ),
        "pair_overlap": len(
            overlap
        ),
        "question_id_overlap": len(
            question_overlap
        ),
        "coverage_pairs": len(
            union
        ),
        "coverage_manifest_rows": total_split_rows,
        "status": "verified",
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # 13. FINAL REPORT
    # ============================================================

    print()
    print("=" * 72)
    print("FINAL SPLIT")
    print("=" * 72)

    print()
    print(
        "Total pairs       :",
        f"{EXPECTED_PAIRS:,}"
    )

    print(
        "Training pairs     :",
        f"{len(train_pairs):,}"
    )

    print(
        "Validation pairs   :",
        f"{len(val_pairs):,}"
    )

    print()
    print(
        "Training QA rows   :",
        f"{len(train_manifest):,}"
    )

    print(
        "Validation QA rows :",
        f"{len(val_manifest):,}"
    )

    print()
    print(
        "Pair overlap       :",
        len(overlap)
    )

    print(
        "Question overlap   :",
        len(question_overlap)
    )

    print(
        "Pair coverage      :",
        f"{len(union):,} / {EXPECTED_PAIRS:,}"
    )

    print()
    print("TRAIN PAIRS FILE:")
    print(train_pairs_path)

    print()
    print("VALIDATION PAIRS FILE:")
    print(val_pairs_path)

    print()
    print("TRAIN MANIFEST:")
    print(train_manifest_path)

    print()
    print("VALIDATION MANIFEST:")
    print(val_manifest_path)

    print()
    print("SPLIT METADATA:")
    print(metadata_path)

    print()
    print("First 10 training pairs:")
    print(train_pairs[:10])

    print()
    print("First 10 validation pairs:")
    print(val_pairs[:10])

    print()
    print("=" * 72)
    print("TEMPORAL TRAIN / VALIDATION SPLIT VERIFIED")
    print("=" * 72)


if __name__ == "__main__":
    main()