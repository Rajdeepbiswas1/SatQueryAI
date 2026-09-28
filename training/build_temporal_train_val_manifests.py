from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"D:\SatQueryAI")

CDVQA_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "CDVQA"
)

FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

PREPROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_preprocessed"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_split"
)

SPLITS = [
    "Train",
    "Val",
    "Test",
]

EXPECTED_COUNTS = {
    "Train": 1600,
    "Val": 400,
    "Test": 968,
}

EXPECTED_TOTAL = 2968


def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def normalize_pair_id(file_name: str) -> str:

    stem = Path(
        str(file_name)
    ).stem

    if not stem.isdigit():
        raise RuntimeError(
            f"Non-numeric pair filename: {file_name}"
        )

    return str(
        int(stem)
    ).zfill(5)


def build_split_manifest(split: str):

    questions = load_json(
        CDVQA_ROOT
        / f"{split}_questions.json"
    )["questions"]

    answers = load_json(
        CDVQA_ROOT
        / f"{split}_answers.json"
    )["answers"]

    images = load_json(
        CDVQA_ROOT
        / f"{split}_images.json"
    )["images"]

    image_by_id = {
        int(image["id"]): image
        for image in images
    }

    answer_by_question = {}

    for answer in answers:

        qid = int(
            answer["question_id"]
        )

        answer_by_question[
            qid
        ] = str(
            answer["answer"]
        )

    rows = []

    for question in questions:

        qid = int(
            question["id"]
        )

        img_id = int(
            question["img_id"]
        )

        if img_id not in image_by_id:
            raise RuntimeError(
                f"{split}: question {qid} "
                f"references missing image {img_id}"
            )

        if qid not in answer_by_question:
            raise RuntimeError(
                f"{split}: question {qid} "
                "has no answer"
            )

        image = image_by_id[
            img_id
        ]

        file_name = str(
            image["file_name"]
        )

        pair_id = normalize_pair_id(
            file_name
        )

        rows.append(
            {
                "pair_id": pair_id,
                "file_name": file_name,
                "question_id": qid,
                "question_type": str(
                    question["type"]
                ),
                "question": str(
                    question["question"]
                ),
                "answer": answer_by_question[
                    qid
                ],
                "source_split": split,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    print("=" * 80)
    print(
        "SATQUERYAI — COMPLETE TEMPORAL "
        "QA / FEATURE MAPPING CHECK"
    )
    print("=" * 80)

    manifests = {}
    pair_sets = {}

    # ============================================================
    # BUILD TRAIN / VAL / TEST MANIFESTS
    # ============================================================

    for split in SPLITS:

        print()
        print("=" * 80)
        print(split.upper())
        print("=" * 80)

        df = build_split_manifest(
            split
        )

        manifests[split] = df

        pairs = set(
            df["pair_id"]
            .astype(str)
            .unique()
        )

        pair_sets[split] = pairs

        print(
            "QA rows:",
            f"{len(df):,}"
        )

        print(
            "Unique pairs:",
            f"{len(pairs):,}"
        )

        expected = EXPECTED_COUNTS[
            split
        ]

        if len(pairs) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} "
                f"pairs, found {len(pairs)}"
            )

    # ============================================================
    # PARTITION CHECK
    # ============================================================

    train_pairs = pair_sets["Train"]
    val_pairs = pair_sets["Val"]
    test_pairs = pair_sets["Test"]

    print()
    print("=" * 80)
    print("PAIR PARTITION")
    print("=" * 80)

    print(
        "Train pairs:",
        len(train_pairs)
    )

    print(
        "Validation pairs:",
        len(val_pairs)
    )

    print(
        "Test pairs:",
        len(test_pairs)
    )

    print(
        "Train ∩ Val:",
        len(train_pairs & val_pairs)
    )

    print(
        "Train ∩ Test:",
        len(train_pairs & test_pairs)
    )

    print(
        "Val ∩ Test:",
        len(val_pairs & test_pairs)
    )

    all_pairs = (
        train_pairs
        | val_pairs
        | test_pairs
    )

    print(
        "Combined unique pairs:",
        len(all_pairs)
    )

    if train_pairs & val_pairs:
        raise RuntimeError(
            "TRAIN / VAL PAIR OVERLAP"
        )

    if train_pairs & test_pairs:
        raise RuntimeError(
            "TRAIN / TEST PAIR OVERLAP"
        )

    if val_pairs & test_pairs:
        raise RuntimeError(
            "VAL / TEST PAIR OVERLAP"
        )

    if len(all_pairs) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} total pairs, "
            f"found {len(all_pairs)}"
        )

    # ============================================================
    # FEATURE CACHE CHECK
    # ============================================================

    print()
    print("=" * 80)
    print("FEATURE FILE CHECK")
    print("=" * 80)

    preprocessed_pairs = {
        path.stem
        for path in PREPROCESSED_DIR.glob(
            "*.pt"
        )
    }

    print(
        "Preprocessed pair files:",
        len(preprocessed_pairs)
    )

    missing_preprocessed = (
        all_pairs
        - preprocessed_pairs
    )

    extra_preprocessed = (
        preprocessed_pairs
        - all_pairs
    )

    print(
        "Missing preprocessed pairs:",
        len(missing_preprocessed)
    )

    print(
        "Extra preprocessed pairs:",
        len(extra_preprocessed)
    )

    if missing_preprocessed:
        print(
            sorted(
                missing_preprocessed
            )[:20]
        )

        raise RuntimeError(
            "Some QA pairs have no "
            "preprocessed temporal file."
        )

    if extra_preprocessed:
        print(
            sorted(
                extra_preprocessed
            )[:20]
        )

        raise RuntimeError(
            "Preprocessed files contain "
            "unexpected pairs."
        )

    # ============================================================
    # SHARDED FEATURE CACHE CHECK
    # ============================================================

    print()
    print("=" * 80)
    print("SHARDED TEMPORAL FEATURE CHECK")
    print("=" * 80)

    shard_paths = sorted(
        FEATURE_DIR.glob(
            "shard_*.pt"
        )
    )

    if not shard_paths:
        raise RuntimeError(
            "No temporal feature shards found."
        )

    feature_pairs = set()

    for shard_path in shard_paths:

        data = pd.Series()

        import torch

        shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        if "pair_ids" not in shard:
            raise RuntimeError(
                f"{shard_path.name}: "
                "missing pair_ids"
            )

        ids = [
            str(x)
            for x in shard["pair_ids"]
        ]

        feature_pairs.update(
            ids
        )

    print(
        "Feature shards:",
        len(shard_paths)
    )

    print(
        "Unique feature-cache pairs:",
        len(feature_pairs)
    )

    missing_sharded = (
        all_pairs
        - feature_pairs
    )

    extra_sharded = (
        feature_pairs
        - all_pairs
    )

    print(
        "Missing from feature cache:",
        len(missing_sharded)
    )

    print(
        "Extra feature-cache pairs:",
        len(extra_sharded)
    )

    if missing_sharded:
        raise RuntimeError(
            "Some QA pairs are absent "
            "from temporal feature cache."
        )

    if extra_sharded:
        raise RuntimeError(
            "Feature cache contains "
            "unexpected pair IDs."
        )

    # ============================================================
    # OUTPUT TRAIN / VAL MANIFESTS
    # ============================================================

    print()
    print("=" * 80)
    print("CREATING TRAIN / VALIDATION MANIFESTS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_manifest = manifests[
        "Train"
    ].drop(
        columns=["source_split"]
    )

    val_manifest = manifests[
        "Val"
    ].drop(
        columns=["source_split"]
    )

    train_path = (
        OUTPUT_DIR
        / "train_manifest.csv"
    )

    val_path = (
        OUTPUT_DIR
        / "val_manifest.csv"
    )

    train_manifest.to_csv(
        train_path,
        index=False,
    )

    val_manifest.to_csv(
        val_path,
        index=False,
    )

    # ============================================================
    # SAVE METADATA
    # ============================================================

    metadata = {
        "dataset": "SECOND / CDVQA",
        "total_pairs": EXPECTED_TOTAL,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_qa_rows": len(
            manifests["Train"]
        ),
        "validation_qa_rows": len(
            manifests["Val"]
        ),
        "test_qa_rows": len(
            manifests["Test"]
        ),
        "split_source": (
            "official CDVQA partition"
        ),
        "split_unit": "pair_id",
        "pair_overlap": {
            "train_val": len(
                train_pairs & val_pairs
            ),
            "train_test": len(
                train_pairs & test_pairs
            ),
            "val_test": len(
                val_pairs & test_pairs
            ),
        },
        "feature_pairs": len(
            feature_pairs
        ),
        "status": "verified",
    }

    metadata_path = (
        OUTPUT_DIR
        / "temporal_split_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    print()
    print(
        "All SECOND training pairs:",
        f"{len(all_pairs):,} / {EXPECTED_TOTAL:,}"
    )

    print(
        "All QA pairs have preprocessed features: PASS"
    )

    print(
        "All QA pairs have DOFA cache entries: PASS"
    )

    print(
        "Train pairs:",
        f"{len(train_pairs):,}"
    )

    print(
        "Validation pairs:",
        f"{len(val_pairs):,}"
    )

    print(
        "Test pairs:",
        f"{len(test_pairs):,}"
    )

    print()
    print("Train manifest:")
    print(train_path)

    print()
    print("Validation manifest:")
    print(val_path)

    print()
    print("Metadata:")
    print(metadata_path)

    print()
    print("=" * 80)
    print(
        "TEMPORAL QA / FEATURE MAPPING VERIFIED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()