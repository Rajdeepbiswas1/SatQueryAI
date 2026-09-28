from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"D:\SatQueryAI")

CDVQA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CDVQA"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:

    print("=" * 76)
    print("SATQUERYAI — CDVQA TEST SPLIT INSPECTION")
    print("=" * 76)

    # ------------------------------------------------------------
    # 1. Test files
    # ------------------------------------------------------------
    print("\n[1] TEST FILES")

    required = [
        "Test_questions.json",
        "Test_answers.json",
        "Test_images.json",
        "Test2_questions.json",
        "Test2_answers.json",
        "Test2_images.json",
    ]

    for name in required:
        path = CDVQA_ROOT / name
        print(f"  {name:<24}: {'PASS' if path.exists() else 'MISSING'}")

        if not path.exists():
            raise FileNotFoundError(path)

    # ------------------------------------------------------------
    # 2. Load Test data
    # ------------------------------------------------------------
    print("\n[2] LOADING TEST DATA")

    test_questions = load_json(
        CDVQA_ROOT / "Test_questions.json"
    )

    test_answers = load_json(
        CDVQA_ROOT / "Test_answers.json"
    )

    test_images = load_json(
        CDVQA_ROOT / "Test_images.json"
    )

    print("Test questions records :", len(test_questions))
    print("Test answers records   :", len(test_answers))
    print("Test image records     :", len(test_images))

    # ------------------------------------------------------------
    # 3. Test2 data
    # ------------------------------------------------------------
    print("\n[3] LOADING TEST2 DATA")

    test2_questions = load_json(
        CDVQA_ROOT / "Test2_questions.json"
    )

    test2_answers = load_json(
        CDVQA_ROOT / "Test2_answers.json"
    )

    test2_images = load_json(
        CDVQA_ROOT / "Test2_images.json"
    )

    print("Test2 questions records:", len(test2_questions))
    print("Test2 answers records  :", len(test2_answers))
    print("Test2 image records    :", len(test2_images))

    # ------------------------------------------------------------
    # 4. Inspect record structure
    # ------------------------------------------------------------
    print("\n[4] SAMPLE RECORD STRUCTURE")

    for label, records in [
        ("Test question", test_questions),
        ("Test answer", test_answers),
        ("Test image", test_images),
    ]:

        print(f"\n--- {label} ---")

        if not records:
            print("EMPTY")
            continue

        first = records[0]

        print(
            "Type:",
            type(first)
        )

        if isinstance(first, dict):
            print(
                "Keys:",
                sorted(first.keys())
            )

            for key, value in first.items():
                print(
                    f"  {key}: "
                    f"type={type(value).__name__}, "
                    f"value={repr(value)[:300]}"
                )

    # ------------------------------------------------------------
    # 5. Question IDs
    # ------------------------------------------------------------
    print("\n[5] TEST QUESTION IDS")

    def extract_question_id(record):
        if not isinstance(record, dict):
            return None

        for key in (
            "question_id",
            "qid",
            "id",
        ):
            if key in record:
                return record[key]

        return None

    test_qids = [
        extract_question_id(x)
        for x in test_questions
    ]

    test_qids = [
        x for x in test_qids
        if x is not None
    ]

    print("Question IDs found:", len(test_qids))

    if test_qids:
        print("First 10:", test_qids[:10])
        print("Last 10 :", test_qids[-10:])
        print("Unique  :", len(set(test_qids)))

    # ------------------------------------------------------------
    # 6. Image / pair structure
    # ------------------------------------------------------------
    print("\n[6] TEST IMAGE / PAIR STRUCTURE")

    def find_field(record, candidates):
        if not isinstance(record, dict):
            return None, None

        for key in candidates:
            if key in record:
                return key, record[key]

        return None, None

    test_image_values = []

    for record in test_images:

        key, value = find_field(
            record,
            [
                "file_name",
                "filename",
                "image",
                "image_id",
                "id",
                "pair_id",
            ],
        )

        if value is not None:
            test_image_values.append(
                (key, value)
            )

    print(
        "Recognized image identifiers:",
        len(test_image_values)
    )

    print(
        "First 20 identifiers:"
    )

    for item in test_image_values[:20]:
        print(" ", item)

    # ------------------------------------------------------------
    # 7. Question type distribution
    # ------------------------------------------------------------
    print("\n[7] TEST QUESTION TYPES")

    type_counter = Counter()

    for record in test_questions:

        if not isinstance(record, dict):
            continue

        qtype = (
            record.get("question_type")
            or record.get("type")
            or record.get("category")
        )

        if qtype is not None:
            type_counter[str(qtype)] += 1

    for qtype, count in sorted(type_counter.items()):
        print(
            f"  {qtype:<24}: {count}"
        )

    # ------------------------------------------------------------
    # 8. Feature cache pair IDs
    # ------------------------------------------------------------
    print("\n[8] CACHED TEMPORAL PAIRS")

    feature_files = sorted(
        FEATURE_ROOT.glob("*.pt")
    )

    if len(feature_files) != 12:
        raise RuntimeError(
            f"Expected 12 feature shards, found {len(feature_files)}"
        )

    feature_pairs = []

    import torch

    for path in feature_files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids = data.get("pair_ids")

        if pair_ids is None:
            raise RuntimeError(
                f"Missing pair_ids in {path}"
            )

        feature_pairs.extend(
            str(x)
            for x in pair_ids
        )

    print(
        "Cached feature pairs:",
        len(feature_pairs)
    )

    print(
        "Unique cached pairs:",
        len(set(feature_pairs))
    )

    print(
        "First 20 cached IDs:",
        feature_pairs[:20]
    )

    # ------------------------------------------------------------
    # 9. Final
    # ------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TEST SPLIT INSPECTION COMPLETE")
    print("=" * 76)
    print("No files were modified.")
    print("No model was trained.")
    print("No embeddings were generated.")
    print("=" * 76)


if __name__ == "__main__":
    main()