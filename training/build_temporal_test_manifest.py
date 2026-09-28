from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\SatQueryAI")

CDVQA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CDVQA"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split"
)

TRAIN_MANIFEST = SPLIT_ROOT / "train_manifest.csv"
VAL_MANIFEST = SPLIT_ROOT / "val_manifest.csv"

OUTPUT_MANIFEST = SPLIT_ROOT / "test_manifest.csv"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_pair_id(value) -> str:
    text = str(value).strip()

    if text.lower().endswith(".png"):
        text = text[:-4]

    try:
        return f"{int(float(text)):05d}"
    except Exception:
        return text


def main() -> None:

    print("=" * 76)
    print("SATQUERYAI — BUILD CDVQA HELD-OUT TEST MANIFEST")
    print("=" * 76)

    # ------------------------------------------------------------
    # 1. Verify inputs
    # ------------------------------------------------------------
    required = [
        CDVQA_ROOT / "Test_questions.json",
        CDVQA_ROOT / "Test_answers.json",
        CDVQA_ROOT / "Test_images.json",
        TRAIN_MANIFEST,
        VAL_MANIFEST,
    ]

    print("\n[1] REQUIRED INPUTS")

    for path in required:

        exists = path.exists()

        print(
            f"  {path.name:<28}: "
            f"{'PASS' if exists else 'MISSING'}"
        )

        if not exists:
            raise FileNotFoundError(path)

    # ------------------------------------------------------------
    # 2. Load raw Test data
    # ------------------------------------------------------------
    print("\n[2] LOADING CDVQA TEST DATA")

    questions_data = load_json(
        CDVQA_ROOT / "Test_questions.json"
    )

    answers_data = load_json(
        CDVQA_ROOT / "Test_answers.json"
    )

    images_data = load_json(
        CDVQA_ROOT / "Test_images.json"
    )

    questions = questions_data["questions"]
    answers = answers_data["answers"]
    images = images_data["images"]

    print("Questions:", len(questions))
    print("Answers  :", len(answers))
    print("Images   :", len(images))

    if len(questions) != 39_686:
        raise RuntimeError(
            f"Expected 39,686 Test questions, "
            f"found {len(questions)}"
        )

    if len(answers) != 39_686:
        raise RuntimeError(
            f"Expected 39,686 Test answers, "
            f"found {len(answers)}"
        )

    # ------------------------------------------------------------
    # 3. Build image lookup
    # ------------------------------------------------------------
    image_by_id = {
        image["id"]: image
        for image in images
    }

    # ------------------------------------------------------------
    # 4. Build answer lookup
    # ------------------------------------------------------------
    answer_by_id = {
        answer["id"]: answer
        for answer in answers
    }

    answers_by_question = {}

    for answer in answers:

        qid = answer["question_id"]

        answers_by_question.setdefault(
            qid,
            []
        ).append(answer)

    # ------------------------------------------------------------
    # 5. Build manifest
    # ------------------------------------------------------------
    print("\n[3] BUILDING TEST MANIFEST")

    rows = []

    missing_images = []
    missing_answers = []
    mismatched_answer_links = []

    for question in questions:

        qid = question["id"]
        img_id = question["img_id"]

        # Image lookup
        image = image_by_id.get(img_id)

        if image is None:
            missing_images.append(
                (qid, img_id)
            )
            continue

        file_name = image["file_name"]

        # Question -> answer linkage
        answer_records = answers_by_question.get(
            qid,
            []
        )

        if not answer_records:
            missing_answers.append(qid)
            continue

        # CDVQA should provide the answer IDs in the question
        declared_answer_ids = set(
            question.get("answers_ids", [])
        )

        actual_answer_ids = {
            record["id"]
            for record in answer_records
        }

        if (
            declared_answer_ids
            and not declared_answer_ids.intersection(
                actual_answer_ids
            )
        ):
            mismatched_answer_links.append(
                {
                    "question_id": qid,
                    "declared": sorted(
                        declared_answer_ids
                    ),
                    "actual": sorted(
                        actual_answer_ids
                    ),
                }
            )

        # One semantic answer per question is expected here.
        answer_record = answer_records[0]

        rows.append(
            {
                "pair_id": normalize_pair_id(
                    file_name
                ),
                "file_name": file_name,
                "question_id": qid,
                "question_type": question["type"],
                "question": question["question"],
                "answer": answer_record["answer"],
            }
        )

    # ------------------------------------------------------------
    # 6. Basic integrity checks
    # ------------------------------------------------------------
    print("\n[4] INTEGRITY")

    print(
        "Rows constructed:",
        len(rows)
    )

    print(
        "Missing image references:",
        len(missing_images)
    )

    print(
        "Missing answer references:",
        len(missing_answers)
    )

    print(
        "Answer-link mismatches:",
        len(mismatched_answer_links)
    )

    if missing_images:
        raise RuntimeError(
            f"Missing image references: "
            f"{missing_images[:10]}"
        )

    if missing_answers:
        raise RuntimeError(
            f"Missing answers: "
            f"{missing_answers[:10]}"
        )

    if mismatched_answer_links:
        raise RuntimeError(
            f"Question/answer ID mismatches: "
            f"{mismatched_answer_links[:10]}"
        )

    if len(rows) != 39_686:
        raise RuntimeError(
            f"Expected 39,686 manifest rows, "
            f"created {len(rows)}"
        )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 7. Verify uniqueness
    # ------------------------------------------------------------
    print("\n[5] TEST MANIFEST STATISTICS")

    unique_pairs = df["pair_id"].nunique()
    unique_questions = df["question_id"].nunique()

    print(
        "Unique pairs:",
        unique_pairs
    )

    print(
        "Unique questions:",
        unique_questions
    )

    if unique_pairs != 968:
        raise RuntimeError(
            f"Expected 968 unique Test pairs, "
            f"found {unique_pairs}"
        )

    if unique_questions != 39_686:
        raise RuntimeError(
            f"Expected 39,686 unique Test question IDs, "
            f"found {unique_questions}"
        )

    # ------------------------------------------------------------
    # 8. Verify against train / validation pair sets
    # ------------------------------------------------------------
    print("\n[6] TRAIN / VALIDATION SEPARATION")

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )

    val_df = pd.read_csv(
        VAL_MANIFEST
    )

    train_pairs = {
        normalize_pair_id(x)
        for x in train_df["pair_id"]
    }

    val_pairs = {
        normalize_pair_id(x)
        for x in val_df["pair_id"]
    }

    test_pairs = set(
        df["pair_id"]
    )

    train_test_overlap = (
        train_pairs
        & test_pairs
    )

    val_test_overlap = (
        val_pairs
        & test_pairs
    )

    train_val_overlap = (
        train_pairs
        & val_pairs
    )

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
        "Train/Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation/Test overlap:",
        len(val_test_overlap)
    )

    print(
        "Train/Validation overlap:",
        len(train_val_overlap)
    )

    if train_test_overlap:
        raise RuntimeError(
            f"Train/Test overlap detected: "
            f"{sorted(train_test_overlap)[:20]}"
        )

    if val_test_overlap:
        raise RuntimeError(
            f"Validation/Test overlap detected: "
            f"{sorted(val_test_overlap)[:20]}"
        )

    if train_val_overlap:
        raise RuntimeError(
            f"Train/Validation overlap detected: "
            f"{sorted(train_val_overlap)[:20]}"
        )

    print("Pair separation: PASS")

    # ------------------------------------------------------------
    # 9. Verify expected total
    # ------------------------------------------------------------
    combined_pairs = (
        train_pairs
        | val_pairs
        | test_pairs
    )

    print("\n[7] COMBINED SPLIT")

    print(
        "Train + Val + Test:",
        len(combined_pairs)
    )

    print(
        "Expected:",
        2_968
    )

    if len(combined_pairs) != 2_968:
        raise RuntimeError(
            f"Expected 2,968 combined pairs, "
            f"found {len(combined_pairs)}"
        )

    # ------------------------------------------------------------
    # 10. Question type distribution
    # ------------------------------------------------------------
    print("\n[8] QUESTION TYPES")

    print(
        df["question_type"]
        .value_counts()
        .sort_index()
    )

    # ------------------------------------------------------------
    # 11. Write manifest
    # ------------------------------------------------------------
    print("\n[9] WRITING TEST MANIFEST")

    SPLIT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    print(
        "Created:",
        OUTPUT_MANIFEST
    )

    print(
        "File size:",
        OUTPUT_MANIFEST.stat().st_size,
        "bytes"
    )

    # ------------------------------------------------------------
    # 12. Final verification
    # ------------------------------------------------------------
    check = pd.read_csv(
        OUTPUT_MANIFEST
    )

    if len(check) != 39_686:
        raise RuntimeError(
            "Written manifest row count mismatch."
        )

    if check["pair_id"].nunique() != 968:
        raise RuntimeError(
            "Written manifest pair count mismatch."
        )

    print("\n" + "=" * 76)
    print("CDVQA TEST MANIFEST: PASS")
    print("=" * 76)

    print(
        "Rows              :",
        len(check)
    )

    print(
        "Unique pairs      :",
        check["pair_id"].nunique()
    )

    print(
        "Unique questions  :",
        check["question_id"].nunique()
    )

    print(
        "Train/Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Val/Test overlap  :",
        len(val_test_overlap)
    )

    print(
        "Combined pairs    :",
        len(combined_pairs)
    )

    print(
        "Output:",
        OUTPUT_MANIFEST
    )

    print("\nNo model was trained.")
    print("No question embeddings were generated.")
    print("=" * 76)


if __name__ == "__main__":
    main()