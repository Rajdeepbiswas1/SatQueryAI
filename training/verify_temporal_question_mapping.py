from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\SatQueryAI")

RAW_CDVQA = PROJECT_ROOT / "data" / "raw" / "CDVQA"

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_manifest.csv"
)


def main() -> None:
    print("=" * 72)
    print("SATQUERYAI — TEMPORAL QUESTION MAPPING VERIFICATION")
    print("=" * 72)

    # ------------------------------------------------------------
    # Load manifest.
    # ------------------------------------------------------------

    manifest = pd.read_csv(MANIFEST)

    print()
    print("Manifest rows:", len(manifest))
    print(
        "Unique manifest question IDs:",
        manifest["question_id"].nunique(),
    )

    # ------------------------------------------------------------
    # Load original CDVQA question records.
    # ------------------------------------------------------------

    with open(
        RAW_CDVQA / "Train_questions.json",
        "r",
        encoding="utf-8",
    ) as f:
        question_data = json.load(f)

    questions = question_data["questions"]

    question_by_id = {
        int(q["id"]): q
        for q in questions
    }

    print("Raw questions:", len(question_by_id))

    # ------------------------------------------------------------
    # Load original CDVQA answer records.
    # ------------------------------------------------------------

    with open(
        RAW_CDVQA / "Train_answers.json",
        "r",
        encoding="utf-8",
    ) as f:
        answer_data = json.load(f)

    answers = answer_data["answers"]

    answers_by_question = {}

    for record in answers:
        question_id = int(record["question_id"])
        answer = str(record["answer"]).strip()

        answers_by_question.setdefault(
            question_id,
            [],
        ).append(answer)

    print("Raw answers:", len(answers))

    # ------------------------------------------------------------
    # Load original CDVQA image records.
    # ------------------------------------------------------------

    with open(
        RAW_CDVQA / "Train_images.json",
        "r",
        encoding="utf-8",
    ) as f:
        image_data = json.load(f)

    images = image_data["images"]

    image_by_id = {
        int(image["id"]): image
        for image in images
    }

    print("Raw image records:", len(image_by_id))

    # ------------------------------------------------------------
    # Verification counters.
    # ------------------------------------------------------------

    missing_questions = []
    question_text_mismatches = []
    question_type_mismatches = []
    answer_mismatches = []

    missing_images = []
    question_not_listed_in_image = []
    image_filename_mismatches = []
    pair_filename_mismatches = []

    # ------------------------------------------------------------
    # Check every manifest row.
    # ------------------------------------------------------------

    for row in manifest.itertuples(index=False):
        question_id = int(row.question_id)

        # -------------------------
        # Question existence.
        # -------------------------

        question = question_by_id.get(question_id)

        if question is None:
            missing_questions.append(question_id)
            continue

        # -------------------------
        # Question text.
        # -------------------------

        manifest_question = str(
            row.question
        ).strip()

        raw_question = str(
            question["question"]
        ).strip()

        if manifest_question != raw_question:
            question_text_mismatches.append(
                question_id
            )

        # -------------------------
        # Question type.
        # -------------------------

        manifest_type = str(
            row.question_type
        ).strip()

        raw_type = str(
            question["type"]
        ).strip()

        if manifest_type != raw_type:
            question_type_mismatches.append(
                question_id
            )

        # -------------------------
        # Answer.
        # -------------------------

        manifest_answer = str(
            row.answer
        ).strip()

        raw_answers = answers_by_question.get(
            question_id,
            [],
        )

        if manifest_answer not in raw_answers:
            answer_mismatches.append(
                question_id
            )

        # -------------------------
        # Image mapping.
        # -------------------------

        image_id = int(
            question["img_id"]
        )

        image = image_by_id.get(image_id)

        if image is None:
            missing_images.append(question_id)
            continue

        # -------------------------
        # Question must belong to
        # its source image record.
        # -------------------------

        image_question_ids = {
            int(qid)
            for qid in image.get(
                "questions_ids",
                [],
            )
        }

        if question_id not in image_question_ids:
            question_not_listed_in_image.append(
                question_id
            )

        # -------------------------
        # Image filename mapping.
        # -------------------------

        manifest_filename = str(
            row.file_name
        ).strip()

        image_filename = str(
            image["file_name"]
        ).strip()

        if manifest_filename != image_filename:
            image_filename_mismatches.append(
                question_id
            )

        # -------------------------
        # Pair ID ↔ filename stem.
        # -------------------------

        manifest_pair_id = int(
            row.pair_id
        )

        filename_stem = Path(
            manifest_filename
        ).stem

        try:
            filename_pair_id = int(
                filename_stem
            )
        except ValueError:
            pair_filename_mismatches.append(
                question_id
            )
            continue

        if manifest_pair_id != filename_pair_id:
            pair_filename_mismatches.append(
                question_id
            )

    # ------------------------------------------------------------
    # Global question-ID coverage.
    # ------------------------------------------------------------

    manifest_question_ids = {
        int(value)
        for value in manifest["question_id"]
    }

    raw_question_ids = set(
        question_by_id.keys()
    )

    missing_from_manifest = (
        raw_question_ids
        - manifest_question_ids
    )

    extra_in_manifest = (
        manifest_question_ids
        - raw_question_ids
    )

    # ------------------------------------------------------------
    # Results.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("VERIFICATION RESULTS")
    print("=" * 72)

    print(
        "Missing question records:",
        len(missing_questions),
    )

    print(
        "Question text mismatches:",
        len(question_text_mismatches),
    )

    print(
        "Question type mismatches:",
        len(question_type_mismatches),
    )

    print(
        "Answer mismatches:",
        len(answer_mismatches),
    )

    print(
        "Missing image records:",
        len(missing_images),
    )

    print(
        "Question not listed in source image:",
        len(question_not_listed_in_image),
    )

    print(
        "Image filename mismatches:",
        len(image_filename_mismatches),
    )

    print(
        "Pair ID ↔ filename mismatches:",
        len(pair_filename_mismatches),
    )

    print(
        "Raw question IDs missing from manifest:",
        len(missing_from_manifest),
    )

    print(
        "Unexpected question IDs in manifest:",
        len(extra_in_manifest),
    )

    # ------------------------------------------------------------
    # Show one complete mapping example.
    # ------------------------------------------------------------

    first = manifest.iloc[0]

    qid = int(first["question_id"])
    q = question_by_id[qid]

    img_id = int(q["img_id"])
    image = image_by_id[img_id]

    print()
    print("=" * 72)
    print("SAMPLE COMPLETE MAPPING")
    print("=" * 72)

    print(
        "Manifest question_id :",
        qid,
    )

    print(
        "Raw question.id      :",
        int(q["id"]),
    )

    print(
        "Raw question.img_id  :",
        img_id,
    )

    print(
        "Source image filename:",
        image["file_name"],
    )

    print(
        "Manifest pair_id     :",
        int(first["pair_id"]),
    )

    print(
        "Manifest filename    :",
        first["file_name"],
    )

    print(
        "Manifest question    :",
        first["question"],
    )

    print(
        "Raw question         :",
        q["question"],
    )

    print(
        "Manifest answer      :",
        first["answer"],
    )

    print(
        "Raw answer(s)        :",
        answers_by_question.get(qid, []),
    )

    print(
        "Question listed in image:",
        qid in {
            int(v)
            for v in image.get(
                "questions_ids",
                [],
            )
        },
    )

    # ------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------

    all_passed = (
        len(missing_questions) == 0
        and len(question_text_mismatches) == 0
        and len(question_type_mismatches) == 0
        and len(answer_mismatches) == 0
        and len(missing_images) == 0
        and len(question_not_listed_in_image) == 0
        and len(image_filename_mismatches) == 0
        and len(pair_filename_mismatches) == 0
        and len(missing_from_manifest) == 0
        and len(extra_in_manifest) == 0
    )

    print()

    if all_passed:
        print("=" * 72)
        print("TEMPORAL QUESTION MAPPING VERIFIED")
        print("=" * 72)
    else:
        print("=" * 72)
        print("TEMPORAL QUESTION MAPPING FAILED")
        print("=" * 72)

        raise RuntimeError(
            "One or more temporal question mapping checks failed."
        )


if __name__ == "__main__":
    main()