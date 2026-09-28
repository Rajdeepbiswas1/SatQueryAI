from pathlib import Path
import json


ROOT = Path(r"D:\SatQueryAI\data\raw\CDVQA")

SPLITS = [
    "Train",
    "Val",
    "Test",
    "Test2",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:

    print("=" * 80)
    print("SATQUERYAI — CDVQA SPLIT METADATA INSPECTION")
    print("=" * 80)

    grand_questions = 0
    grand_answers = 0
    grand_images = 0

    for split in SPLITS:

        questions_path = ROOT / f"{split}_questions.json"
        answers_path = ROOT / f"{split}_answers.json"
        images_path = ROOT / f"{split}_images.json"

        print()
        print("=" * 80)
        print(split.upper())
        print("=" * 80)

        for path in (
            questions_path,
            answers_path,
            images_path,
        ):
            print(
                f"{path.name:24s}: "
                f"{'FOUND' if path.exists() else 'MISSING'}"
            )

        if not all(
            path.exists()
            for path in (
                questions_path,
                answers_path,
                images_path,
            )
        ):
            continue

        questions_data = load_json(
            questions_path
        )
        answers_data = load_json(
            answers_path
        )
        images_data = load_json(
            images_path
        )

        questions = questions_data["questions"]
        answers = answers_data["answers"]
        images = images_data["images"]

        grand_questions += len(questions)
        grand_answers += len(answers)
        grand_images += len(images)

        question_ids = {
            int(q["id"])
            for q in questions
        }

        answer_question_ids = {
            int(a["question_id"])
            for a in answers
        }

        image_ids = {
            int(image["id"])
            for image in images
        }

        question_img_ids = {
            int(q["img_id"])
            for q in questions
        }

        filenames = {
            str(image["file_name"])
            for image in images
        }

        listed_question_ids = set()

        for image in images:
            for qid in image.get(
                "questions_ids",
                []
            ):
                listed_question_ids.add(
                    int(qid)
                )

        missing_answers = (
            question_ids
            - answer_question_ids
        )

        missing_images = (
            question_img_ids
            - image_ids
        )

        not_listed = (
            question_ids
            - listed_question_ids
        )

        print()
        print(
            "Questions              :",
            f"{len(questions):,}"
        )

        print(
            "Answers                :",
            f"{len(answers):,}"
        )

        print(
            "Image records          :",
            f"{len(images):,}"
        )

        print(
            "Unique image filenames :",
            f"{len(filenames):,}"
        )

        print(
            "Unique question IDs    :",
            f"{len(question_ids):,}"
        )

        print(
            "Unique image IDs       :",
            f"{len(image_ids):,}"
        )

        print(
            "Question IDs missing answers:",
            f"{len(missing_answers):,}"
        )

        print(
            "Question img_ids missing image:",
            f"{len(missing_images):,}"
        )

        print(
            "Questions not listed by source image:",
            f"{len(not_listed):,}"
        )

        print()
        print("Question ID range:")

        if question_ids:
            print(
                f"{min(question_ids)} "
                f"to "
                f"{max(question_ids)}"
            )
        else:
            print("NONE")

        print()
        print("Sample image records:")

        for image in images[:3]:
            print(image)

        print()
        print("Sample questions:")

        for question in questions[:3]:
            print(question)

        print()
        print("Sample answers:")

        for answer in answers[:3]:
            print(answer)

    print()
    print("=" * 80)
    print("ALL CDVQA SPLITS — GRAND TOTALS")
    print("=" * 80)

    print(
        "Questions    :",
        f"{grand_questions:,}"
    )

    print(
        "Answers      :",
        f"{grand_answers:,}"
    )

    print(
        "Image records:",
        f"{grand_images:,}"
    )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()