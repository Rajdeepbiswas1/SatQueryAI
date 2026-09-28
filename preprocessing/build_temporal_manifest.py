import json
import os
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = r"D:\SatQueryAI"
CDVQA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "CDVQA")

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_CSV = os.path.join(OUT_DIR, "temporal_manifest.csv")
OUT_STATS = os.path.join(OUT_DIR, "temporal_manifest_stats.txt")


def load_json(path, key):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if key not in data:
        raise KeyError(f"Expected key '{key}' in {path}")

    return data[key]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    images = load_json(
        os.path.join(CDVQA_ROOT, "Train_images.json"),
        "images"
    )

    questions = load_json(
        os.path.join(CDVQA_ROOT, "Train_questions.json"),
        "questions"
    )

    answers = load_json(
        os.path.join(CDVQA_ROOT, "Train_answers.json"),
        "answers"
    )

    # Question lookup by question ID.
    question_by_id = {
        q["id"]: q
        for q in questions
    }

    # Answers grouped by question ID.
    answers_by_question = defaultdict(list)

    for answer_record in answers:
        answers_by_question[
            answer_record["question_id"]
        ].append(answer_record["answer"])

    # ---------------------------------------------------------
    # Merge repeated CDVQA image records by actual pair ID.
    #
    # Example:
    # several records may have:
    # file_name = 11344.png
    #
    # They all belong to the same SECOND temporal pair.
    # ---------------------------------------------------------

    pair_to_question_ids = defaultdict(set)

    for image_record in images:
        file_name = image_record["file_name"]

        pair_id = os.path.splitext(file_name)[0]

        for question_id in image_record["questions_ids"]:
            pair_to_question_ids[pair_id].add(question_id)

    rows = []

    missing_questions = []
    missing_answers = []

    # ---------------------------------------------------------
    # Create one manifest row for each question-answer example.
    # ---------------------------------------------------------

    for pair_id in sorted(pair_to_question_ids.keys()):

        question_ids = sorted(
            pair_to_question_ids[pair_id]
        )

        for question_id in question_ids:

            question = question_by_id.get(question_id)

            if question is None:
                missing_questions.append(question_id)
                continue

            answer_list = answers_by_question.get(
                question_id,
                []
            )

            if not answer_list:
                missing_answers.append(question_id)
                continue

            for answer in answer_list:

                rows.append({
                    "pair_id": pair_id,
                    "file_name": f"{pair_id}.png",

                    "t1_path": f"im1/{pair_id}.png",
                    "t2_path": f"im2/{pair_id}.png",

                    "label1_path": f"label1/{pair_id}.png",
                    "label2_path": f"label2/{pair_id}.png",

                    "question_id": question_id,
                    "question_type": question["type"],
                    "question": question["question"],
                    "answer": answer,
                })

    if not rows:
        raise RuntimeError(
            "No temporal manifest rows were created."
        )

    df = pd.DataFrame(rows)

    # Stable ordering.
    df = df.sort_values(
        ["pair_id", "question_id", "answer"],
        kind="stable"
    ).reset_index(drop=True)

    # Save manifest.
    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    questions_per_pair = (
        df.groupby("pair_id")["question_id"]
        .nunique()
    )

    question_type_counts = (
        df["question_type"]
        .value_counts()
        .sort_index()
    )

    answer_counts = (
        df["answer"]
        .value_counts()
        .sort_index()
    )

    stats = []

    stats.append(
        "SatQueryAI Temporal Manifest Statistics"
    )
    stats.append("=" * 50)

    stats.append(
        f"Unique temporal pairs: "
        f"{df['pair_id'].nunique()}"
    )

    stats.append(
        f"Manifest QA rows: "
        f"{len(df)}"
    )

    stats.append(
        f"Missing question records: "
        f"{len(missing_questions)}"
    )

    stats.append(
        f"Missing answer records: "
        f"{len(missing_answers)}"
    )

    stats.append(
        f"Minimum questions per pair: "
        f"{questions_per_pair.min()}"
    )

    stats.append(
        f"Maximum questions per pair: "
        f"{questions_per_pair.max()}"
    )

    stats.append(
        f"Mean questions per pair: "
        f"{questions_per_pair.mean():.2f}"
    )

    stats.append("")
    stats.append("Question types:")

    for question_type, count in question_type_counts.items():
        stats.append(
            f"  {question_type}: {count}"
        )

    stats.append("")
    stats.append("Top answers:")

    for answer, count in answer_counts.head(20).items():
        stats.append(
            f"  {answer}: {count}"
        )

    stats.append("")
    stats.append(f"Manifest CSV: {OUT_CSV}")
    stats.append(f"Statistics: {OUT_STATS}")

    with open(
        OUT_STATS,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(stats) + "\n"
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print("")
    print("=" * 60)
    print("TEMPORAL MANIFEST CREATED")
    print("=" * 60)

    print(
        f"Unique temporal pairs: "
        f"{df['pair_id'].nunique()}"
    )

    print(
        f"Manifest QA rows: "
        f"{len(df)}"
    )

    print(
        f"Missing question records: "
        f"{len(missing_questions)}"
    )

    print(
        f"Missing answer records: "
        f"{len(missing_answers)}"
    )

    print(
        f"Minimum questions/pair: "
        f"{questions_per_pair.min()}"
    )

    print(
        f"Maximum questions/pair: "
        f"{questions_per_pair.max()}"
    )

    print(
        f"Mean questions/pair: "
        f"{questions_per_pair.mean():.2f}"
    )

    print("")
    print("Question types:")

    for question_type, count in question_type_counts.items():
        print(
            f"  {question_type}: {count}"
        )

    print("")
    print(f"CSV: {OUT_CSV}")
    print(f"Stats: {OUT_STATS}")


if __name__ == "__main__":
    main()