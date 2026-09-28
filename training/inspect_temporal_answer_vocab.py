from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_manifest.csv"
)


EXPECTED_TYPES = [
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "smallest_change",
    "largest_change",
    "change_to_what",
    "change_ratio",
    "change_ratio_types",
]


def main() -> None:
    print("=" * 72)
    print("SATQUERYAI — TEMPORAL ANSWER VOCABULARY INSPECTION")
    print("=" * 72)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    required = {
        "pair_id",
        "question_id",
        "question_type",
        "question",
        "answer",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    print()
    print("Manifest:")
    print(MANIFEST)

    print()
    print("Total rows:", len(df))
    print(
        "Unique pairs:",
        df["pair_id"].nunique(),
    )
    print(
        "Unique questions:",
        df["question_id"].nunique(),
    )

    print()
    print("=" * 72)
    print("QUESTION TYPE DISTRIBUTION")
    print("=" * 72)

    type_counts = df["question_type"].value_counts()

    for question_type in EXPECTED_TYPES:
        count = int(
            type_counts.get(question_type, 0)
        )

        print(
            f"{question_type:22s}: {count}"
        )

    unexpected_types = sorted(
        set(df["question_type"].astype(str))
        - set(EXPECTED_TYPES)
    )

    if unexpected_types:
        print()
        print(
            "WARNING — unexpected question types:"
        )

        for value in unexpected_types:
            print(" ", value)

    print()
    print("=" * 72)
    print("EXACT ANSWER VOCABULARY")
    print("=" * 72)

    for question_type in EXPECTED_TYPES:

        subset = df[
            df["question_type"].astype(str)
            == question_type
        ].copy()

        print()
        print(
            f"[{question_type}]"
        )

        if subset.empty:
            print("  No rows.")
            continue

        answers = (
            subset["answer"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        counter = Counter(answers)

        print(
            "  Rows:",
            len(subset),
        )

        print(
            "  Unique answers:",
            len(counter),
        )

        for answer, count in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            print(
                f"    {repr(answer):45s}"
                f" -> {count}"
            )

    print()
    print("=" * 72)
    print("GLOBAL ANSWER SUMMARY")
    print("=" * 72)

    global_answers = (
        df["answer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(
        "Unique answers globally:",
        global_answers.nunique(),
    )

    print()
    print("Empty answers:")

    empty_count = int(
        (global_answers == "").sum()
    )

    print(
        " ",
        empty_count,
    )

    print()
    print("=" * 72)
    print("ANSWER VOCABULARY INSPECTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()