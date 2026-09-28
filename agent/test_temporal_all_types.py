import pandas as pd
from types import SimpleNamespace

from agent.temporal_runtime_store import TemporalRuntimeStore
from agent.temporal_change_vqa_tool import SatQueryAITemporalChangeV3Tool


EXPECTED_HEADS = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio_types",
}


MANIFEST = (
    r"D:\SatQueryAI\data\processed\temporal_split\test_manifest.csv"
)


def main():
    print("=" * 80)
    print("SATQUERYAI — TEMPORAL V3 ALL QUESTION-TYPE SMOKE TEST")
    print("=" * 80)

    manifest = pd.read_csv(MANIFEST)
    store = TemporalRuntimeStore()
    tool = SatQueryAITemporalChangeV3Tool()

    analysis = SimpleNamespace(
        task="temporal_change",
        confidence=1.0,
    )

    results = []

    for question_type in EXPECTED_HEADS:

        rows = manifest[
            manifest["question_type"] == question_type
        ]

        if rows.empty:
            raise RuntimeError(
                f"No Test question found for type: {question_type}"
            )

        row = rows.iloc[0]

        data = store.get(
            int(row["question_id"])
        )

        result = tool(
            query=data["question"],
            analysis=analysis,
            t1=data["t1"],
            t2=data["t2"],
            diff=data["diff"],
            text_embedding=data["text_embedding"],
        )

        expected_head = EXPECTED_HEADS[
            question_type
        ]

        actual_head = result[
            "selected_head"
        ]

        passed = (
            data["question_type"] == question_type
            and actual_head == expected_head
            and result["answer"]
        )

        results.append(passed)

        print(
            f"{question_type:22s} | "
            f"head={actual_head:20s} | "
            f"answer={result['answer']:15s} | "
            f"expected={data['answer']:15s} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("=" * 80)

    if not all(results):
        raise RuntimeError(
            "At least one temporal question-type test failed."
        )

    print("RESULT: PASS")


if __name__ == "__main__":
    main()