from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(r"D:\SatQueryAI")

CDVQA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CDVQA"
)


FILES = [
    "Test_questions.json",
    "Test_answers.json",
    "Test_images.json",
    "Test2_questions.json",
    "Test2_answers.json",
    "Test2_images.json",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def describe(value, indent=0, max_list_items=3):
    prefix = " " * indent

    print(
        f"{prefix}Python type: {type(value).__name__}"
    )

    if isinstance(value, dict):

        print(
            f"{prefix}Dictionary keys: {list(value.keys())}"
        )

        for key, child in value.items():

            print(
                f"{prefix}  KEY = {key!r}"
            )

            print(
                f"{prefix}    type = {type(child).__name__}"
            )

            if isinstance(child, list):

                print(
                    f"{prefix}    length = {len(child)}"
                )

                for i, item in enumerate(
                    child[:max_list_items]
                ):

                    print(
                        f"{prefix}    item[{i}] "
                        f"type={type(item).__name__}"
                    )

                    if isinstance(item, dict):
                        print(
                            f"{prefix}      keys="
                            f"{list(item.keys())}"
                        )

                        for k, v in list(
                            item.items()
                        )[:20]:
                            print(
                                f"{prefix}        "
                                f"{k!r} = "
                                f"{repr(v)[:300]}"
                            )

                    else:
                        print(
                            f"{prefix}      "
                            f"value={repr(item)[:300]}"
                        )

            elif isinstance(child, dict):

                print(
                    f"{prefix}    "
                    f"nested dictionary keys="
                    f"{list(child.keys())[:30]}"
                )

            else:

                print(
                    f"{prefix}    "
                    f"value={repr(child)[:500]}"
                )

    elif isinstance(value, list):

        print(
            f"{prefix}List length: {len(value)}"
        )

        for i, item in enumerate(
            value[:max_list_items]
        ):

            print(
                f"{prefix}  item[{i}] "
                f"type={type(item).__name__}"
            )

            if isinstance(item, dict):

                print(
                    f"{prefix}    keys="
                    f"{list(item.keys())}"
                )

                for k, v in list(
                    item.items()
                )[:20]:

                    print(
                        f"{prefix}      "
                        f"{k!r} = "
                        f"{repr(v)[:300]}"
                    )

            else:

                print(
                    f"{prefix}    "
                    f"value={repr(item)[:300]}"
                )

    else:

        print(
            f"{prefix}Value: {repr(value)[:500]}"
        )


def main():

    print("=" * 76)
    print("SATQUERYAI — CDVQA TEST JSON STRUCTURE")
    print("=" * 76)

    for filename in FILES:

        path = CDVQA_ROOT / filename

        print()
        print("=" * 76)
        print(filename)
        print("=" * 76)

        if not path.exists():
            raise FileNotFoundError(path)

        data = load_json(path)

        describe(data)

    print()
    print("=" * 76)
    print("JSON STRUCTURE INSPECTION COMPLETE")
    print("=" * 76)
    print("No files were modified.")
    print("No manifests were created.")
    print("No embeddings were generated.")
    print("=" * 76)


if __name__ == "__main__":
    main()