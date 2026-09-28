from pathlib import Path
import json


ROOT = Path(r"D:\SatQueryAI\data\raw\CDVQA")

SPLITS = [
    "Train",
    "Val",
    "Test",
    "Test2",
]


def load_images(split: str):
    path = ROOT / f"{split}_images.json"

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data["images"]


def get_filenames(images):
    return {
        str(image["file_name"])
        for image in images
    }


def main() -> None:

    print("=" * 80)
    print("SATQUERYAI — CDVQA PAIR PARTITION VERIFICATION")
    print("=" * 80)

    split_sets = {}

    for split in SPLITS:

        images = load_images(split)

        filenames = get_filenames(images)

        split_sets[split] = filenames

        print()
        print(
            f"{split:7s}"
            f" image records = {len(images):6d}"
            f" unique filenames = {len(filenames):4d}"
        )

    train = split_sets["Train"]
    val = split_sets["Val"]
    test = split_sets["Test"]
    test2 = split_sets["Test2"]

    print()
    print("=" * 80)
    print("OVERLAP CHECKS")
    print("=" * 80)

    checks = {
        "Train ∩ Val": train & val,
        "Train ∩ Test": train & test,
        "Train ∩ Test2": train & test2,
        "Val ∩ Test": val & test,
        "Val ∩ Test2": val & test2,
        "Test ∩ Test2": test & test2,
    }

    for name, overlap in checks.items():

        print(
            f"{name:18s}: {len(overlap):4d}"
        )

    print()
    print("=" * 80)
    print("PARTITION COUNTS")
    print("=" * 80)

    train_val_test = (
        train
        | val
        | test
    )

    all_four = (
        train
        | val
        | test
        | test2
    )

    print(
        "Train pairs:",
        len(train)
    )

    print(
        "Val pairs:",
        len(val)
    )

    print(
        "Test pairs:",
        len(test)
    )

    print(
        "Test2 pairs:",
        len(test2)
    )

    print()
    print(
        "Train + Val + Test:",
        len(train_val_test)
    )

    print(
        "All four unique:",
        len(all_four)
    )

    print()
    print("=" * 80)
    print("TEST / TEST2 IDENTITY")
    print("=" * 80)

    print(
        "Test == Test2:",
        test == test2
    )

    only_test = test - test2
    only_test2 = test2 - test

    print(
        "Only in Test:",
        len(only_test)
    )

    print(
        "Only in Test2:",
        len(only_test2)
    )

    print()
    print("=" * 80)
    print("EXPECTED STRUCTURE CHECK")
    print("=" * 80)

    expected_train = 1600
    expected_val = 400
    expected_test = 968
    expected_total = 2968

    passed = True

    if len(train) != expected_train:
        print(
            f"FAIL: Train expected {expected_train}, "
            f"found {len(train)}"
        )
        passed = False

    if len(val) != expected_val:
        print(
            f"FAIL: Val expected {expected_val}, "
            f"found {len(val)}"
        )
        passed = False

    if len(test) != expected_test:
        print(
            f"FAIL: Test expected {expected_test}, "
            f"found {len(test)}"
        )
        passed = False

    if len(train_val_test) != expected_total:
        print(
            f"FAIL: Train+Val+Test expected "
            f"{expected_total}, "
            f"found {len(train_val_test)}"
        )
        passed = False

    if test != test2:
        print(
            "FAIL: Test and Test2 are not "
            "the same image-pair set."
        )
        passed = False

    if any(checks.values()):
        print(
            "FAIL: Unexpected pair overlap detected."
        )
        passed = False

    print()

    if passed:
        print(
            "CDVQA PAIR PARTITION VERIFIED"
        )
    else:
        print(
            "CDVQA PAIR PARTITION VERIFICATION FAILED"
        )
        raise RuntimeError(
            "Pair partition checks failed."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()