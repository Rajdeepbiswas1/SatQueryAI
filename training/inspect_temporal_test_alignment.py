from __future__ import annotations

import json
from pathlib import Path

import torch


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


def normalize_pair_id(value):
    """
    Convert values such as:
        7308
        '7308'
        '07308.png'
        '07308'
    into:
        '07308'
    """
    s = str(value).strip()

    if s.lower().endswith(".png"):
        s = s[:-4]

    try:
        return f"{int(float(s)):05d}"
    except Exception:
        return s


def get_questions(data):
    if not isinstance(data, dict) or "questions" not in data:
        raise RuntimeError("Expected top-level 'questions' key.")
    return data["questions"]


def get_images(data):
    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Expected top-level 'images' key.")
    return data["images"]


def main():

    print("=" * 76)
    print("SATQUERYAI — CDVQA TEST / TEMPORAL FEATURE ALIGNMENT")
    print("=" * 76)

    # ------------------------------------------------------------
    # 1. Load Test / Test2
    # ------------------------------------------------------------
    test_questions = get_questions(
        load_json(CDVQA_ROOT / "Test_questions.json")
    )

    test_images = get_images(
        load_json(CDVQA_ROOT / "Test_images.json")
    )

    test2_questions = get_questions(
        load_json(CDVQA_ROOT / "Test2_questions.json")
    )

    test2_images = get_images(
        load_json(CDVQA_ROOT / "Test2_images.json")
    )

    print("\n[1] RAW COUNTS")

    print("Test questions :", len(test_questions))
    print("Test images    :", len(test_images))

    print("Test2 questions:", len(test2_questions))
    print("Test2 images   :", len(test2_images))

    # ------------------------------------------------------------
    # 2. Build question -> image mapping
    # ------------------------------------------------------------
    print("\n[2] QUESTION / IMAGE MAPPING")

    image_by_id = {
        img["id"]: img
        for img in test_images
    }

    test_question_image_ids = set()
    test_question_file_names = set()

    missing_test_images = []

    for q in test_questions:

        img_id = q["img_id"]

        test_question_image_ids.add(img_id)

        if img_id not in image_by_id:
            missing_test_images.append(img_id)
            continue

        file_name = image_by_id[img_id]["file_name"]

        test_question_file_names.add(
            normalize_pair_id(file_name)
        )

    print(
        "Test unique img_ids:",
        len(test_question_image_ids)
    )

    print(
        "Test unique file names:",
        len(test_question_file_names)
    )

    print(
        "Missing Test image references:",
        len(missing_test_images)
    )

    # ------------------------------------------------------------
    # 3. Do the same for Test2
    # ------------------------------------------------------------
    image2_by_id = {
        img["id"]: img
        for img in test2_images
    }

    test2_question_image_ids = set()
    test2_question_file_names = set()

    missing_test2_images = []

    for q in test2_questions:

        img_id = q["img_id"]

        test2_question_image_ids.add(img_id)

        if img_id not in image2_by_id:
            missing_test2_images.append(img_id)
            continue

        file_name = image2_by_id[img_id]["file_name"]

        test2_question_file_names.add(
            normalize_pair_id(file_name)
        )

    print(
        "Test2 unique img_ids:",
        len(test2_question_image_ids)
    )

    print(
        "Test2 unique file names:",
        len(test2_question_file_names)
    )

    print(
        "Missing Test2 image references:",
        len(missing_test2_images)
    )

    # ------------------------------------------------------------
    # 4. Compare Test and Test2 image sets
    # ------------------------------------------------------------
    print("\n[3] TEST vs TEST2 IMAGE SET")

    test_and_test2_common = (
        test_question_file_names
        & test2_question_file_names
    )

    test_only = (
        test_question_file_names
        - test2_question_file_names
    )

    test2_only = (
        test2_question_file_names
        - test_question_file_names
    )

    print(
        "Common file names:",
        len(test_and_test2_common)
    )

    print(
        "Test-only file names:",
        len(test_only)
    )

    print(
        "Test2-only file names:",
        len(test2_only)
    )

    print(
        "Example common IDs:",
        sorted(test_and_test2_common)[:20]
    )

    # ------------------------------------------------------------
    # 5. Load temporal feature IDs
    # ------------------------------------------------------------
    print("\n[4] TEMPORAL FEATURE CACHE")

    feature_files = sorted(
        FEATURE_ROOT.glob("*.pt")
    )

    print(
        "Feature shards:",
        len(feature_files)
    )

    feature_pair_ids = set()

    for path in feature_files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        ids = data.get("pair_ids")

        if ids is None:
            raise RuntimeError(
                f"Missing pair_ids in {path}"
            )

        for pid in ids:
            feature_pair_ids.add(
                normalize_pair_id(pid)
            )

    print(
        "Unique feature pairs:",
        len(feature_pair_ids)
    )

    # ------------------------------------------------------------
    # 6. Alignment
    # ------------------------------------------------------------
    print("\n[5] ALIGNMENT")

    test_feature_intersection = (
        test_question_file_names
        & feature_pair_ids
    )

    test2_feature_intersection = (
        test2_question_file_names
        & feature_pair_ids
    )

    print(
        "Test file names matching feature cache:",
        len(test_feature_intersection)
    )

    print(
        "Test2 file names matching feature cache:",
        len(test2_feature_intersection)
    )

    test_missing_from_features = (
        test_question_file_names
        - feature_pair_ids
    )

    test2_missing_from_features = (
        test2_question_file_names
        - feature_pair_ids
    )

    print(
        "Test names missing from feature cache:",
        len(test_missing_from_features)
    )

    print(
        "Test2 names missing from feature cache:",
        len(test2_missing_from_features)
    )

    # ------------------------------------------------------------
    # 7. Compare against the 968-pair expectation
    # ------------------------------------------------------------
    print("\n[6] 968-PAIR CHECK")

    print(
        "Expected held-out CDVQA pairs:",
        968
    )

    if len(test_feature_intersection) == 968:
        print(
            "Test intersection = 968: PASS"
        )
    else:
        print(
            "Test intersection = 968: NOT YET CONFIRMED"
        )

    if len(test2_feature_intersection) == 968:
        print(
            "Test2 intersection = 968: PASS"
        )
    else:
        print(
            "Test2 intersection = 968: NOT YET CONFIRMED"
        )

    # ------------------------------------------------------------
    # 8. Final sample IDs
    # ------------------------------------------------------------
    print("\n[7] SAMPLE MATCHED PAIR IDS")

    print(
        "Test:",
        sorted(test_feature_intersection)[:30]
    )

    print(
        "Test2:",
        sorted(test2_feature_intersection)[:30]
    )

    print("\n" + "=" * 76)
    print("TEST / FEATURE ALIGNMENT INSPECTION COMPLETE")
    print("=" * 76)
    print("No files were modified.")
    print("No manifests were created.")
    print("No embeddings were generated.")
    print("=" * 76)


if __name__ == "__main__":
    main()