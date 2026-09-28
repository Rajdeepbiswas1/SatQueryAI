import os
import sys
import time

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = r"D:\SatQueryAI"

INPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_raw",
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_preprocessed",
)


RGB_TO_CLASS = {
    (255, 255, 255): 0,  # background / non-change
    (0, 128, 0): 1,      # low vegetation
    (0, 255, 0): 2,      # tree
    (128, 0, 0): 3,      # building
    (0, 0, 255): 4,      # water
    (128, 128, 128): 5,  # non-vegetated ground
    (255, 0, 0): 6,      # playground
}

CLASS_NAMES = {
    0: "background",
    1: "low_vegetation",
    2: "tree",
    3: "building",
    4: "water",
    5: "non_vegetated_ground",
    6: "playground",
}


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def rgb_to_class_map(
    rgb: np.ndarray,
    path: str,
) -> np.ndarray:

    h, w, _ = rgb.shape

    result = np.full(
        (h, w),
        -1,
        dtype=np.int64,
    )

    for color, class_id in RGB_TO_CLASS.items():

        mask = np.all(
            rgb == np.array(
                color,
                dtype=np.uint8,
            ),
            axis=2,
        )

        result[mask] = class_id

    if np.any(result == -1):

        unknown = np.unique(
            rgb[result == -1],
            axis=0,
        )

        unknown = [
            tuple(int(v) for v in color)
            for color in unknown
        ]

        raise ValueError(
            f"Unknown RGB labels in {path}: "
            f"{unknown}"
        )

    return result


def image_to_tensor(
    rgb: np.ndarray,
) -> torch.Tensor:

    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    )

    return tensor.permute(
        2,
        0,
        1,
    ).contiguous()


def process_pair(pair_id: str):

    paths = {
        "t1": os.path.join(
            INPUT_ROOT,
            "im1",
            f"{pair_id}.png",
        ),
        "t2": os.path.join(
            INPUT_ROOT,
            "im2",
            f"{pair_id}.png",
        ),
        "label1": os.path.join(
            INPUT_ROOT,
            "label1",
            f"{pair_id}.png",
        ),
        "label2": os.path.join(
            INPUT_ROOT,
            "label2",
            f"{pair_id}.png",
        ),
    }

    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    t1_rgb = load_rgb(paths["t1"])
    t2_rgb = load_rgb(paths["t2"])
    label1_rgb = load_rgb(paths["label1"])
    label2_rgb = load_rgb(paths["label2"])

    expected = (512, 512, 3)

    for name, arr in [
        ("t1", t1_rgb),
        ("t2", t2_rgb),
        ("label1", label1_rgb),
        ("label2", label2_rgb),
    ]:

        if arr.shape != expected:
            raise ValueError(
                f"{pair_id}: {name} has "
                f"shape {arr.shape}; "
                f"expected {expected}"
            )

    t1 = image_to_tensor(t1_rgb)
    t2 = image_to_tensor(t2_rgb)

    label1 = rgb_to_class_map(
        label1_rgb,
        paths["label1"],
    )

    label2 = rgb_to_class_map(
        label2_rgb,
        paths["label2"],
    )

    change_mask = (
        label1 != label2
    ).astype(np.uint8)

    transition_map = (
        label1 * 10 + label2
    ).astype(np.int16)

    sample = {
        "pair_id": pair_id,

        "t1": t1,
        "t2": t2,

        "label1": torch.from_numpy(
            label1
        ).long(),

        "label2": torch.from_numpy(
            label2
        ).long(),

        "change_mask": torch.from_numpy(
            change_mask
        ).to(torch.uint8),

        "transition_map": torch.from_numpy(
            transition_map
        ).short(),

        "class_names": CLASS_NAMES,
    }

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{pair_id}.pt",
    )

    torch.save(
        sample,
        output_path,
    )


def main():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    im1_dir = os.path.join(
        INPUT_ROOT,
        "im1",
    )

    if not os.path.isdir(im1_dir):
        raise FileNotFoundError(
            f"Missing directory: {im1_dir}"
        )

    pair_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(im1_dir)
        if f.lower().endswith(".png")
    )

    total = len(pair_ids)

    print("")
    print("=" * 70)
    print("SECOND TEMPORAL DATASET PREPROCESSOR")
    print("=" * 70)

    print(
        f"Found temporal pairs: {total}"
    )

    existing = sum(
        os.path.isfile(
            os.path.join(
                OUTPUT_ROOT,
                f"{pair_id}.pt",
            )
        )
        for pair_id in pair_ids
    )

    print(
        f"Already processed: {existing}"
    )

    remaining = total - existing

    print(
        f"Remaining to process: {remaining}"
    )

    print("")

    processed = 0
    failed = 0
    start_time = time.time()

    for index, pair_id in enumerate(pair_ids, start=1):

        output_path = os.path.join(
            OUTPUT_ROOT,
            f"{pair_id}.pt",
        )

        if os.path.isfile(output_path):
            continue

        try:

            process_pair(pair_id)

            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time

                print(
                    f"Processed {processed}/"
                    f"{remaining} new pairs "
                    f"(pair {index}/{total}) "
                    f"in {elapsed:.1f}s"
                )

        except Exception as exc:

            failed += 1

            print(
                f"FAILED {pair_id}: {exc}",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time

    final_count = len([
        f
        for f in os.listdir(OUTPUT_ROOT)
        if f.lower().endswith(".pt")
    ])

    print("")
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Total source pairs: {total}"
    )

    print(
        f"Previously existing: {existing}"
    )

    print(
        f"Newly processed: {processed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"PT files now present: {final_count}"
    )

    print(
        f"Elapsed time: {elapsed:.1f}s"
    )

    if failed == 0 and final_count == total:
        print("")
        print("SUCCESS")
    else:
        print("")
        print(
            "WARNING: dataset is incomplete."
        )


if __name__ == "__main__":
    main()