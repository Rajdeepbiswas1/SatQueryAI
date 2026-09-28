import os
import random
from collections import Counter

import numpy as np
from PIL import Image


PROJECT_ROOT = r"D:\SatQueryAI"

TEMPORAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_raw",
)

SEED = 42
NUM_SAMPLES = 10


# SECOND semantic-label RGB mapping.
CLASS_COLORS = {
    (255, 255, 255): "background",
    (0, 128, 0): "low_vegetation",
    (0, 255, 0): "tree",
    (128, 0, 0): "building",
    (0, 0, 255): "water",
    (128, 128, 128): "non_vegetated_ground",
    (255, 0, 0): "playground",
}


def load_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def get_class_counts(label):
    pixels = label.reshape(-1, 3)

    counts = Counter(
        map(tuple, pixels)
    )

    result = {}

    for rgb, count in counts.items():
        name = CLASS_COLORS.get(
            rgb,
            f"UNKNOWN_RGB_{rgb}",
        )
        result[name] = count

    return result


def summarize_change(label1, label2):
    same = np.all(
        label1 == label2,
        axis=2,
    )

    changed_pixels = int(
        np.count_nonzero(~same)
    )

    total_pixels = int(
        same.size
    )

    change_percent = (
        changed_pixels / total_pixels * 100.0
    )

    # Transition counts:
    # (T1 class RGB, T2 class RGB) -> pixels
    t1 = label1.reshape(-1, 3)
    t2 = label2.reshape(-1, 3)

    transitions = Counter(
        zip(
            map(tuple, t1),
            map(tuple, t2),
        )
    )

    # Remove unchanged transitions.
    changed_transitions = {
        (a, b): count
        for (a, b), count in transitions.items()
        if a != b
    }

    changed_transitions = sorted(
        changed_transitions.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return (
        changed_pixels,
        total_pixels,
        change_percent,
        changed_transitions,
    )


def validate_pair(pair_id):
    paths = {
        "im1": os.path.join(
            TEMPORAL_ROOT,
            "im1",
            f"{pair_id}.png",
        ),
        "im2": os.path.join(
            TEMPORAL_ROOT,
            "im2",
            f"{pair_id}.png",
        ),
        "label1": os.path.join(
            TEMPORAL_ROOT,
            "label1",
            f"{pair_id}.png",
        ),
        "label2": os.path.join(
            TEMPORAL_ROOT,
            "label2",
            f"{pair_id}.png",
        ),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    im1 = load_rgb(paths["im1"])
    im2 = load_rgb(paths["im2"])
    label1 = load_rgb(paths["label1"])
    label2 = load_rgb(paths["label2"])

    # -------------------------
    # Shape checks
    # -------------------------

    expected_shape = (512, 512, 3)

    for name, arr in [
        ("im1", im1),
        ("im2", im2),
        ("label1", label1),
        ("label2", label2),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{pair_id}: {name} has "
                f"shape {arr.shape}, "
                f"expected {expected_shape}"
            )

    # -------------------------
    # Label validation
    # -------------------------

    valid_colors = set(CLASS_COLORS.keys())

    for name, label in [
        ("label1", label1),
        ("label2", label2),
    ]:
        unique_colors = set(
            map(
                tuple,
                np.unique(
                    label.reshape(-1, 3),
                    axis=0,
                ),
            )
        )

        unknown = (
            unique_colors - valid_colors
        )

        if unknown:
            raise ValueError(
                f"{pair_id}: {name} contains "
                f"unknown RGB values: {sorted(unknown)}"
            )

    # -------------------------
    # Change statistics
    # -------------------------

    (
        changed_pixels,
        total_pixels,
        change_percent,
        transitions,
    ) = summarize_change(
        label1,
        label2,
    )

    return {
        "pair_id": pair_id,
        "image_shape": im1.shape,
        "label_shape": label1.shape,
        "t1_classes": get_class_counts(label1),
        "t2_classes": get_class_counts(label2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percent": change_percent,
        "changed_transitions": transitions,
    }


def main():
    im1_dir = os.path.join(
        TEMPORAL_ROOT,
        "im1",
    )

    filenames = [
        f
        for f in os.listdir(im1_dir)
        if f.lower().endswith(".png")
    ]

    pair_ids = sorted(
        os.path.splitext(f)[0]
        for f in filenames
    )

    if len(pair_ids) < NUM_SAMPLES:
        raise RuntimeError(
            f"Only {len(pair_ids)} pairs found."
        )

    rng = random.Random(SEED)

    selected = rng.sample(
        pair_ids,
        NUM_SAMPLES,
    )

    print("")
    print("=" * 70)
    print("TEMPORAL SAMPLE VERIFICATION")
    print("=" * 70)

    print(
        f"Available pairs: {len(pair_ids)}"
    )

    print(
        f"Random samples: {NUM_SAMPLES}"
    )

    print(
        f"Random seed: {SEED}"
    )

    print("")

    total_changed = 0

    for pair_id in selected:

        result = validate_pair(
            pair_id
        )

        print(
            "-" * 70
        )

        print(
            f"Pair: {pair_id}"
        )

        print(
            f"Image shape: "
            f"{result['image_shape']}"
        )

        print(
            f"Label shape: "
            f"{result['label_shape']}"
        )

        print(
            f"Changed pixels: "
            f"{result['changed_pixels']}/"
            f"{result['total_pixels']}"
        )

        print(
            f"Change percentage: "
            f"{result['change_percent']:.2f}%"
        )

        total_changed += (
            result["changed_pixels"]
        )

        print("T1 classes:")

        for name, count in sorted(
            result["t1_classes"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(
                f"  {name}: {count}"
            )

        print("T2 classes:")

        for name, count in sorted(
            result["t2_classes"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(
                f"  {name}: {count}"
            )

        print("Top changed transitions:")

        for (rgb1, rgb2), count in (
            result["changed_transitions"][:10]
        ):
            name1 = CLASS_COLORS[rgb1]
            name2 = CLASS_COLORS[rgb2]

            print(
                f"  {name1} -> {name2}: "
                f"{count} pixels"
            )

    print("")
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    print(
        "All sampled temporal pairs passed "
        "file, shape, and label checks."
    )


if __name__ == "__main__":
    main()