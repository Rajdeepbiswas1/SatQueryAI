import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = r"D:\SatQueryAI"

TEMPORAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_raw",
)

PAIR_ID = "11344"


LABEL_COLORS = {
    (255, 255, 255): "background",
    (0, 128, 0): "low vegetation",
    (0, 255, 0): "tree",
    (128, 0, 0): "building",
    (0, 0, 255): "water",
    (128, 128, 128): "non-vegetated ground",
    (255, 0, 0): "playground",
}


def load_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def main():
    paths = {
        "T1": os.path.join(
            TEMPORAL_ROOT, "im1", f"{PAIR_ID}.png"
        ),
        "T2": os.path.join(
            TEMPORAL_ROOT, "im2", f"{PAIR_ID}.png"
        ),
        "Label T1": os.path.join(
            TEMPORAL_ROOT, "label1", f"{PAIR_ID}.png"
        ),
        "Label T2": os.path.join(
            TEMPORAL_ROOT, "label2", f"{PAIR_ID}.png"
        ),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    t1 = load_rgb(paths["T1"])
    t2 = load_rgb(paths["T2"])
    label1 = load_rgb(paths["Label T1"])
    label2 = load_rgb(paths["Label T2"])

    # Derive semantic change mask.
    change_mask = np.any(
        label1 != label2,
        axis=2,
    )

    # Create figure.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes[0, 0].imshow(t1)
    axes[0, 0].set_title("T1 — im1")

    axes[0, 1].imshow(t2)
    axes[0, 1].set_title("T2 — im2")

    axes[0, 2].imshow(change_mask, cmap="gray")
    axes[0, 2].set_title(
        f"Semantic Change ({change_mask.mean() * 100:.2f}%)"
    )

    axes[1, 0].imshow(label1)
    axes[1, 0].set_title("T1 Semantic Label")

    axes[1, 1].imshow(label2)
    axes[1, 1].set_title("T2 Semantic Label")

    # Overlay change on T2.
    overlay = t2.copy()
    overlay[change_mask] = (
        255,
        0,
        0,
    )

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(
        "T2 with Changed Pixels"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        f"temporal_pair_{PAIR_ID}_visualization.png",
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("")
    print("=" * 60)
    print("TEMPORAL VISUALIZATION")
    print("=" * 60)
    print(f"Pair: {PAIR_ID}")
    print(f"Changed pixels: {change_mask.sum()}")
    print(f"Total pixels: {change_mask.size}")
    print(
        f"Change percentage: "
        f"{change_mask.mean() * 100:.2f}%"
    )
    print(f"Saved: {output_path}")
    print("SUCCESS")


if __name__ == "__main__":
    main()