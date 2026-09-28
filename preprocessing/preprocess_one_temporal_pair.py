import os

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = r"D:\SatQueryAI"

TEMPORAL_ROOT = os.path.join(
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

PAIR_ID = "11344"


# ------------------------------------------------------------
# SECOND RGB -> class-ID mapping
# ------------------------------------------------------------
#
# 0 = background / non-change
# 1 = low vegetation
# 2 = tree
# 3 = building
# 4 = water
# 5 = non-vegetated ground surface
# 6 = playground
#
# These IDs are OUR internal IDs. The original dataset stores
# the semantic classes as RGB colors.
# ------------------------------------------------------------

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


def load_rgb_image(path: str) -> np.ndarray:
    """
    Load an RGB PNG and return uint8 array [H, W, 3].
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        array = np.array(img)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image at {path}, got {array.shape}"
        )

    return array


def rgb_to_class_map(
    rgb: np.ndarray,
    path: str,
) -> np.ndarray:
    """
    Convert SECOND's RGB semantic label image into
    an integer class-ID map [H, W].
    """
    h, w, _ = rgb.shape

    class_map = np.full(
        (h, w),
        -1,
        dtype=np.int64,
    )

    for color, class_id in RGB_TO_CLASS.items():
        mask = np.all(
            rgb == np.array(color, dtype=np.uint8),
            axis=2,
        )
        class_map[mask] = class_id

    unknown_mask = class_map == -1

    if np.any(unknown_mask):
        unknown_colors = np.unique(
            rgb[unknown_mask],
            axis=0,
        )

        unknown_colors = [
            tuple(int(v) for v in color)
            for color in unknown_colors
        ]

        raise ValueError(
            f"Unknown label colors in {path}: "
            f"{unknown_colors}"
        )

    return class_map


def rgb_to_float_tensor(
    rgb: np.ndarray,
) -> torch.Tensor:
    """
    Convert uint8 RGB [H,W,3] to float32 tensor [3,H,W]
    in [0,1].
    """
    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    ).contiguous()

    return tensor


def main():
    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    paths = {
        "t1": os.path.join(
            TEMPORAL_ROOT,
            "im1",
            f"{PAIR_ID}.png",
        ),
        "t2": os.path.join(
            TEMPORAL_ROOT,
            "im2",
            f"{PAIR_ID}.png",
        ),
        "label1": os.path.join(
            TEMPORAL_ROOT,
            "label1",
            f"{PAIR_ID}.png",
        ),
        "label2": os.path.join(
            TEMPORAL_ROOT,
            "label2",
            f"{PAIR_ID}.png",
        ),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    t1_rgb = load_rgb_image(paths["t1"])
    t2_rgb = load_rgb_image(paths["t2"])

    label1_rgb = load_rgb_image(
        paths["label1"]
    )
    label2_rgb = load_rgb_image(
        paths["label2"]
    )

    expected_shape = (
        512,
        512,
        3,
    )

    for name, array in [
        ("T1", t1_rgb),
        ("T2", t2_rgb),
        ("label1", label1_rgb),
        ("label2", label2_rgb),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape is {array.shape}; "
                f"expected {expected_shape}"
            )

    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    t1 = rgb_to_float_tensor(t1_rgb)
    t2 = rgb_to_float_tensor(t2_rgb)

    # --------------------------------------------------------
    # Convert semantic labels
    # --------------------------------------------------------

    label1 = rgb_to_class_map(
        label1_rgb,
        paths["label1"],
    )

    label2 = rgb_to_class_map(
        label2_rgb,
        paths["label2"],
    )

    # --------------------------------------------------------
    # Derive binary semantic change mask
    # --------------------------------------------------------

    change_mask = (
        label1 != label2
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Derive transition map
    #
    # transition = class1 * 10 + class2
    #
    # Example:
    # 1 -> 3 becomes 13
    # meaning low vegetation -> building.
    # --------------------------------------------------------

    transition_map = (
        label1 * 10 + label2
    ).astype(np.int16)

    # --------------------------------------------------------
    # Convert maps to tensors
    # --------------------------------------------------------

    label1_tensor = torch.from_numpy(
        label1
    ).long()

    label2_tensor = torch.from_numpy(
        label2
    ).long()

    change_mask_tensor = torch.from_numpy(
        change_mask
    ).to(torch.uint8)

    transition_map_tensor = torch.from_numpy(
        transition_map
    ).short()

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not (
        torch.isfinite(t1).all()
        and torch.isfinite(t2).all()
    ):
        raise ValueError(
            "T1/T2 contains non-finite values."
        )

    if t1.min() < 0 or t1.max() > 1:
        raise ValueError(
            "T1 is outside [0,1]."
        )

    if t2.min() < 0 or t2.max() > 1:
        raise ValueError(
            "T2 is outside [0,1]."
        )

    if not torch.all(
        (label1_tensor >= 0)
        & (label1_tensor <= 6)
    ):
        raise ValueError(
            "Invalid class ID in label1."
        )

    if not torch.all(
        (label2_tensor >= 0)
        & (label2_tensor <= 6)
    ):
        raise ValueError(
            "Invalid class ID in label2."
        )

    # --------------------------------------------------------
    # Build output object
    # --------------------------------------------------------

    sample = {
        "pair_id": PAIR_ID,

        "t1": t1,
        "t2": t2,

        "label1": label1_tensor,
        "label2": label2_tensor,

        "change_mask": change_mask_tensor,
        "transition_map": transition_map_tensor,

        "class_names": CLASS_NAMES,
    }

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{PAIR_ID}.pt",
    )

    torch.save(
        sample,
        output_path,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("")
    print("=" * 60)
    print("TEMPORAL ONE-PAIR PREPROCESSOR")
    print("=" * 60)

    print(f"Pair: {PAIR_ID}")

    print(
        f"T1 tensor: "
        f"{tuple(t1.shape)}, "
        f"{t1.dtype}, "
        f"range=({t1.min().item():.4f}, "
        f"{t1.max().item():.4f})"
    )

    print(
        f"T2 tensor: "
        f"{tuple(t2.shape)}, "
        f"{t2.dtype}, "
        f"range=({t2.min().item():.4f}, "
        f"{t2.max().item():.4f})"
    )

    print(
        f"Label1: "
        f"{tuple(label1_tensor.shape)}, "
        f"{label1_tensor.dtype}"
    )

    print(
        f"Label2: "
        f"{tuple(label2_tensor.shape)}, "
        f"{label2_tensor.dtype}"
    )

    changed = int(
        change_mask_tensor.sum().item()
    )

    total = int(
        change_mask_tensor.numel()
    )

    print(
        f"Changed pixels: "
        f"{changed}/{total}"
    )

    print(
        f"Change percentage: "
        f"{changed / total * 100:.2f}%"
    )

    print("")
    print("T1 class counts:")

    for class_id, name in CLASS_NAMES.items():
        count = int(
            (label1_tensor == class_id)
            .sum()
            .item()
        )

        if count > 0:
            print(
                f"  {class_id}: "
                f"{name}: {count}"
            )

    print("")
    print("T2 class counts:")

    for class_id, name in CLASS_NAMES.items():
        count = int(
            (label2_tensor == class_id)
            .sum()
            .item()
        )

        if count > 0:
            print(
                f"  {class_id}: "
                f"{name}: {count}"
            )

    print("")
    print(f"Saved: {output_path}")
    print("")
    print("SUCCESS")


if __name__ == "__main__":
    main()