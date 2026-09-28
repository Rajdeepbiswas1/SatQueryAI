from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import rasterio
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

AUDIT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
)

COMPONENT_FILE = (
    AUDIT_DIR
    / "component_selection_results.csv"
)

ROW_AUDIT_FILE = (
    AUDIT_DIR
    / "reference_row_clc_audit.csv"
)

MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

PREPROCESSED_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_component_visuals"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

K = 3

IMAGE_SIZE = 120

STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8,
)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(text: str) -> str:

    text = str(text)

    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    )

    return text[:120]


def parse_codes(
    text: str,
) -> list[int]:

    try:
        value = json.loads(
            str(text)
        )
    except json.JSONDecodeError:

        value = [
            int(x)
            for x in str(text)
            .strip("[]")
            .split(",")
            if x.strip()
        ]

    return [
        int(x)
        for x in value
    ]


def find_map(
    patch_id: str,
) -> Path:

    matches = list(
        MAP_ROOT.rglob(
            f"{patch_id}_reference_map.tif"
        )
    )

    if len(matches) == 0:

        raise FileNotFoundError(
            f"Reference map not found:\n"
            f"{patch_id}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple reference maps found:\n"
            f"{patch_id}\n"
            f"{matches}"
        )

    return matches[0]


def percentile_stretch(
    channel: np.ndarray,
) -> np.ndarray:

    channel = np.asarray(
        channel,
        dtype=np.float32,
    )

    finite = np.isfinite(
        channel
    )

    if not finite.any():

        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    values = channel[
        finite
    ]

    low = np.percentile(
        values,
        2,
    )

    high = np.percentile(
        values,
        98,
    )

    if high <= low:

        low = float(
            values.min()
        )

        high = float(
            values.max()
        )

    if high <= low:

        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    result = (
        channel - low
    ) / (
        high - low
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def load_rgb(
    patch_id: str,
) -> np.ndarray:

    path = (
        PREPROCESSED_ROOT
        / f"{patch_id}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Preprocessed tensor not found:\n"
            f"{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    image = data[
        "image"
    ]

    if not isinstance(
        image,
        torch.Tensor,
    ):

        raise TypeError(
            f"Image is not a Tensor:\n"
            f"{type(image)}"
        )

    image = image.float().numpy()

    if image.shape != (
        14,
        120,
        120,
    ):

        raise ValueError(
            f"Unexpected image shape:\n"
            f"{image.shape}"
        )

    channel_names = data.get(
        "channel_names"
    )

    # --------------------------------------------------------
    # Find B04/B03/B02 using saved channel names.
    # --------------------------------------------------------

    if channel_names is not None:

        channel_names = [
            str(x)
            for x in channel_names
        ]

        lookup = {
            name.upper(): idx
            for idx, name
            in enumerate(channel_names)
        }

        red_idx = lookup.get(
            "B04"
        )

        green_idx = lookup.get(
            "B03"
        )

        blue_idx = lookup.get(
            "B02"
        )

    else:

        red_idx = None
        green_idx = None
        blue_idx = None

    # --------------------------------------------------------
    # Fallback to the known 14-channel ordering:
    #
    # VH,VV,B01,B02,B03,B04,B05,B06,B07,B08,B09,B11,B12,B8A
    #
    # B02 = 3
    # B03 = 4
    # B04 = 5
    # --------------------------------------------------------

    if (
        red_idx is None
        or green_idx is None
        or blue_idx is None
    ):

        blue_idx = 3
        green_idx = 4
        red_idx = 5

    red = percentile_stretch(
        image[red_idx]
    )

    green = percentile_stretch(
        image[green_idx]
    )

    blue = percentile_stretch(
        image[blue_idx]
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    return rgb


def normalize_bbox(
    row: pd.Series,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def bbox_pixels(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    int,
    int,
    int,
    int,
]:

    x1, y1, x2, y2 = bbox

    px1 = int(
        round(
            x1 * IMAGE_SIZE
        )
    )

    py1 = int(
        round(
            y1 * IMAGE_SIZE
        )
    )

    px2 = int(
        round(
            x2 * IMAGE_SIZE
        )
    )

    py2 = int(
        round(
            y2 * IMAGE_SIZE
        )
    )

    px1 = max(
        0,
        min(
            IMAGE_SIZE,
            px1,
        ),
    )

    py1 = max(
        0,
        min(
            IMAGE_SIZE,
            py1,
        ),
    )

    px2 = max(
        0,
        min(
            IMAGE_SIZE,
            px2,
        ),
    )

    py2 = max(
        0,
        min(
            IMAGE_SIZE,
            py2,
        ),
    )

    return (
        px1,
        py1,
        px2,
        py2,
    )


def load_reference_map(
    patch_id: str,
) -> np.ndarray:

    path = find_map(
        patch_id
    )

    with rasterio.open(
        path
    ) as src:

        clc_map = src.read(
            1
        )

    if clc_map.shape != (
        120,
        120,
    ):

        raise ValueError(
            f"Unexpected CLC map shape:\n"
            f"{clc_map.shape}"
        )

    return clc_map


def get_components(
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
]:

    labels, count = (
        ndimage.label(
            mask,
            structure=STRUCTURE,
        )
    )

    return (
        labels,
        int(count),
    )


def largest_component(
    labels: np.ndarray,
) -> np.ndarray:

    counts = np.bincount(
        labels.ravel()
    )

    if len(counts) <= 1:

        return np.zeros_like(
            labels,
            dtype=bool,
        )

    component = (
        int(
            np.argmax(
                counts[1:]
            )
        )
        + 1
    )

    return (
        labels
        == component
    )


def smallest_component(
    labels: np.ndarray,
) -> np.ndarray:

    counts = np.bincount(
        labels.ravel()
    )

    if len(counts) <= 1:

        return np.zeros_like(
            labels,
            dtype=bool,
        )

    component = (
        int(
            np.argmin(
                counts[1:]
            )
        )
        + 1
    )

    return (
        labels
        == component
    )


def best_oracle_component(
    labels: np.ndarray,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    np.ndarray | None,
    float,
]:

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Exact normalized bbox area.
    # --------------------------------------------------------

    bbox_area = (
        max(
            0.0,
            x2 - x1,
        )
        * max(
            0.0,
            y2 - y1,
        )
    )

    # --------------------------------------------------------
    # Pixel weights for bbox intersection.
    # --------------------------------------------------------

    pixel_x0 = np.linspace(
        0.0,
        1.0,
        IMAGE_SIZE + 1,
    )

    pixel_x1 = pixel_x0[1:]

    pixel_y0 = np.linspace(
        0.0,
        1.0,
        IMAGE_SIZE + 1,
    )

    pixel_y1 = pixel_y0[1:]

    x_overlap = np.maximum(
        0.0,
        np.minimum(
            pixel_x1,
            x2,
        )
        - np.maximum(
            pixel_x0[:-1],
            x1,
        ),
    )

    y_overlap = np.maximum(
        0.0,
        np.minimum(
            pixel_y1,
            y2,
        )
        - np.maximum(
            pixel_y0[:-1],
            y1,
        ),
    )

    weights = (
        y_overlap[:, None]
        * x_overlap[None, :]
    )

    best_mask = None
    best_iou = -1.0

    component_ids = np.unique(
        labels
    )

    component_ids = component_ids[
        component_ids > 0
    ]

    for component_id in component_ids:

        mask = (
            labels
            == component_id
        )

        component_area = (
            float(
                mask.sum()
            )
            / (
                IMAGE_SIZE
                * IMAGE_SIZE
            )
        )

        intersection = float(
            weights[
                mask
            ].sum()
        )

        union = (
            component_area
            + bbox_area
            - intersection
        )

        if union <= 0:

            iou = 0.0

        else:

            iou = (
                intersection
                / union
            )

        if iou > best_iou:

            best_iou = float(
                iou
            )

            best_mask = mask

    return (
        best_mask,
        best_iou,
    )


def draw_bbox(
    ax,
    bbox,
    label: str = "GT bbox",
) -> None:

    x1, y1, x2, y2 = bbox

    rect = plt.Rectangle(
        (
            x1 * IMAGE_SIZE,
            y1 * IMAGE_SIZE,
        ),
        (x2 - x1) * IMAGE_SIZE,
        (y2 - y1) * IMAGE_SIZE,
        fill=False,
        linewidth=2.5,
        edgecolor="red",
        label=label,
    )

    ax.add_patch(
        rect
    )


def overlay_mask(
    ax,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> None:

    overlay = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            4,
        ),
        dtype=np.float32,
    )

    overlay[:, :, 0] = 1.0
    overlay[:, :, 3] = (
        mask.astype(
            np.float32
        )
        * alpha
    )

    ax.imshow(
        overlay
    )


# ============================================================
# CHOOSE REPRESENTATIVE EXAMPLES
# ============================================================

def choose_examples(
    components: pd.DataFrame,
) -> list[tuple[str, pd.Series]]:

    df = components[
        components["k"] == K
    ].copy()

    examples = []

    # --------------------------------------------------------
    # LARGEST QUERIES
    # --------------------------------------------------------

    largest = df[
        df["comparator"] == "largest"
    ].copy()

    if len(largest) > 0:

        # Best deterministic largest-component result.
        row = largest.sort_values(
            "selected_iou",
            ascending=False,
        ).iloc[0]

        examples.append(
            (
                "largest_best",
                row,
            )
        )

        # Worst deterministic largest-component result.
        row = largest.sort_values(
            "selected_iou",
            ascending=True,
        ).iloc[0]

        examples.append(
            (
                "largest_worst",
                row,
            )
        )

        # Largest gap between the best available component
        # and the component selected by the deterministic rule.
        largest[
            "selection_gap"
        ] = (
            largest[
                "oracle_best_iou"
            ]
            - largest[
                "selected_iou"
            ]
        )

        row = largest.sort_values(
            "selection_gap",
            ascending=False,
        ).iloc[0]

        examples.append(
            (
                "largest_big_selection_gap",
                row,
            )
        )

    # --------------------------------------------------------
    # SMALLEST QUERIES
    # --------------------------------------------------------

    smallest = df[
        df["comparator"] == "smallest"
    ].copy()

    if len(smallest) > 0:

        # Best deterministic smallest-component result.
        row = smallest.sort_values(
            "selected_iou",
            ascending=False,
        ).iloc[0]

        examples.append(
            (
                "smallest_best",
                row,
            )
        )

        # Worst deterministic smallest-component result.
        row = smallest.sort_values(
            "selected_iou",
            ascending=True,
        ).iloc[0]

        examples.append(
            (
                "smallest_worst",
                row,
            )
        )

    # --------------------------------------------------------
    # DIFFICULT PERMANENT-CROPS CASE
    # --------------------------------------------------------

    permanent = largest[
        largest[
            "semantic_family"
        ]
        == "permanent crops"
    ].copy()

    if len(permanent) > 0:

        row = permanent.sort_values(
            "selected_iou",
            ascending=True,
        ).iloc[0]

        examples.append(
            (
                "permanent_crops_hard",
                row,
            )
        )

    return examples

# ============================================================
# CREATE ONE VISUAL
# ============================================================

def create_visual(
    name: str,
    component_row: pd.Series,
    row_lookup: dict[int, pd.Series],
) -> Path:

    original_row_index = int(
        component_row[
            "original_row_index"
        ]
    )

    if original_row_index not in row_lookup:

        raise RuntimeError(
            "Could not find matching row-audit entry:\n"
            f"{original_row_index}"
        )

    audit_row = row_lookup[
        original_row_index
    ]

    patch_id = str(
        component_row[
            "patch_id"
        ]
    )

    phrase = str(
        component_row[
            "reference_phrase"
        ]
    )

    comparator = str(
        component_row[
            "comparator"
        ]
    )

    candidate_codes = parse_codes(
        component_row[
            "candidate_clc_codes"
        ]
    )

    bbox = normalize_bbox(
        audit_row
    )

    clc_map = load_reference_map(
        patch_id
    )

    rgb = load_rgb(
        patch_id
    )

    candidate_mask = np.isin(
        clc_map,
        candidate_codes,
    )

    labels, count = (
        get_components(
            candidate_mask
        )
    )

    largest_mask = (
        largest_component(
            labels
        )
    )

    smallest_mask = (
        smallest_component(
            labels
        )
    )

    oracle_mask, oracle_iou = (
        best_oracle_component(
            labels,
            bbox,
        )
    )

    # --------------------------------------------------------
    # Create figure.
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 5),
    )

    # --------------------------------------------------------
    # Panel 1 — Satellite RGB.
    # --------------------------------------------------------

    ax = axes[0]

    ax.imshow(
        rgb
    )

    draw_bbox(
        ax,
        bbox,
    )

    ax.set_title(
        "Satellite image\nGT bbox"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Panel 2 — Candidate CLC mask.
    # --------------------------------------------------------

    ax = axes[1]

    ax.imshow(
        candidate_mask,
        cmap="gray",
    )

    draw_bbox(
        ax,
        bbox,
    )

    ax.set_title(
        "CLC candidate mask\n"
        f"codes={candidate_codes}"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Panel 3 — Largest component.
    # --------------------------------------------------------

    ax = axes[2]

    ax.imshow(
        rgb
    )

    overlay_mask(
        ax,
        largest_mask,
        alpha=0.45,
    )

    draw_bbox(
        ax,
        bbox,
    )

    ax.set_title(
        "Largest component\n"
        f"IoU={component_row['largest_iou']:.3f}"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Panel 4 — Smallest component.
    # --------------------------------------------------------

    ax = axes[3]

    ax.imshow(
        rgb
    )

    overlay_mask(
        ax,
        smallest_mask,
        alpha=0.45,
    )

    draw_bbox(
        ax,
        bbox,
    )

    ax.set_title(
        "Smallest component\n"
        f"IoU={component_row['smallest_iou']:.3f}"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Panel 5 — Oracle best component.
    # --------------------------------------------------------

    ax = axes[4]

    ax.imshow(
        rgb
    )

    if oracle_mask is not None:

        overlay_mask(
            ax,
            oracle_mask,
            alpha=0.45,
        )

    draw_bbox(
        ax,
        bbox,
    )

    ax.set_title(
        "Best component available\n"
        f"IoU={oracle_iou:.3f}"
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Global title.
    # --------------------------------------------------------

    selected_iou = float(
        component_row[
            "selected_iou"
        ]
    )

    fig.suptitle(
        (
            f"{comparator.upper()} | "
            f"{phrase}\n"
            f"patch={patch_id} | "
            f"candidate components={count} | "
            f"selected IoU={selected_iou:.3f}"
        ),
        fontsize=13,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    output_path = (
        OUTPUT_ROOT
        / (
            f"{name}_"
            f"{safe_filename(patch_id)}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 COMPONENT-SELECTION VISUAL DIAGNOSTICS")
    print("=" * 70)

    if not COMPONENT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{COMPONENT_FILE}"
        )

    if not ROW_AUDIT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{ROW_AUDIT_FILE}"
        )

    components = pd.read_csv(
        COMPONENT_FILE
    )

    row_audit = pd.read_csv(
        ROW_AUDIT_FILE
    )

    print(
        f"Component rows: {len(components)}"
    )

    print(
        f"Row-audit rows: {len(row_audit)}"
    )

    # --------------------------------------------------------
    # Build row lookup.
    # --------------------------------------------------------

    row_lookup = {}

    for _, row in row_audit.iterrows():

        row_lookup[
            int(
                row[
                    "original_row_index"
                ]
            )
        ] = row

    # --------------------------------------------------------
    # Choose examples.
    # --------------------------------------------------------

    examples = choose_examples(
        components
    )

    print(
        f"Visual examples selected: "
        f"{len(examples)}"
    )

    generated = []

    # --------------------------------------------------------
    # Generate visuals.
    # --------------------------------------------------------

    for number, (
        name,
        component_row,
    ) in enumerate(
        examples,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"EXAMPLE {number}/{len(examples)}"
        )

        print(
            f"TYPE: {name}"
        )

        print(
            f"PATCH: "
            f"{component_row['patch_id']}"
        )

        print(
            f"PHRASE: "
            f"{component_row['reference_phrase']}"
        )

        print(
            f"COMPARATOR: "
            f"{component_row['comparator']}"
        )

        print(
            f"SELECTED IoU: "
            f"{component_row['selected_iou']:.6f}"
        )

        path = create_visual(
            name,
            component_row,
            row_lookup,
        )

        generated.append(
            path
        )

        print(
            f"CREATED: {path}"
        )

    # --------------------------------------------------------
    # Save a simple manifest.
    # --------------------------------------------------------

    manifest = []

    for path in generated:

        manifest.append(
            {
                "path": str(path),
            }
        )

    manifest_path = (
        OUTPUT_ROOT
        / "visualization_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final result.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VISUAL DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print(
        f"Images generated: "
        f"{len(generated)}"
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Manifest:\n"
        f"{manifest_path}"
    )

    for path in generated:

        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()