import os
import sys
import re

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch
import numpy as np

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_heatmap_model import GroundingHeatmapModel


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    r"D:\SatQueryAI\models\checkpoints"
    r"\grounding_heatmap_geometry"
    r"\model_256_reference.pt"
)

# IMPORTANT:
# This patch is NOT the leaked _83_31 patch.
TARGET_PATCH_ID = (
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_85_62"
)

# Number of real point examples to inspect
NUM_EXAMPLES = 5

QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

DEVICE = torch.device("cpu")


# ============================================================
# HELPERS
# ============================================================

def quantile_from_distribution(values, q):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.maximum(
        values,
        0.0,
    )

    total = values.sum()

    if total <= 0.0:
        return 0.5

    values = values / total

    cdf = np.cumsum(values)

    index = np.searchsorted(
        cdf,
        q,
        side="left",
    )

    index = max(
        0,
        min(index, len(values) - 1),
    )

    return float(
        (index + 0.5) / len(values)
    )


def heatmap_to_box(heatmap):
    """
    Convert a 14x14 probability heatmap
    into [x1,y1,x2,y2] using 10%-90%
    marginal quantiles.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float64,
    )

    heatmap = np.maximum(
        heatmap,
        0.0,
    )

    total = heatmap.sum()

    if total <= 0.0:
        return [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

    heatmap = heatmap / total

    x_distribution = heatmap.sum(
        axis=0
    )

    y_distribution = heatmap.sum(
        axis=1
    )

    x1 = quantile_from_distribution(
        x_distribution,
        QUANTILE_LOW,
    )

    x2 = quantile_from_distribution(
        x_distribution,
        QUANTILE_HIGH,
    )

    y1 = quantile_from_distribution(
        y_distribution,
        QUANTILE_LOW,
    )

    y2 = quantile_from_distribution(
        y_distribution,
        QUANTILE_HIGH,
    )

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


def box_center(box):
    return np.array(
        [
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        ],
        dtype=np.float64,
    )


def parse_point(text):
    """
    Extract point coordinates from:
        <point>(x, y)</point>
    """

    if text is None:
        return None

    text = str(text)

    match = re.search(
        r"<point>\s*\(\s*"
        r"([0-9]*\.?[0-9]+)"
        r"\s*,\s*"
        r"([0-9]*\.?[0-9]+)"
        r"\s*\)\s*</point>",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


def extract_point_from_row(row):
    """
    Search likely textual columns for the actual point.
    """

    for column in [
        "input",
        "output",
        "question",
    ]:

        if column not in row.index:
            continue

        value = row[column]

        if value is None:
            continue

        point = parse_point(
            value
        )

        if point is not None:
            return point

    return None


def box_distance(box_a, box_b):
    """
    Mean absolute difference across four box coordinates.
    """

    a = np.asarray(
        box_a,
        dtype=np.float64,
    )

    b = np.asarray(
        box_b,
        dtype=np.float64,
    )

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def center_distance(center_a, center_b):
    """
    Euclidean distance between box centers.
    """

    return float(
        np.linalg.norm(
            np.asarray(center_a)
            - np.asarray(center_b)
        )
    )


def load_checkpoint(model):

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    current_state = model.state_dict()

    compatible_state = {}

    for key, value in state_dict.items():

        if key not in current_state:
            continue

        if (
            current_state[key].shape
            != value.shape
        ):
            continue

        compatible_state[key] = value

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    return len(
        compatible_state
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POINT-GROUNDING SENSITIVITY TEST")
    print("=" * 70)

    print(
        "\nTarget patch:"
    )

    print(
        TARGET_PATCH_ID
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    dataset = GroundingDataset()

    df = dataset.df

    # --------------------------------------------------------
    # Find all rows belonging to target patch
    # and category == point.
    # --------------------------------------------------------

    matching_indices = []

    for index, row in df.iterrows():

        patch_id = str(
            row.get(
                "patch_id",
                "",
            )
        )

        category = str(
            row.get(
                "category",
                "",
            )
        ).strip().lower()

        if (
            patch_id == TARGET_PATCH_ID
            and category == "point"
        ):

            point = extract_point_from_row(
                row
            )

            if point is not None:
                matching_indices.append(
                    index
                )

    print(
        "\nPoint examples found on patch: "
        f"{len(matching_indices)}"
    )

    if len(matching_indices) < NUM_EXAMPLES:

        raise RuntimeError(
            "Not enough point examples "
            "were found on the selected patch."
        )

    # ========================================================
    # SELECT FIVE DIVERSE POINTS
    # ========================================================

    candidates = []

    for index in matching_indices:

        row = df.iloc[index]

        point = extract_point_from_row(
            row
        )

        if point is None:
            continue

        candidates.append(
            (
                index,
                point[0],
                point[1],
            )
        )

    # --------------------------------------------------------
    # Select points spread throughout the image.
    #
    # We prefer one point near each corner and one
    # point near the center if such points exist.
    # Otherwise, use the first five sufficiently
    # separated coordinates.
    # --------------------------------------------------------

    desired_locations = [
        (0.10, 0.10),
        (0.90, 0.10),
        (0.50, 0.50),
        (0.10, 0.90),
        (0.90, 0.90),
    ]

    selected = []
    used_indices = set()

    for desired_x, desired_y in desired_locations:

        best = None
        best_distance = float("inf")

        for index, x, y in candidates:

            if index in used_indices:
                continue

            distance = (
                (x - desired_x) ** 2
                +
                (y - desired_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance

                best = (
                    index,
                    x,
                    y,
                )

        if best is not None:

            selected.append(
                best
            )

            used_indices.add(
                best[0]
            )

    # Safety fallback
    if len(selected) < NUM_EXAMPLES:

        for candidate in candidates:

            if candidate[0] in used_indices:
                continue

            selected.append(
                candidate
            )

            used_indices.add(
                candidate[0]
            )

            if len(selected) >= NUM_EXAMPLES:
                break

    selected = selected[
        :NUM_EXAMPLES
    ]

    print(
        "\nSelected points:"
    )

    for index, x, y in selected:

        print(
            f"  index={index} "
            f"point=({x:.4f}, {y:.4f})"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    model = GroundingHeatmapModel()

    loaded = load_checkpoint(
        model
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded compatible parameters: "
        f"{loaded}"
    )

    # ========================================================
    # RUN CONTROLLED TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNNING CONTROLLED POINT TEST")
    print("=" * 70)

    results = []

    with torch.no_grad():

        for position, dataset_index in enumerate(
            selected,
            start=1,
        ):

            index = dataset_index[0]

            x = dataset_index[1]
            y = dataset_index[2]

            item = dataset[
                index
            ]

            s1 = item["s1"]
            s2 = item["s2"]
            text_embedding = item[
                "text_embedding"
            ]

            if s1.dim() == 2:
                s1 = s1.unsqueeze(0)

            if s2.dim() == 2:
                s2 = s2.unsqueeze(0)

            if text_embedding.dim() == 1:
                text_embedding = (
                    text_embedding.unsqueeze(0)
                )

            output = model(
                s1=s1.to(DEVICE),
                s2=s2.to(DEVICE),
                text_embedding=text_embedding.to(
                    DEVICE
                ),
            )

            heatmap = (
                output["heatmap"][0]
                .detach()
                .cpu()
                .numpy()
            )

            predicted_box = heatmap_to_box(
                heatmap
            )

            predicted_center = box_center(
                predicted_box
            )

            # For point grounding, the queried point
            # is expected to lie inside the target
            # bounding box.
            target_box = (
                item["target_box"]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
                .tolist()
            )

            target_center = box_center(
                target_box
            )

            results.append(
                {
                    "index": index,
                    "point": (x, y),
                    "target_box": target_box,
                    "predicted_box":
                        predicted_box,
                    "target_center":
                        target_center,
                    "predicted_center":
                        predicted_center,
                    "heatmap":
                        heatmap,
                }
            )

            print(
                f"\nExample {position}"
            )

            print(
                f"Dataset index: {index}"
            )

            print(
                f"Query point: "
                f"({x:.4f}, {y:.4f})"
            )

            print(
                "Target box:    "
                f"["
                f"{target_box[0]:.4f}, "
                f"{target_box[1]:.4f}, "
                f"{target_box[2]:.4f}, "
                f"{target_box[3]:.4f}"
                f"]"
            )

            print(
                "Predicted box: "
                f"["
                f"{predicted_box[0]:.4f}, "
                f"{predicted_box[1]:.4f}, "
                f"{predicted_box[2]:.4f}, "
                f"{predicted_box[3]:.4f}"
                f"]"
            )

            print(
                "Predicted center: "
                f"("
                f"{predicted_center[0]:.4f}, "
                f"{predicted_center[1]:.4f}"
                f")"
            )

    # ========================================================
    # CONDITION SENSITIVITY
    # ========================================================

    print("\n" + "=" * 70)
    print("POINT-CONDITION SENSITIVITY")
    print("=" * 70)

    prediction_boxes = [
        result["predicted_box"]
        for result in results
    ]

    prediction_centers = [
        result["predicted_center"]
        for result in results
    ]

    # --------------------------------------------------------
    # Pairwise prediction distances
    # --------------------------------------------------------

    pairwise_box_distances = []
    pairwise_center_distances = []

    for i in range(
        len(results)
    ):

        for j in range(
            i + 1,
            len(results),
        ):

            box_dist = box_distance(
                prediction_boxes[i],
                prediction_boxes[j],
            )

            center_dist = center_distance(
                prediction_centers[i],
                prediction_centers[j],
            )

            pairwise_box_distances.append(
                box_dist
            )

            pairwise_center_distances.append(
                center_dist
            )

    mean_box_distance = float(
        np.mean(
            pairwise_box_distances
        )
    )

    mean_center_distance = float(
        np.mean(
            pairwise_center_distances
        )
    )

    print(
        "\nMean pairwise predicted-box L1: "
        f"{mean_box_distance:.6f}"
    )

    print(
        "Mean pairwise predicted-center "
        f"distance: {mean_center_distance:.6f}"
    )

    # ========================================================
    # QUERY POINT → PREDICTED BOX RELATIONSHIP
    # ========================================================

    print("\n" + "=" * 70)
    print("QUERY POINT → PREDICTED BOX CHECK")
    print("=" * 70)

    point_inside_count = 0

    point_to_center_distances = []

    for result in results:

        x, y = result[
            "point"
        ]

        box = result[
            "predicted_box"
        ]

        center = result[
            "predicted_center"
        ]

        inside = (
            box[0] <= x <= box[2]
            and
            box[1] <= y <= box[3]
        )

        if inside:
            point_inside_count += 1

        distance = center_distance(
            (x, y),
            center,
        )

        point_to_center_distances.append(
            distance
        )

        print(
            f"\nPoint "
            f"({x:.4f}, {y:.4f})"
        )

        print(
            f"Predicted center "
            f"({center[0]:.4f}, "
            f"{center[1]:.4f})"
        )

        print(
            f"Point inside predicted box: "
            f"{inside}"
        )

        print(
            f"Point → predicted-center "
            f"distance: {distance:.4f}"
        )

    mean_point_center_distance = float(
        np.mean(
            point_to_center_distances
        )
    )

    print(
        "\nPoints inside predicted boxes: "
        f"{point_inside_count}/{len(results)}"
    )

    print(
        "Mean point → predicted-center "
        f"distance: "
        f"{mean_point_center_distance:.6f}"
    )

    # ========================================================
    # HEATMAP PEAK LOCATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("HEATMAP PEAK LOCATIONS")
    print("=" * 70)

    peak_locations = []

    for result in results:

        heatmap = result[
            "heatmap"
        ]

        peak_index = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape,
        )

        peak_y = peak_index[0]
        peak_x = peak_index[1]

        normalized_peak = (
            (peak_x + 0.5)
            / heatmap.shape[1],

            (peak_y + 0.5)
            / heatmap.shape[0],
        )

        peak_locations.append(
            normalized_peak
        )

        print(
            f"\nPoint: "
            f"({result['point'][0]:.4f}, "
            f"{result['point'][1]:.4f})"
        )

        print(
            f"Heatmap peak: "
            f"grid=({peak_x}, {peak_y}) "
            f"normalized="
            f"({normalized_peak[0]:.4f}, "
            f"{normalized_peak[1]:.4f})"
        )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if mean_box_distance < 0.05:

        print(
            "\nVERY LOW POINT SENSITIVITY."
        )

        print(
            "Changing the query point produces "
            "almost the same predicted box."
        )

        print(
            "This strongly suggests that the "
            "current point-conditioning path "
            "is not effectively using the "
            "coordinate information."
        )

    elif mean_box_distance < 0.15:

        print(
            "\nMODERATE POINT SENSITIVITY."
        )

        print(
            "The predictions move somewhat when "
            "the queried point changes, but "
            "the localization signal appears "
            "weak."
        )

    else:

        print(
            "\nSTRONG POINT SENSITIVITY."
        )

        print(
            "Changing the queried point produces "
            "substantial movement in the predicted "
            "boxes."
        )

    print(
        "\nThe point-grounding result should be "
        "interpreted separately from reference "
        "grounding."
    )

    print(
        "\nTEST COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()