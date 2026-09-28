import os
import sys

# ============================================================
# PROJECT IMPORT PATH
# ============================================================

# Add D:\SatQueryAI to Python's import path so that
# "from src...." imports work even when this script is
# executed from inside src\models.
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

# Number of unseen examples to evaluate
EXAMPLES = 256

# IMPORTANT:
# Original overfit experiment used indices 0..255.
# Therefore, evaluate on completely unseen examples:
# 256..511
START_INDEX = 256

# Boundary extraction quantiles
QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

DEVICE = torch.device("cpu")


# ============================================================
# BOX IoU
# ============================================================

def box_iou(box_a, box_b):
    """
    Calculate IoU between two boxes.

    Box format:
        [x1, y1, x2, y2]

    Coordinates are normalized to [0,1].
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    # Areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Union
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


# ============================================================
# QUANTILE EXTRACTION
# ============================================================

def quantile_from_distribution(values, q):
    """
    Extract a normalized spatial coordinate from a 1D
    probability distribution.

    Parameters
    ----------
    values : array-like
        Probability distribution.

    q : float
        Quantile in [0,1].

    Returns
    -------
    float
        Normalized coordinate in [0,1].
    """

    values = np.asarray(values, dtype=np.float64)

    # Remove numerical negatives
    values = np.maximum(values, 0.0)

    total = values.sum()

    if total <= 0.0:
        return 0.5

    # Normalize
    values = values / total

    # Cumulative distribution
    cdf = np.cumsum(values)

    # Find first cell whose cumulative mass reaches q
    index = np.searchsorted(
        cdf,
        q,
        side="left",
    )

    # Clamp index
    index = max(
        0,
        min(index, len(values) - 1),
    )

    # Convert grid-cell index into normalized coordinate.
    #
    # For a 14-cell dimension:
    #
    # cell 0  -> center 0.0357
    # cell 13 -> center 0.9643
    #
    coordinate = (
        (index + 0.5) / len(values)
    )

    return float(coordinate)


# ============================================================
# HEATMAP → BOX
# ============================================================

def heatmap_to_box(
    heatmap,
    low=QUANTILE_LOW,
    high=QUANTILE_HIGH,
):
    """
    Convert a [14,14] spatial probability heatmap
    into a bounding box.

    Steps:

        heatmap
           ↓
        x marginal + y marginal
           ↓
        low/high quantiles
           ↓
        [x1,y1,x2,y2]
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float64,
    )

    # Numerical safety
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

    # Normalize entire spatial distribution
    heatmap = heatmap / total

    # --------------------------------------------------------
    # Spatial marginals
    # --------------------------------------------------------

    # Sum vertically → probability over x
    x_distribution = heatmap.sum(axis=0)

    # Sum horizontally → probability over y
    y_distribution = heatmap.sum(axis=1)

    # --------------------------------------------------------
    # Quantile boundaries
    # --------------------------------------------------------

    x1 = quantile_from_distribution(
        x_distribution,
        low,
    )

    x2 = quantile_from_distribution(
        x_distribution,
        high,
    )

    y1 = quantile_from_distribution(
        y_distribution,
        low,
    )

    y2 = quantile_from_distribution(
        y_distribution,
        high,
    )

    # Make sure the box is valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Clamp to valid normalized coordinates
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HELD-OUT GROUNDING GENERALIZATION TEST")
    print("=" * 70)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    dataset = GroundingDataset()

    print(
        f"Total grounding examples: "
        f"{len(dataset):,}"
    )

    print(
        f"Evaluation examples: "
        f"{EXAMPLES}"
    )

    print(
        f"Evaluation range: "
        f"{START_INDEX} .. "
        f"{START_INDEX + EXAMPLES - 1}"
    )

    # Safety check
    assert (
        START_INDEX + EXAMPLES <= len(dataset)
    ), (
        "Held-out evaluation range exceeds "
        "dataset size."
    )

    # Explicit unseen indices
    indices = list(
        range(
            START_INDEX,
            START_INDEX + EXAMPLES,
        )
    )

    print("\nDataset loaded successfully.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    model = GroundingHeatmapModel()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Support common checkpoint formats
    # --------------------------------------------------------

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

    # ========================================================
    # COMPATIBLE PARAMETERS ONLY
    # ========================================================

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

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded compatible parameters: "
        f"{len(compatible_state):,}"
    )

    print("Model loaded successfully.")

    # ========================================================
    # HELD-OUT EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("RUNNING HELD-OUT EVALUATION")
    print("=" * 70)

    ious = []

    example_records = []

    # --------------------------------------------------------
    # No DataLoader is used here.
    #
    # GroundingDataset contains some optional fields which
    # may be None. Evaluating items individually avoids the
    # default PyTorch collate error while leaving the dataset
    # implementation untouched.
    # --------------------------------------------------------

    with torch.no_grad():

        for eval_position, dataset_index in enumerate(
            indices
        ):

            # ----------------------------------------------
            # Fetch one unseen grounding example
            # ----------------------------------------------

            item = dataset[
                dataset_index
            ]

            # ----------------------------------------------
            # Required model inputs only
            # ----------------------------------------------

            s1 = item["s1"]

            s2 = item["s2"]

            text_embedding = item[
                "text_embedding"
            ]

            # Add batch dimension
            if s1.dim() == 2:
                s1 = s1.unsqueeze(0)

            if s2.dim() == 2:
                s2 = s2.unsqueeze(0)

            if text_embedding.dim() == 1:
                text_embedding = (
                    text_embedding.unsqueeze(0)
                )

            # CPU
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            text_embedding = (
                text_embedding.to(DEVICE)
            )

            # ----------------------------------------------
            # Forward pass
            # ----------------------------------------------

            output = model(
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
            )

            # ----------------------------------------------
            # Extract 14×14 heatmap
            # ----------------------------------------------

            heatmap = (
                output["heatmap"][0]
                .detach()
                .cpu()
                .numpy()
            )

            # ----------------------------------------------
            # Convert heatmap → boundary box
            # ----------------------------------------------

            prediction = heatmap_to_box(
                heatmap,
                low=QUANTILE_LOW,
                high=QUANTILE_HIGH,
            )

            # ----------------------------------------------
            # Ground-truth target box
            # ----------------------------------------------

            target_tensor = item[
                "target_box"
            ]

            if torch.is_tensor(
                target_tensor
            ):
                target = (
                    target_tensor
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .tolist()
                )
            else:
                target = (
                    np.asarray(
                        target_tensor,
                        dtype=np.float64,
                    )
                    .tolist()
                )

            # ----------------------------------------------
            # IoU
            # ----------------------------------------------

            iou = box_iou(
                prediction,
                target,
            )

            ious.append(iou)

            example_records.append(
                {
                    "dataset_index":
                        dataset_index,
                    "target":
                        target,
                    "prediction":
                        prediction,
                    "iou":
                        iou,
                }
            )

            # Progress
            if (
                eval_position + 1
            ) % 32 == 0:

                print(
                    f"Processed "
                    f"{eval_position + 1}"
                    f"/{EXAMPLES}"
                )

    # ========================================================
    # METRICS
    # ========================================================

    ious = np.asarray(
        ious,
        dtype=np.float64,
    )

    mean_iou = float(
        ious.mean()
    )

    median_iou = float(
        np.median(ious)
    )

    min_iou = float(
        ious.min()
    )

    max_iou = float(
        ious.max()
    )

    count_50 = int(
        (ious >= 0.50).sum()
    )

    count_25 = int(
        (ious >= 0.25).sum()
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("HELD-OUT RESULTS")
    print("=" * 70)

    print(
        f"Training examples used for checkpoint: "
        f"0 .. {START_INDEX - 1}"
    )

    print(
        f"Held-out examples evaluated: "
        f"{START_INDEX} .. "
        f"{START_INDEX + EXAMPLES - 1}"
    )

    print(
        f"\nBoundary quantiles: "
        f"{QUANTILE_LOW:.2f} - "
        f"{QUANTILE_HIGH:.2f}"
    )

    print(
        f"Examples: "
        f"{len(ious)}"
    )

    print(
        f"\nMean IoU:   "
        f"{mean_iou:.6f}"
    )

    print(
        f"Median IoU: "
        f"{median_iou:.6f}"
    )

    print(
        f"Min IoU:    "
        f"{min_iou:.6f}"
    )

    print(
        f"Max IoU:    "
        f"{max_iou:.6f}"
    )

    print(
        f"\nIoU >= 0.50: "
        f"{count_50}/{EXAMPLES} "
        f"("
        f"{100.0 * count_50 / EXAMPLES:.2f}%"
        f")"
    )

    print(
        f"IoU >= 0.25: "
        f"{count_25}/{EXAMPLES} "
        f"("
        f"{100.0 * count_25 / EXAMPLES:.2f}%"
        f")"
    )

    # ========================================================
    # COMPARISON WITH TRAINING RESULT
    # ========================================================

    TRAINING_MEAN_IOU = 0.490380

    drop = (
        TRAINING_MEAN_IOU
        - mean_iou
    )

    print("\n" + "=" * 70)
    print("TRAINING vs HELD-OUT")
    print("=" * 70)

    print(
        f"Training-set mean IoU: "
        f"{TRAINING_MEAN_IOU:.6f}"
    )

    print(
        f"Held-out mean IoU:    "
        f"{mean_iou:.6f}"
    )

    print(
        f"Absolute difference:  "
        f"{drop:.6f}"
    )

    if TRAINING_MEAN_IOU > 0:

        retained = (
            mean_iou
            / TRAINING_MEAN_IOU
        ) * 100.0

        print(
            f"Performance retained: "
            f"{retained:.2f}%"
        )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE HELD-OUT BOXES")
    print("=" * 70)

    for record in example_records[:10]:

        target = record["target"]

        prediction = record[
            "prediction"
        ]

        print(
            f"\nDataset index: "
            f"{record['dataset_index']}"
        )

        print(
            "Target:    "
            f"["
            f"{target[0]:.4f}, "
            f"{target[1]:.4f}, "
            f"{target[2]:.4f}, "
            f"{target[3]:.4f}"
            f"]"
        )

        print(
            "Predicted: "
            f"["
            f"{prediction[0]:.4f}, "
            f"{prediction[1]:.4f}, "
            f"{prediction[2]:.4f}, "
            f"{prediction[3]:.4f}"
            f"]"
        )

        print(
            f"IoU:       "
            f"{record['iou']:.4f}"
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        "\nThe checkpoint was trained only on "
        f"indices 0-{START_INDEX - 1}."
    )

    print(
        "The current examples were not used "
        "during that training experiment."
    )

    print(
        "\nThe critical comparison is between:"
    )

    print(
        f"  Training mean IoU = "
        f"{TRAINING_MEAN_IOU:.4f}"
    )

    print(
        f"  Held-out mean IoU  = "
        f"{mean_iou:.4f}"
    )

    if mean_iou >= 0.45:

        print(
            "\nGENERALIZATION LOOKS PROMISING."
        )

        print(
            "The learned heatmap appears to retain "
            "useful image-conditioned spatial "
            "information on unseen examples."
        )

    elif mean_iou >= 0.30:

        print(
            "\nPARTIAL GENERALIZATION."
        )

        print(
            "The model transfers some spatial "
            "grounding ability to unseen examples, "
            "but there is a noticeable performance "
            "drop relative to the training subset."
        )

    else:

        print(
            "\nGENERALIZATION IS WEAK."
        )

        print(
            "The strong training result was likely "
            "substantially influenced by memorization "
            "of the 256 training examples."
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()