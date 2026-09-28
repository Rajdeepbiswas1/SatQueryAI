import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt

from torch.utils.data import Subset

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_heatmap_model import (
    GroundingHeatmapModel,
    boxes_to_heatmaps,
)


# ============================================================
# Configuration
# ============================================================

NUM_EXAMPLES = 3

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "grounding_heatmap_overfit"
    / "model_256_reference.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "grounding_heatmaps"
)

GRID_SIZE = 14


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("GROUNDING HEATMAP VISUALIZATION")
print("=" * 70)

dataset = GroundingDataset()

reference_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "reference"
]

reference_subset = reference_indices[:256]

print(
    f"Reference examples: {len(reference_indices):,}"
)

# ============================================================
# Load model
# ============================================================

print()
print("Loading checkpoint:")

print(
    CHECKPOINT
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

model = GroundingHeatmapModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint[
        "model_state_dict"
    ]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict
)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print(
    f"Output directory:\n{OUTPUT_DIR}"
)


# ============================================================
# Visualization helper
# ============================================================

def save_heatmap(
    heatmap,
    path,
    title,
):
    """
    Save one 14x14 heatmap as a PNG.
    """

    data = heatmap.detach().cpu().numpy()

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        data,
        interpolation="nearest",
        aspect="equal",
        cmap="inferno",
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
        label="Spatial probability",
    )

    plt.title(title)

    plt.xlabel(
        "Spatial column"
    )

    plt.ylabel(
        "Spatial row"
    )

    plt.xticks(
        range(GRID_SIZE)
    )

    plt.yticks(
        range(GRID_SIZE)
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Comparison visualization
# ============================================================

def save_comparison(
    target,
    predicted,
    path,
    title,
):
    """
    Save target and predicted heatmaps together.
    """

    target_np = (
        target.detach()
        .cpu()
        .numpy()
    )

    predicted_np = (
        predicted.detach()
        .cpu()
        .numpy()
    )

    difference_np = (
        predicted_np
        - target_np
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    image0 = axes[0].imshow(
        target_np,
        interpolation="nearest",
        aspect="equal",
        cmap="inferno",
    )

    axes[0].set_title(
        "Ground-truth heatmap"
    )

    axes[0].set_xlabel(
        "Column"
    )

    axes[0].set_ylabel(
        "Row"
    )

    axes[0].set_xticks(
        range(GRID_SIZE)
    )

    axes[0].set_yticks(
        range(GRID_SIZE)
    )

    fig.colorbar(
        image0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    image1 = axes[1].imshow(
        predicted_np,
        interpolation="nearest",
        aspect="equal",
        cmap="inferno",
    )

    axes[1].set_title(
        "Predicted heatmap"
    )

    axes[1].set_xlabel(
        "Column"
    )

    axes[1].set_ylabel(
        "Row"
    )

    axes[1].set_xticks(
        range(GRID_SIZE)
    )

    axes[1].set_yticks(
        range(GRID_SIZE)
    )

    fig.colorbar(
        image1,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    max_difference = max(
        abs(difference_np.min()),
        abs(difference_np.max()),
        1e-8,
    )

    image2 = axes[2].imshow(
        difference_np,
        interpolation="nearest",
        aspect="equal",
        cmap="coolwarm",
        vmin=-max_difference,
        vmax=max_difference,
    )

    axes[2].set_title(
        "Prediction - target"
    )

    axes[2].set_xlabel(
        "Column"
    )

    axes[2].set_ylabel(
        "Row"
    )

    axes[2].set_xticks(
        range(GRID_SIZE)
    )

    axes[2].set_yticks(
        range(GRID_SIZE)
    )

    fig.colorbar(
        image2,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    fig.suptitle(
        title,
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Run visualization
# ============================================================

for example_number in range(
    min(
        NUM_EXAMPLES,
        len(reference_subset),
    )
):

    dataset_index = reference_subset[
        example_number
    ]

    item = dataset[
        dataset_index
    ]

    s1 = item[
        "s1"
    ].unsqueeze(0)

    s2 = item[
        "s2"
    ].unsqueeze(0)

    text_embedding = item[
        "text_embedding"
    ].unsqueeze(0)

    target_box = item[
        "target_box"
    ].unsqueeze(0)

    print()
    print(
        "=" * 70
    )

    print(
        f"EXAMPLE {example_number:02d}"
    )

    print(
        f"Dataset index: "
        f"{dataset_index}"
    )

    print(
        f"Patch ID: "
        f"{item['patch_id']}"
    )

    with torch.no_grad():

        outputs = model(
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
        )

    predicted_heatmap = (
        outputs["heatmap"][0]
    )

    target_heatmap = (
        boxes_to_heatmaps(
            target_box
        )[0]
    )

    predicted_box = (
        outputs["box"][0]
        .tolist()
    )

    target_box_list = (
        target_box[0]
        .tolist()
    )

    # --------------------------------------------------------
    # Peak locations
    # --------------------------------------------------------

    predicted_peak = torch.argmax(
        predicted_heatmap.reshape(-1)
    ).item()

    target_peak = torch.argmax(
        target_heatmap.reshape(-1)
    ).item()

    predicted_peak_row = (
        predicted_peak // GRID_SIZE
    )

    predicted_peak_col = (
        predicted_peak % GRID_SIZE
    )

    target_peak_row = (
        target_peak // GRID_SIZE
    )

    target_peak_col = (
        target_peak % GRID_SIZE
    )

    print(
        "Target box:   ",
        target_box_list
    )

    print(
        "Predicted box:",
        predicted_box
    )

    print(
        "Target peak: "
        f"(row={target_peak_row}, "
        f"col={target_peak_col})"
    )

    print(
        "Predicted peak: "
        f"(row={predicted_peak_row}, "
        f"col={predicted_peak_col})"
    )

    # --------------------------------------------------------
    # Save individual heatmaps
    # --------------------------------------------------------

    target_path = (
        OUTPUT_DIR
        / f"example_{example_number:02d}_target.png"
    )

    predicted_path = (
        OUTPUT_DIR
        / f"example_{example_number:02d}_predicted.png"
    )

    comparison_path = (
        OUTPUT_DIR
        / f"example_{example_number:02d}_comparison.png"
    )

    save_heatmap(
        target_heatmap,
        target_path,
        (
            f"Target heatmap - "
            f"example {example_number:02d}"
        ),
    )

    save_heatmap(
        predicted_heatmap,
        predicted_path,
        (
            f"Predicted heatmap - "
            f"example {example_number:02d}"
        ),
    )

    save_comparison(
        target_heatmap,
        predicted_heatmap,
        comparison_path,
        (
            f"Grounding heatmap comparison - "
            f"example {example_number:02d}"
        ),
    )

    print(
        f"Saved: {target_path}"
    )

    print(
        f"Saved: {predicted_path}"
    )

    print(
        f"Saved: {comparison_path}"
    )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"PNG files are in:\n{OUTPUT_DIR}"
)