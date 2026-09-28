# =============================================================================
# SATQUERYAI — INPUT VALIDATOR
# =============================================================================
#
# Validation layer for the project's processed remote-sensing tensor format.
#
# Supported processed input:
#
# {
#     "patch_id": ...,
#     "channel_names": [...],
#     "image": Tensor[14,120,120]
# }
#
# This module:
#   - validates structure
#   - validates channel contract
#   - validates tensor dimensions
#   - validates numerical values
#
# This module does NOT:
#   - run DOFA
#   - run any specialist
#   - modify input files
#   - modify model checkpoints
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_CHANNEL_NAMES = [
    "VH",
    "VV",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
]

EXPECTED_SHAPE = (
    14,
    120,
    120,
)

MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0


# =============================================================================
# VALIDATION ERROR
# =============================================================================

class InputValidationError(ValueError):
    """Raised when a processed SatQueryAI input is invalid."""


# =============================================================================
# VALIDATE OBJECT
# =============================================================================

def validate_processed_tensor(
    data: Any,
) -> Dict[str, Any]:
    """
    Validate one processed SatQueryAI .pt object.

    Returns a compact metadata dictionary when validation succeeds.
    """

    if not isinstance(data, dict):
        raise InputValidationError(
            "Processed input must be a dictionary."
        )

    required_keys = {
        "patch_id",
        "channel_names",
        "image",
    }

    missing_keys = (
        required_keys
        - set(data.keys())
    )

    if missing_keys:
        raise InputValidationError(
            "Processed input is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    # -------------------------------------------------------------------------
    # Patch ID
    # -------------------------------------------------------------------------

    patch_id = data["patch_id"]

    if not isinstance(
        patch_id,
        str,
    ):
        raise InputValidationError(
            "patch_id must be a string."
        )

    patch_id = patch_id.strip()

    if not patch_id:
        raise InputValidationError(
            "patch_id cannot be empty."
        )

    # -------------------------------------------------------------------------
    # Channel names
    # -------------------------------------------------------------------------

    channel_names = data[
        "channel_names"
    ]

    if not isinstance(
        channel_names,
        (list, tuple),
    ):
        raise InputValidationError(
            "channel_names must be a list or tuple."
        )

    channel_names = list(
        channel_names
    )

    if channel_names != (
        EXPECTED_CHANNEL_NAMES
    ):
        raise InputValidationError(
            "Channel-name contract mismatch.\n"
            f"Expected: {EXPECTED_CHANNEL_NAMES}\n"
            f"Received: {channel_names}"
        )

    # -------------------------------------------------------------------------
    # Image tensor
    # -------------------------------------------------------------------------

    image = data["image"]

    if not isinstance(
        image,
        torch.Tensor,
    ):
        raise InputValidationError(
            "image must be a torch.Tensor."
        )

    if tuple(image.shape) != EXPECTED_SHAPE:
        raise InputValidationError(
            "Invalid image tensor shape: "
            f"{tuple(image.shape)}; "
            f"expected {EXPECTED_SHAPE}."
        )

    if not torch.is_floating_point(
        image
    ):
        raise InputValidationError(
            "image tensor must use a floating-point dtype."
        )

    if not torch.isfinite(
        image
    ).all():
        raise InputValidationError(
            "image tensor contains non-finite values."
        )

    # -------------------------------------------------------------------------
    # Normalized value range
    # -------------------------------------------------------------------------

    image_float = (
        image.detach()
        .cpu()
        .float()
    )

    minimum = float(
        image_float.min().item()
    )

    maximum = float(
        image_float.max().item()
    )

    if minimum < (
        MIN_NORMALIZED_VALUE
    ) or maximum > (
        MAX_NORMALIZED_VALUE
    ):
        raise InputValidationError(
            "image tensor contains values outside "
            "the expected normalized range [0,1]. "
            f"Observed min={minimum}, max={maximum}."
        )

    return {
        "valid": True,
        "patch_id": patch_id,
        "channel_names": channel_names,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": minimum,
        "max": maximum,
        "num_elements": int(
            image.numel()
        ),
    }


# =============================================================================
# LOAD + VALIDATE .PT FILE
# =============================================================================

def load_and_validate_processed_file(
    path: str | Path,
) -> Dict[str, Any]:
    """
    Load and validate a processed .pt file.

    The source file itself is never modified.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed input not found: {path}"
        )

    if not path.is_file():
        raise InputValidationError(
            f"Input path is not a file: {path}"
        )

    if path.suffix.lower() != ".pt":
        raise InputValidationError(
            f"Unsupported processed input extension: "
            f"{path.suffix}"
        )

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise InputValidationError(
            f"Could not load PyTorch input: {path}"
        ) from exc

    metadata = validate_processed_tensor(
        data
    )

    metadata["path"] = str(
        path
    )

    return metadata


# =============================================================================
# DIRECT SELF-TEST
# =============================================================================

if __name__ == "__main__":

    SAMPLE_PATH = (
        Path(r"D:\SatQueryAI")
        / "data"
        / "processed"
        / "preprocessed"
        / (
            "S2A_MSIL2A_20170613T101031_"
            "N9999_R022_T33UUP_40_90.pt"
        )
    )

    print("=" * 80)
    print(
        "SATQUERYAI — INPUT VALIDATOR TEST"
    )
    print("=" * 80)

    metadata = (
        load_and_validate_processed_file(
            SAMPLE_PATH
        )
    )

    print()
    print("VALIDATION RESULT")
    print(
        "  valid:",
        metadata["valid"],
    )
    print(
        "  patch_id:",
        metadata["patch_id"],
    )
    print(
        "  shape:",
        metadata["shape"],
    )
    print(
        "  dtype:",
        metadata["dtype"],
    )
    print(
        "  min:",
        metadata["min"],
    )
    print(
        "  max:",
        metadata["max"],
    )
    print(
        "  elements:",
        metadata["num_elements"],
    )

    if metadata["shape"] != [
        14,
        120,
        120,
    ]:
        raise RuntimeError(
            "Input shape verification failed."
        )

    if metadata[
        "channel_names"
    ] != EXPECTED_CHANNEL_NAMES:
        raise RuntimeError(
            "Channel-name verification failed."
        )

    print()
    print(
        "PROCESSED INPUT VALIDATION: PASS"
    )