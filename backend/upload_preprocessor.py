from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from preprocessing.preprocess_dataset import (
    CHANNEL_NAMES,
    S2_BANDS,
    read_and_normalize_s1,
    read_and_normalize_s2,
)


SUPPORTED_SUFFIXES = {
    ".tif",
    ".tiff",
}


class UploadPreprocessingError(Exception):
    """Raised when uploaded remote-sensing inputs cannot be preprocessed."""


def _validate_file(path: Path, label: str) -> None:
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_file():
        raise UploadPreprocessingError(
            f"{label} does not exist: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UploadPreprocessingError(
            f"{label} must be a TIFF/GeoTIFF file: {path.name}"
        )


def preprocess_uploaded_pair(
    s1_files: Mapping[str, Path],
    s2_files: Mapping[str, Path],
    patch_id: str = "uploaded_pair",
) -> torch.Tensor:
    """
    Convert a raw Sentinel-1 + Sentinel-2 upload into the exact
    14-channel tensor contract used by SatQueryAI.

    Required S1 files:
        VH, VV

    Required S2 files:
        B01, B02, B03, B04, B05, B06, B07,
        B08, B09, B11, B12, B8A

    Output:
        float32 tensor with shape [14, 120, 120]
        values clamped to [0, 1]
    """

    if not patch_id or not patch_id.strip():
        raise UploadPreprocessingError(
            "patch_id cannot be empty."
        )

    expected_s1 = {"VH", "VV"}
    actual_s1 = set(s1_files.keys())

    if actual_s1 != expected_s1:
        raise UploadPreprocessingError(
            "Sentinel-1 files must contain exactly: VH, VV. "
            f"Received: {sorted(actual_s1)}"
        )

    expected_s2 = set(S2_BANDS)
    actual_s2 = set(s2_files.keys())

    if actual_s2 != expected_s2:
        raise UploadPreprocessingError(
            "Sentinel-2 files must contain exactly: "
            f"{', '.join(S2_BANDS)}. "
            f"Received: {sorted(actual_s2)}"
        )

    for channel in ("VH", "VV"):
        _validate_file(
            Path(s1_files[channel]),
            f"S1 {channel}",
        )

    for band in S2_BANDS:
        _validate_file(
            Path(s2_files[band]),
            f"S2 {band}",
        )

    channels = []

    # -------------------------------------------------------------
    # Sentinel-1
    # Exact same normalization as the existing dataset pipeline.
    # -------------------------------------------------------------

    channels.append(
        read_and_normalize_s1(
            Path(s1_files["VH"]),
            "VH",
        )
    )

    channels.append(
        read_and_normalize_s1(
            Path(s1_files["VV"]),
            "VV",
        )
    )

    # -------------------------------------------------------------
    # Sentinel-2
    # Exact same normalization and 120x120 bilinear resampling.
    # -------------------------------------------------------------

    for band in S2_BANDS:
        channels.append(
            read_and_normalize_s2(
                Path(s2_files[band]),
                band,
            )
        )

    array = np.stack(
        channels,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        14,
        120,
        120,
    )

    if array.shape != expected_shape:
        raise UploadPreprocessingError(
            f"Unexpected processed shape for {patch_id}: "
            f"{array.shape}; expected {expected_shape}"
        )

    tensor = torch.from_numpy(array)

    if not torch.isfinite(tensor).all().item():
        raise UploadPreprocessingError(
            f"Non-finite values found while preprocessing {patch_id}."
        )

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    if tensor.dtype != torch.float32:
        raise UploadPreprocessingError(
            f"Unexpected dtype for {patch_id}: "
            f"{tensor.dtype}; expected torch.float32"
        )

    return tensor


def validate_upload_tensor(
    tensor: torch.Tensor,
) -> dict:
    """
    Validate the final tensor against the canonical
    SatQueryAI preprocessing contract.
    """

    if not isinstance(tensor, torch.Tensor):
        raise UploadPreprocessingError(
            "Processed input is not a torch.Tensor."
        )

    if tuple(tensor.shape) != (
        14,
        120,
        120,
    ):
        raise UploadPreprocessingError(
            "Processed input must have shape "
            "[14,120,120], got "
            f"{tuple(tensor.shape)}"
        )

    if tensor.dtype != torch.float32:
        raise UploadPreprocessingError(
            "Processed input must use torch.float32, got "
            f"{tensor.dtype}"
        )

    if not torch.isfinite(tensor).all().item():
        raise UploadPreprocessingError(
            "Processed input contains NaN or Inf."
        )

    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())

    if minimum < 0.0 or maximum > 1.0:
        raise UploadPreprocessingError(
            "Processed input contains values outside [0,1]: "
            f"min={minimum}, max={maximum}"
        )

    return {
        "channel_names": list(CHANNEL_NAMES),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": minimum,
        "max": maximum,
        "elements": int(tensor.numel()),
    }


if __name__ == "__main__":
    import pandas as pd

    from preprocessing.preprocess_dataset import (
        build_s1_paths,
        build_s2_paths,
        process_pair,
    )

    selected = pd.read_csv(
        r"data\splits\selected_pairs.csv"
    )

    metadata = pd.read_parquet(
        r"data\raw\BigEarthNet_LMDB\metadata.parquet",
        columns=[
            "patch_id",
            "s1_name",
        ],
    )

    row = (
        selected
        .merge(
            metadata,
            on="patch_id",
            how="inner",
        )
        .iloc[0]
    )

    patch_id = row.patch_id
    s1_name = row.s1_name

    s1_paths = build_s1_paths(s1_name)
    s2_paths = build_s2_paths(patch_id)

    adapter_tensor = preprocess_uploaded_pair(
        s1_files=s1_paths,
        s2_files=s2_paths,
        patch_id=patch_id,
    )

    reference_tensor = process_pair(
        patch_id,
        s1_name,
    )

    metadata_result = validate_upload_tensor(
        adapter_tensor
    )

    max_difference = (
        adapter_tensor - reference_tensor
    ).abs().max().item()

    print(
        "SATQUERYAI — UPLOAD PREPROCESSOR TEST"
    )
    print("=" * 80)
    print()
    print(
        f"patch_id: {patch_id}"
    )
    print(
        f"shape: {metadata_result['shape']}"
    )
    print(
        f"dtype: {metadata_result['dtype']}"
    )
    print(
        f"min: {metadata_result['min']}"
    )
    print(
        f"max: {metadata_result['max']}"
    )
    print(
        f"elements: {metadata_result['elements']}"
    )
    print()
    print(
        "REFERENCE process_pair() COMPARISON"
    )
    print(
        f"max absolute difference: {max_difference}"
    )
    print()

    if max_difference == 0.0:
        print(
            "UPLOAD PREPROCESSOR TEST: PASS"
        )
    else:
        raise RuntimeError(
            "Upload preprocessing does not exactly match "
            "the existing process_pair() output."
        )