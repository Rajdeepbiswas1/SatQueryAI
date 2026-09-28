from pathlib import Path
from typing import Iterable

import torch

from preprocessing.preprocess_dataset import (
    CHANNEL_NAMES,
    S2_BANDS,
    build_s1_paths,
    build_s2_paths,
)

from backend.upload_preprocessor import (
    UploadPreprocessingError,
    preprocess_uploaded_pair,
    validate_upload_tensor,
)


SUPPORTED_SUFFIXES = {
    ".tif",
    ".tiff",
}


def identify_channel(path: Path) -> str | None:
    """
    Identify the SatQueryAI channel represented by a TIFF filename.

    Accepted examples:
        image_VH.tif
        image_VV.tif
        image_B04.tif
        B04.tiff
    """

    name = path.name
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        return None

    stem = path.stem.upper()

    for channel in CHANNEL_NAMES:
        if (
            stem == channel
            or stem.endswith(f"_{channel}")
        ):
            return channel

    return None


def build_upload_channel_maps(
    paths: Iterable[Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    """
    Convert uploaded TIFF paths into the exact S1/S2 channel maps
    required by preprocess_uploaded_pair().
    """

    paths = [
        Path(path)
        for path in paths
    ]

    if len(paths) != 14:
        raise UploadPreprocessingError(
            "SatQueryAI currently requires exactly 14 TIFF files "
            "for the trained S1+S2 input contract: "
            "2 Sentinel-1 channels + 12 Sentinel-2 bands."
        )

    channel_to_path: dict[str, Path] = {}

    for path in paths:

        if not path.is_file():
            raise UploadPreprocessingError(
                f"Uploaded file does not exist: {path}"
            )

        channel = identify_channel(path)

        if channel is None:
            raise UploadPreprocessingError(
                "Could not identify a supported SatQueryAI channel "
                f"from filename: {path.name}"
            )

        if channel in channel_to_path:
            raise UploadPreprocessingError(
                f"Duplicate channel '{channel}' detected: "
                f"{channel_to_path[channel].name} and {path.name}"
            )

        channel_to_path[channel] = path

    expected = set(CHANNEL_NAMES)
    actual = set(channel_to_path.keys())

    missing = [
        channel
        for channel in CHANNEL_NAMES
        if channel not in actual
    ]

    extra = sorted(actual - expected)

    if missing:
        raise UploadPreprocessingError(
            "Missing required channels: "
            + ", ".join(missing)
        )

    if extra:
        raise UploadPreprocessingError(
            "Unexpected channels: "
            + ", ".join(extra)
        )

    s1_files = {
        "VH": channel_to_path["VH"],
        "VV": channel_to_path["VV"],
    }

    s2_files = {
        band: channel_to_path[band]
        for band in S2_BANDS
    }

    return s1_files, s2_files


def preprocess_upload_bundle(
    paths: Iterable[Path],
    patch_id: str = "uploaded_pair",
) -> tuple[torch.Tensor, dict, dict[str, str]]:
    """
    Validate filenames, preprocess the 14 TIFF files using the exact
    project preprocessing rules, and validate the resulting tensor.

    Returns:
        tensor:
            [14,120,120] float32 tensor

        metadata:
            canonical preprocessing metadata

        file_map:
            channel -> original file path
    """

    s1_files, s2_files = build_upload_channel_maps(
        paths
    )

    tensor = preprocess_uploaded_pair(
        s1_files=s1_files,
        s2_files=s2_files,
        patch_id=patch_id,
    )

    metadata = validate_upload_tensor(
        tensor
    )

    file_map = {
        "VH": str(s1_files["VH"]),
        "VV": str(s1_files["VV"]),
    }

    for band in S2_BANDS:
        file_map[band] = str(
            s2_files[band]
        )

    return (
        tensor,
        metadata,
        file_map,
    )


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Integration test against one real BigEarthNet pair.
    #
    # This does NOT process the 10k dataset. It processes exactly
    # one pair and compares the upload-bundle path against the
    # already-verified process_pair() path.
    # -------------------------------------------------------------

    import pandas as pd

    from preprocessing.preprocess_dataset import (
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

    s1_paths = build_s1_paths(
        s1_name
    )

    s2_paths = build_s2_paths(
        patch_id
    )

    all_paths = list(
        s1_paths.values()
    ) + list(
        s2_paths.values()
    )

    upload_tensor, upload_metadata, file_map = (
        preprocess_upload_bundle(
            paths=all_paths,
            patch_id=patch_id,
        )
    )

    reference_tensor = process_pair(
        patch_id,
        s1_name,
    )

    max_difference = (
        upload_tensor - reference_tensor
    ).abs().max().item()

    print(
        "SATQUERYAI — UPLOAD SERVICE TEST"
    )
    print("=" * 80)
    print()
    print(
        f"patch_id: {patch_id}"
    )
    print(
        f"files detected: {len(file_map)}"
    )
    print(
        f"shape: {upload_metadata['shape']}"
    )
    print(
        f"dtype: {upload_metadata['dtype']}"
    )
    print(
        f"min: {upload_metadata['min']}"
    )
    print(
        f"max: {upload_metadata['max']}"
    )
    print()
    print(
        "CHANNEL MAP"
    )

    for channel in CHANNEL_NAMES:
        print(
            f"  {channel}: "
            f"{Path(file_map[channel]).name}"
        )

    print()
    print(
        "REFERENCE process_pair() COMPARISON"
    )
    print(
        f"max absolute difference: {max_difference}"
    )
    print()

    if max_difference != 0.0:
        raise RuntimeError(
            "Upload service does not exactly match "
            "the existing preprocessing pipeline."
        )

    print(
        "UPLOAD SERVICE TEST: PASS"
    )