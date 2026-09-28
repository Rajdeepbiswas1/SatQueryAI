from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "evaluation_manifest.csv"
)

METADATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BigEarthNet_LMDB"
    / "metadata.parquet"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images_eval"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_eval"
)

S2_BANDS = [
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

CHANNEL_NAMES = ["VH", "VV"] + S2_BANDS


# IMPORTANT:
# These are the SAME normalization values used for training.
NORMALIZATION = {
    "VH": (-30.3527, -12.2195),
    "VV": (-23.3586, -6.3003),

    "B01": (29.0, 859.0),
    "B02": (89.0, 1056.0),
    "B03": (101.0, 1360.0),
    "B04": (52.0, 1678.0),
    "B05": (43.0, 2000.0),
    "B06": (30.0, 3423.0),
    "B07": (32.0, 4095.0),
    "B08": (30.0, 4380.0),
    "B09": (1.0, 4270.0),
    "B11": (17.0, 3382.0),
    "B12": (17.0, 2629.0),
    "B8A": (27.0, 4362.0),
}


def normalize(
    data: np.ndarray,
    p05: float,
    p95: float,
) -> np.ndarray:

    data = data.astype(
        np.float32,
        copy=False,
    )

    data = np.clip(
        data,
        p05,
        p95,
    )

    denominator = p95 - p05

    if denominator <= 0:
        raise ValueError(
            f"Invalid normalization range: "
            f"{p05}, {p95}"
        )

    return (data - p05) / denominator


def read_and_normalize_s1(
    path: Path,
    channel: str,
) -> np.ndarray:

    with rasterio.open(path) as src:
        data = src.read(1)

    p05, p95 = NORMALIZATION[channel]

    return normalize(
        data,
        p05,
        p95,
    )


def read_and_normalize_s2(
    path: Path,
    channel: str,
) -> np.ndarray:

    with rasterio.open(path) as src:

        data = src.read(
            1,
            out_shape=(120, 120),
            resampling=Resampling.bilinear,
        )

    p05, p95 = NORMALIZATION[channel]

    return normalize(
        data,
        p05,
        p95,
    )


def build_s1_paths(
    s1_name: str,
) -> dict[str, Path]:

    parts = s1_name.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected S1 name: {s1_name}"
        )

    acquisition = "_".join(
        parts[:-3]
    )

    base = (
        IMAGE_ROOT
        / "BigEarthNet-S1"
        / acquisition
        / s1_name
    )

    return {
        "VH": base / f"{s1_name}_VH.tif",
        "VV": base / f"{s1_name}_VV.tif",
    }


def build_s2_paths(
    patch_id: str,
) -> dict[str, Path]:

    parts = patch_id.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected patch ID: {patch_id}"
        )

    acquisition = "_".join(
        parts[:-2]
    )

    base = (
        IMAGE_ROOT
        / "BigEarthNet-S2"
        / acquisition
        / patch_id
    )

    return {
        band: base / f"{patch_id}_{band}.tif"
        for band in S2_BANDS
    }


def validate_paths(
    paths: dict[str, Path],
    patch_id: str,
    sensor: str,
) -> None:

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.is_file()
    ]

    if missing:

        raise FileNotFoundError(
            f"Missing {sensor} files for "
            f"{patch_id}:\n"
            + "\n".join(missing)
        )


def process_pair(
    patch_id: str,
    s1_name: str,
) -> torch.Tensor:

    s1_paths = build_s1_paths(
        s1_name
    )

    s2_paths = build_s2_paths(
        patch_id
    )

    validate_paths(
        s1_paths,
        patch_id,
        "Sentinel-1",
    )

    validate_paths(
        s2_paths,
        patch_id,
        "Sentinel-2",
    )

    channels = []

    # Sentinel-1
    channels.append(
        read_and_normalize_s1(
            s1_paths["VH"],
            "VH",
        )
    )

    channels.append(
        read_and_normalize_s1(
            s1_paths["VV"],
            "VV",
        )
    )

    # Sentinel-2
    for band in S2_BANDS:

        channels.append(
            read_and_normalize_s2(
                s2_paths[band],
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

    if array.shape != (
        14,
        120,
        120,
    ):
        raise RuntimeError(
            f"Unexpected tensor shape "
            f"for {patch_id}: "
            f"{array.shape}"
        )

    tensor = torch.from_numpy(
        array
    )

    if not torch.isfinite(
        tensor
    ).all():
        raise RuntimeError(
            f"Non-finite values found "
            f"in {patch_id}"
        )

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    return tensor


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"patch_id": str},
    )

    metadata = pd.read_parquet(
        METADATA,
        columns=[
            "patch_id",
            "s1_name",
        ],
    )

    metadata["patch_id"] = (
        metadata["patch_id"]
        .astype(str)
    )

    pairs = manifest.merge(
        metadata,
        on="patch_id",
        how="left",
    )

    if pairs["s1_name"].isna().any():

        count = pairs[
            "s1_name"
        ].isna().sum()

        raise RuntimeError(
            f"{count:,} evaluation patches "
            "have no S1 metadata."
        )

    total = len(pairs)
    processed = 0
    skipped = 0
    failed = 0

    print(
        "=== SatQueryAI evaluation "
        "preprocessing ==="
    )
    print()
    print(
        f"Evaluation pairs: {total:,}"
    )
    print()

    for index, row in enumerate(
        pairs.itertuples(
            index=False
        ),
        start=1,
    ):

        patch_id = row.patch_id
        dataset_split = (
            row.dataset_split
        )

        output_file = (
            OUTPUT_DIR
            / f"{patch_id}.pt"
        )

        if output_file.exists():

            skipped += 1
            continue

        try:

            tensor = process_pair(
                patch_id,
                row.s1_name,
            )

            torch.save(
                {
                    "patch_id": patch_id,
                    "dataset_split": (
                        dataset_split
                    ),
                    "channel_names": (
                        CHANNEL_NAMES
                    ),
                    "image": tensor,
                },
                output_file,
            )

            processed += 1

        except Exception as exc:

            failed += 1

            print(
                f"\nERROR processing "
                f"{patch_id}:",
                file=sys.stderr,
            )

            print(
                str(exc),
                file=sys.stderr,
            )

            continue

        if (
            index % 100 == 0
            or index == total
        ):

            print(
                f"[{index:,}/{total:,}] "
                f"New: {processed:,} | "
                f"Existing: {skipped:,} | "
                f"Failed: {failed:,}",
                flush=True,
            )

    actual = len(
        list(
            OUTPUT_DIR.glob("*.pt")
        )
    )

    print()
    print(
        "=== Preprocessing finished ==="
    )
    print(
        f"Newly processed: {processed:,}"
    )
    print(
        f"Already existed: {skipped:,}"
    )
    print(
        f"Failed:          {failed:,}"
    )
    print(
        f"PT files present: {actual:,}"
    )

    if (
        actual == total
        and failed == 0
    ):
        print()
        print(
            "SUCCESS: all evaluation "
            "pairs are preprocessed."
        )
    else:
        print()
        print(
            "WARNING: evaluation "
            "preprocessing is incomplete."
        )


if __name__ == "__main__":
    main()