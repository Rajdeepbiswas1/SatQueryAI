from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images_raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_test"
)

PATCH_ID = "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"

S1_DIR = (
    IMAGE_ROOT
    / "BigEarthNet-S1"
    / "S1A_IW_GRDH_1SDV_20180526T045642"
    / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
)

S2_DIR = (
    IMAGE_ROOT
    / "BigEarthNet-S2"
    / "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU"
    / PATCH_ID
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

# Dataset-wide robust normalization limits from dataset_stats.txt
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


def normalize(data: np.ndarray, p05: float, p95: float) -> np.ndarray:
    data = data.astype(np.float32)

    data = np.clip(data, p05, p95)

    return (data - p05) / (p95 - p05)


def read_s1(name: str) -> np.ndarray:
    path = S1_DIR / (
        f"S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16_{name}.tif"
    )

    with rasterio.open(path) as src:
        data = src.read(1)

    p05, p95 = NORMALIZATION[name]

    return normalize(data, p05, p95)


def read_s2(band: str) -> np.ndarray:
    path = S2_DIR / f"{PATCH_ID}_{band}.tif"

    with rasterio.open(path) as src:

        data = src.read(
            1,
            out_shape=(120, 120),
            resampling=Resampling.bilinear,
        )

    p05, p95 = NORMALIZATION[band]

    return normalize(data, p05, p95)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    channels = []

    print("=== Preprocessing test pair ===")
    print(f"Patch: {PATCH_ID}")
    print()

    # S1
    print("Reading S1...")

    vh = read_s1("VH")
    vv = read_s1("VV")

    channels.append(vh)
    channels.append(vv)

    # S2
    print("Reading S2...")

    for band in S2_BANDS:
        print(f"  {band}")
        channels.append(read_s2(band))

    tensor = np.stack(channels, axis=0)

    tensor = torch.from_numpy(tensor)

    print()
    print("=== Result ===")
    print("Shape:", tuple(tensor.shape))
    print("Dtype:", tensor.dtype)
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())
    print("Mean:", tensor.mean().item())

    print()
    print("Channel statistics:")

    names = ["VH", "VV"] + S2_BANDS

    for i, name in enumerate(names):
        channel = tensor[i]

        print(
            f"{name:>3}: "
            f"shape={tuple(channel.shape)}, "
            f"min={channel.min().item():.4f}, "
            f"max={channel.max().item():.4f}, "
            f"mean={channel.mean().item():.4f}"
        )

    output_file = OUTPUT_DIR / f"{PATCH_ID}.pt"

    torch.save(
        {
            "patch_id": PATCH_ID,
            "channel_names": names,
            "image": tensor,
        },
        output_file,
    )

    print()
    print("Saved:")
    print(output_file)


if __name__ == "__main__":
    main()