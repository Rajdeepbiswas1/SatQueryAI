from pathlib import Path
import numpy as np
import rasterio


PROJECT_ROOT = Path(r"D:\SatQueryAI")

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images_raw"
)

PATCH_ID = "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"

S1_FILES = {
    "VH": (
        IMAGE_ROOT
        / "BigEarthNet-S1"
        / "S1A_IW_GRDH_1SDV_20180526T045642"
        / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
        / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16_VH.tif"
    ),
    "VV": (
        IMAGE_ROOT
        / "BigEarthNet-S1"
        / "S1A_IW_GRDH_1SDV_20180526T045642"
        / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
        / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16_VV.tif"
    ),
}

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


def inspect_file(name: str, path: Path) -> None:
    with rasterio.open(path) as src:
        data = src.read(1)

        print(f"{name}")
        print(f"  Size:       {src.width} x {src.height}")
        print(f"  Dtype:      {src.dtypes[0]}")
        print(f"  CRS:        {src.crs}")
        print(f"  Min:        {data.min()}")
        print(f"  Max:        {data.max()}")
        print(f"  Mean:       {data.mean():.4f}")
        print(f"  Zeros:      {(data == 0).mean() * 100:.2f}%")
        print()


print("=== Sentinel-1 ===\n")

for name, path in S1_FILES.items():
    inspect_file(name, path)

print("=== Sentinel-2 ===\n")

for band in S2_BANDS:
    path = S2_DIR / f"{PATCH_ID}_{band}.tif"
    inspect_file(band, path)