from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(r"D:\SatQueryAI")
TEST_ROOT = PROJECT_ROOT / "data" / "test_inputs"

SINGLE_ROOT = TEST_ROOT / "single"
PREVIEW_ROOT = TEST_ROOT / "previews"


def normalize_percentile(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)

    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.uint8)

    values = data[valid]

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = (data - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)

    return (scaled * 255.0).astype(np.uint8)


def create_optical_preview() -> None:
    source = SINGLE_ROOT / "optical_test.tif"
    output = PREVIEW_ROOT / "optical_preview.png"

    if not source.is_file():
        raise FileNotFoundError(source)

    with rasterio.open(source) as src:

        if src.count != 9:
            raise RuntimeError(
                f"Expected 9 optical bands, found {src.count}"
            )

        # Channel order in our test TIFF:
        # 1 = B04
        # 2 = B03
        # 3 = B02
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)

    rgb = np.stack(
        [
            normalize_percentile(red),
            normalize_percentile(green),
            normalize_percentile(blue),
        ],
        axis=-1,
    )

    Image.fromarray(rgb, mode="RGB").save(output)

    print(f"Created: {output}")


def create_sar_preview() -> None:
    source = SINGLE_ROOT / "sar_test.tif"
    output = PREVIEW_ROOT / "sar_preview.png"

    if not source.is_file():
        raise FileNotFoundError(source)

    with rasterio.open(source) as src:

        if src.count != 2:
            raise RuntimeError(
                f"Expected 2 SAR bands, found {src.count}"
            )

        # Channel order:
        # 1 = VH
        # 2 = VV
        vh = src.read(1).astype(np.float32)

    preview = normalize_percentile(vh)

    Image.fromarray(
        preview,
        mode="L",
    ).save(output)

    print(f"Created: {output}")


def main() -> None:
    print("=" * 72)
    print("SATQUERYAI — CREATE TEST IMAGE PREVIEWS")
    print("=" * 72)

    PREVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_optical_preview()
    create_sar_preview()

    print()
    print("=" * 72)
    print("TEST PREVIEW CREATION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()