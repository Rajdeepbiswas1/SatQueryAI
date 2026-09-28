from pathlib import Path
import shutil

import numpy as np
import rasterio


PROJECT_ROOT = Path(r"D:\SatQueryAI")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "test_inputs"
)

SINGLE_ROOT = OUTPUT_ROOT / "single"
CROSS_ROOT = OUTPUT_ROOT / "cross_modal"
TEMPORAL_ROOT = OUTPUT_ROOT / "temporal"


# ---------------------------------------------------------------------------
# Verified BigEarthNet source pair
# ---------------------------------------------------------------------------

S1_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images_raw"
    / "BigEarthNet-S1"
    / "S1A_IW_GRDH_1SDV_20180526T045642"
    / "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
)

S2_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images_raw"
    / "BigEarthNet-S2"
    / "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU"
    / "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"
)


S1_PATCH_ID = (
    "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
)

S2_PATCH_ID = (
    "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"
)


S1_BANDS = [
    "VH",
    "VV",
]

S2_BANDS = [
    "B04",
    "B03",
    "B02",
    "B05",
    "B06",
    "B07",
    "B08",
    "B11",
    "B12",
]


# ---------------------------------------------------------------------------
# Temporal source pair
# ---------------------------------------------------------------------------

TEMPORAL_SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_raw"
)

TEMPORAL_ID = "00003"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def stack_tiff(
    source_dir: Path,
    patch_id: str,
    band_names: list[str],
    output_path: Path,
) -> None:

    source_paths = [
        source_dir / f"{patch_id}_{band}.tif"
        for band in band_names
    ]

    for path, band in zip(source_paths, band_names):
        require_file(
            path,
            f"Source band {band}",
        )

    arrays = []

    reference_profile = None
    reference_height = None
    reference_width = None
    reference_transform = None
    reference_crs = None

    for index, (path, band) in enumerate(
        zip(source_paths, band_names)
    ):

        with rasterio.open(path) as src:

            # Use the first band as the spatial reference.
            if index == 0:

                reference_height = src.height
                reference_width = src.width
                reference_profile = src.profile.copy()
                reference_transform = src.transform
                reference_crs = src.crs

                data = src.read(
                    1
                ).astype(
                    np.float32
                )

            else:

                if src.crs != reference_crs:
                    raise RuntimeError(
                        f"CRS mismatch in {path}"
                    )

                # Resample bands with lower native resolution
                # to the reference band's dimensions.
                data = src.read(
                    1,
                    out_shape=(
                        reference_height,
                        reference_width,
                    ),
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(
                    np.float32
                )

            arrays.append(data)

    stacked = np.stack(
        arrays,
        axis=0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_profile.update(
        count=len(band_names),
        height=reference_height,
        width=reference_width,
        dtype="float32",
        compress="none",
        transform=reference_transform,
    )

    with rasterio.open(
        output_path,
        "w",
        **reference_profile,
    ) as dst:

        dst.write(
            stacked
        )

        for channel_index, band_name in enumerate(
            band_names,
            start=1,
        ):

            dst.set_band_description(
                channel_index,
                band_name,
            )

    print(
        f"Created: {output_path}"
    )

    print(
        f"  Channels: {len(band_names)}"
    )

    print(
        f"  Shape: {stacked.shape}"
    )

    print(
        f"  Dtype: {stacked.dtype}"
    )


def copy_temporal_pair() -> None:

    source_before = (
        TEMPORAL_SOURCE_ROOT
        / "im1"
        / f"{TEMPORAL_ID}.png"
    )

    source_after = (
        TEMPORAL_SOURCE_ROOT
        / "im2"
        / f"{TEMPORAL_ID}.png"
    )

    require_file(
        source_before,
        "Temporal BEFORE image",
    )

    require_file(
        source_after,
        "Temporal AFTER image",
    )

    before_output = (
        TEMPORAL_ROOT
        / f"before_{TEMPORAL_ID}.png"
    )

    after_output = (
        TEMPORAL_ROOT
        / f"after_{TEMPORAL_ID}.png"
    )

    TEMPORAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_before,
        before_output,
    )

    shutil.copy2(
        source_after,
        after_output,
    )

    print(
        f"Created: {before_output}"
    )

    print(
        f"Created: {after_output}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("SATQUERYAI — CREATE SMALL END-TO-END TEST INPUT SET")
    print("=" * 72)

    # Create output directories.
    SINGLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROSS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMPORAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Build one real optical / multispectral 9-channel TIFF.
    # -----------------------------------------------------------------------

    optical_single = (
        SINGLE_ROOT
        / "optical_test.tif"
    )

    stack_tiff(
        source_dir=S2_SOURCE_DIR,
        patch_id=S2_PATCH_ID,
        band_names=S2_BANDS,
        output_path=optical_single,
    )

    # -----------------------------------------------------------------------
    # Build one real SAR 2-channel TIFF.
    # -----------------------------------------------------------------------

    sar_single = (
        SINGLE_ROOT
        / "sar_test.tif"
    )

    stack_tiff(
        source_dir=S1_SOURCE_DIR,
        patch_id=S1_PATCH_ID,
        band_names=S1_BANDS,
        output_path=sar_single,
    )

    # -----------------------------------------------------------------------
    # Reuse those exact same real files for cross-modal testing.
    # -----------------------------------------------------------------------

    optical_cross = (
        CROSS_ROOT
        / "optical_test.tif"
    )

    sar_cross = (
        CROSS_ROOT
        / "sar_test.tif"
    )

    shutil.copy2(
        optical_single,
        optical_cross,
    )

    shutil.copy2(
        sar_single,
        sar_cross,
    )

    print(
        f"Created: {optical_cross}"
    )

    print(
        f"Created: {sar_cross}"
    )

    # -----------------------------------------------------------------------
    # Temporal BEFORE / AFTER pair.
    # -----------------------------------------------------------------------

    copy_temporal_pair()

    # -----------------------------------------------------------------------
    # Final verification.
    # -----------------------------------------------------------------------

    expected_files = [
        optical_single,
        sar_single,
        optical_cross,
        sar_cross,
        TEMPORAL_ROOT / "before_00003.png",
        TEMPORAL_ROOT / "after_00003.png",
    ]

    for path in expected_files:
        require_file(
            path,
            "Generated test file",
        )

    print()
    print("=" * 72)
    print("TEST INPUT CREATION: PASS")
    print("=" * 72)
    print()
    print(
        f"Test root:\n{OUTPUT_ROOT}"
    )

    print()
    print("Files:")

    for path in expected_files:
        print(
            f"  {path.relative_to(OUTPUT_ROOT)}"
        )


if __name__ == "__main__":
    main()