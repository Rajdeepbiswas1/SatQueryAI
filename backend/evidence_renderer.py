from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
from PIL import Image, ImageDraw


class EvidenceRenderingError(Exception):
    """Raised when visual evidence rendering fails."""


RGB_BANDS = (
    "B04",  # Red
    "B03",  # Green
    "B02",  # Blue
)


def _read_band(
    path: Path,
) -> np.ndarray:

    path = Path(path)

    if not path.is_file():
        raise EvidenceRenderingError(
            f"Band file does not exist: {path}"
        )

    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(
                np.float32,
                copy=False,
            )

    except Exception as exc:
        raise EvidenceRenderingError(
            f"Could not read raster band {path}: {exc}"
        ) from exc

    if data.ndim != 2:
        raise EvidenceRenderingError(
            f"Expected 2D raster for {path.name}, "
            f"got shape {data.shape}"
        )

    if not np.isfinite(data).all():
        raise EvidenceRenderingError(
            f"Raster contains NaN or Inf: {path.name}"
        )

    return data


def _read_combined_tiff(
    path: Path,
    modality: str,
) -> tuple[np.ndarray, list[str]]:

    path = Path(path)

    if not path.is_file():
        raise EvidenceRenderingError(
            f"Combined TIFF does not exist: {path}"
        )

    modality = modality.lower().strip()

    try:
        with rasterio.open(path) as src:

            if modality == "s2":

                if src.count != 9:
                    raise EvidenceRenderingError(
                        "Expected a 9-channel Sentinel-2 TIFF, "
                        f"got {src.count} channels: {path.name}"
                    )

                # Combined S2 application TIFF channel order:
                # 1 -> B04 (Red)
                # 2 -> B03 (Green)
                # 3 -> B02 (Blue)
                red = src.read(1).astype(
                    np.float32,
                    copy=False,
                )

                green = src.read(2).astype(
                    np.float32,
                    copy=False,
                )

                blue = src.read(3).astype(
                    np.float32,
                    copy=False,
                )

                bands = [
                    red,
                    green,
                    blue,
                ]

                band_names = [
                    "B04",
                    "B03",
                    "B02",
                ]

            elif modality == "s1":

                if src.count != 2:
                    raise EvidenceRenderingError(
                        "Expected a 2-channel Sentinel-1 TIFF "
                        "(VH + VV), "
                        f"got {src.count} channels: {path.name}"
                    )

                # Combined S1 application TIFF channel order:
                # 1 -> VH
                # 2 -> VV
                #
                # For visual evidence we use VH as a grayscale
                # background and replicate it into RGB channels.
                vh = src.read(1).astype(
                    np.float32,
                    copy=False,
                )

                bands = [
                    vh,
                    vh,
                    vh,
                ]

                band_names = [
                    "VH",
                ]

            else:
                raise EvidenceRenderingError(
                    f"Unsupported modality for evidence rendering: "
                    f"{modality!r}"
                )

    except EvidenceRenderingError:
        raise

    except Exception as exc:
        raise EvidenceRenderingError(
            f"Could not read combined raster {path}: {exc}"
        ) from exc

    reference_shape = bands[0].shape

    if len(reference_shape) != 2:
        raise EvidenceRenderingError(
            f"Expected 2D raster channels in {path.name}, "
            f"got shape {reference_shape}"
        )

    for index, band in enumerate(bands):

        if band.ndim != 2:
            raise EvidenceRenderingError(
                f"Expected 2D channel in {path.name}, "
                f"got shape {band.shape}"
            )

        if not np.isfinite(band).all():
            raise EvidenceRenderingError(
                f"Raster contains NaN or Inf: "
                f"{path.name}, channel {index + 1}"
            )

        bands[index] = _resize_to_reference(
            band,
            reference_shape,
        )

    return (
        np.stack(
            bands,
            axis=-1,
        ),
        band_names,
    )


def _resize_to_reference(
    data: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:

    target_height, target_width = target_shape

    if data.shape == target_shape:
        return data

    image = Image.fromarray(
        data,
        mode="F",
    )

    image = image.resize(
        (target_width, target_height),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    )


def _percentile_stretch(
    data: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
) -> np.ndarray:

    low = float(
        np.percentile(
            data,
            lower,
        )
    )

    high = float(
        np.percentile(
            data,
            upper,
        )
    )

    if high <= low:
        minimum = float(
            np.min(data)
        )

        maximum = float(
            np.max(data)
        )

        if maximum <= minimum:
            return np.zeros_like(
                data,
                dtype=np.float32,
            )

        low = minimum
        high = maximum

    stretched = (
        np.clip(
            data,
            low,
            high,
        )
        - low
    ) / (
        high - low
    )

    return stretched.astype(
        np.float32,
        copy=False,
    )


def _normalize_bbox(
    bbox: Sequence[float],
) -> tuple[float, float, float, float]:

    if len(bbox) != 4:
        raise EvidenceRenderingError(
            "Bounding box must contain exactly "
            "[x1, y1, x2, y2]."
        )

    try:
        x1, y1, x2, y2 = (
            float(value)
            for value in bbox
        )

    except Exception as exc:
        raise EvidenceRenderingError(
            f"Invalid bounding box: {bbox}"
        ) from exc

    values = [
        x1,
        y1,
        x2,
        y2,
    ]

    if not all(
        np.isfinite(value)
        for value in values
    ):
        raise EvidenceRenderingError(
            f"Bounding box contains non-finite values: "
            f"{bbox}"
        )

    # Grounding specialist boxes are expected to be normalized
    # to the image coordinate system.
    x1 = min(
        max(x1, 0.0),
        1.0,
    )

    y1 = min(
        max(y1, 0.0),
        1.0,
    )

    x2 = min(
        max(x2, 0.0),
        1.0,
    )

    y2 = min(
        max(y2, 0.0),
        1.0,
    )

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return (
        x1,
        y1,
        x2,
        y2,
    )


def create_true_color_preview(
    s2_files: Mapping[str, Path],
) -> Image.Image:
    """
    Create a true-color Sentinel-2 preview using:

        B04 -> red
        B03 -> green
        B02 -> blue

    Each band receives an independent percentile stretch.

    This is the legacy separate-band rendering path.
    """

    missing = [
        band
        for band in RGB_BANDS
        if band not in s2_files
    ]

    if missing:
        raise EvidenceRenderingError(
            "Missing RGB Sentinel-2 bands: "
            + ", ".join(missing)
        )

    bands = {
        band: _read_band(
            Path(s2_files[band])
        )
        for band in RGB_BANDS
    }

    reference_shape = bands[
        "B04"
    ].shape

    for band in RGB_BANDS:
        bands[band] = _resize_to_reference(
            bands[band],
            reference_shape,
        )

    red = _percentile_stretch(
        bands["B04"]
    )

    green = _percentile_stretch(
        bands["B03"]
    )

    blue = _percentile_stretch(
        bands["B02"]
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb = np.clip(
        rgb * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


def create_combined_tiff_preview(
    path: Path,
    modality: str,
) -> tuple[Image.Image, list[str]]:
    """
    Create a browser-displayable RGB preview from a combined TIFF.

    Supported inputs:

        S2:
            9-channel TIFF
            channel 1 -> B04
            channel 2 -> B03
            channel 3 -> B02

        S1:
            2-channel TIFF
            channel 1 -> VH
            rendered as grayscale RGB
    """

    data, band_names = _read_combined_tiff(
        path=path,
        modality=modality,
    )

    red = _percentile_stretch(
        data[:, :, 0]
    )

    green = _percentile_stretch(
        data[:, :, 1]
    )

    blue = _percentile_stretch(
        data[:, :, 2]
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    rgb = np.clip(
        rgb * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return (
        Image.fromarray(
            rgb,
            mode="RGB",
        ),
        band_names,
    )


def render_grounding_evidence(
    s2_files: Mapping[str, Path] | str | Path,
    bbox: Sequence[float],
    output_path: Path,
    query: str | None = None,
    modality: str | None = None,
) -> dict[str, Any]:
    """
    Render a browser-displayable RGB image with the predicted
    grounding bounding box.

    Supported source formats:

    1. Legacy separate Sentinel-2 bands:

        s2_files = {
            "B04": Path(...),
            "B03": Path(...),
            "B02": Path(...),
        }

    2. New combined application TIFF:

        s2_files = Path(...)
        modality = "s2"

    3. New combined Sentinel-1 TIFF:

        s2_files = Path(...)
        modality = "s1"

    bbox:
        normalized [x1, y1, x2, y2]
    """

    display_bands: list[str]

    if isinstance(
        s2_files,
        (str, Path),
    ):

        if modality is None:
            raise EvidenceRenderingError(
                "Modality is required when rendering "
                "a combined TIFF."
            )

        image, display_bands = (
            create_combined_tiff_preview(
                path=Path(s2_files),
                modality=modality,
            )
        )

    else:

        image = create_true_color_preview(
            s2_files
        )

        display_bands = list(
            RGB_BANDS
        )

    x1, y1, x2, y2 = _normalize_bbox(
        bbox
    )

    width, height = image.size

    left = int(
        round(
            x1 * width
        )
    )

    top = int(
        round(
            y1 * height
        )
    )

    right = int(
        round(
            x2 * width
        )
    )

    bottom = int(
        round(
            y2 * height
        )
    )

    left = max(
        0,
        min(
            left,
            width - 1,
        ),
    )

    top = max(
        0,
        min(
            top,
            height - 1,
        ),
    )

    right = max(
        left + 1,
        min(
            right,
            width - 1,
        ),
    )

    bottom = max(
        top + 1,
        min(
            bottom,
            height - 1,
        ),
    )

    annotated = image.copy()

    draw = ImageDraw.Draw(
        annotated
    )

    draw.rectangle(
        [
            left,
            top,
            right,
            bottom,
        ],
        outline="red",
        width=max(
            3,
            min(
                width,
                height,
            ) // 100,
        ),
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated.save(
        output_path,
        format="PNG",
    )

    return {
        "available": True,
        "path": str(
            output_path
        ),
        "format": "PNG",
        "image_width": width,
        "image_height": height,
        "bbox_normalized": [
            x1,
            y1,
            x2,
            y2,
        ],
        "bbox_pixels": [
            left,
            top,
            right,
            bottom,
        ],
        "rgb_bands": display_bands,
        "query": query,
    }


if __name__ == "__main__":

    from preprocessing.preprocess_dataset import (
        build_s2_paths,
    )

    print(
        "SATQUERYAI — GROUNDING EVIDENCE RENDERER TEST"
    )
    print("=" * 80)
    print()

    patch_id = (
        "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"
    )

    s2_paths = build_s2_paths(
        patch_id
    )

    bbox = [
        0.4842033386230469,
        0.32561010192299194,
        0.8260698318481445,
        0.7012012600898743,
    ]

    output_path = (
        Path("outputs")
        / "grounding_evidence_test.png"
    )

    result = render_grounding_evidence(
        s2_files=s2_paths,
        bbox=bbox,
        output_path=output_path,
        query=(
            "Where is the largest urban area "
            "in the image?"
        ),
    )

    print(
        "RENDER RESULT"
    )
    print(
        f"  available: {result['available']}"
    )
    print(
        f"  format: {result['format']}"
    )
    print(
        f"  image size: "
        f"{result['image_width']} x "
        f"{result['image_height']}"
    )
    print(
        f"  bbox normalized: "
        f"{result['bbox_normalized']}"
    )
    print(
        f"  bbox pixels: "
        f"{result['bbox_pixels']}"
    )
    print(
        f"  output: {result['path']}"
    )

    if not Path(
        result["path"]
    ).is_file():
        raise RuntimeError(
            "Evidence image was not created."
        )

    print()
    print(
        "GROUNDING EVIDENCE RENDERER TEST: PASS"
    )