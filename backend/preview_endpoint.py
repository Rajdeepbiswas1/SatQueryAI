from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
from fastapi import APIRouter, File, HTTPException, UploadFile
from rasterio.io import MemoryFile
from PIL import Image


router = APIRouter(
    prefix="/preview",
    tags=["preview"],
)


def normalize_for_display(
    data: np.ndarray,
) -> np.ndarray:
    data = np.asarray(
        data,
        dtype=np.float32,
    )

    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros(
            data.shape,
            dtype=np.uint8,
        )

    values = data[valid]

    low = float(
        np.percentile(
            values,
            2,
        )
    )

    high = float(
        np.percentile(
            values,
            98,
        )
    )

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros(
            data.shape,
            dtype=np.uint8,
        )

    scaled = (
        data - low
    ) / (
        high - low
    )

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    return (
        scaled * 255.0
    ).astype(
        np.uint8
    )


def render_preview(
    raw_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:

    try:

        with MemoryFile(
            raw_bytes
        ) as memory_file:

            with memory_file.open() as src:

                if src.count < 1:
                    raise ValueError(
                        "The TIFF contains no raster bands."
                    )

                width = src.width
                height = src.height
                band_count = src.count

                # -------------------------------------------------
                # Optical / multispectral
                #
                # Our 9-channel optical test TIFF is:
                #   1 = B04
                #   2 = B03
                #   3 = B02
                #
                # Display it as RGB.
                # -------------------------------------------------
                if band_count >= 3:

                    red = src.read(
                        1
                    )

                    green = src.read(
                        2
                    )

                    blue = src.read(
                        3
                    )

                    rgb = np.stack(
                        [
                            normalize_for_display(red),
                            normalize_for_display(green),
                            normalize_for_display(blue),
                        ],
                        axis=-1,
                    )

                    image = Image.fromarray(
                        rgb,
                        mode="RGB",
                    )

                    display_mode = "RGB"
                    display_bands = [
                        1,
                        2,
                        3,
                    ]

                # -------------------------------------------------
                # SAR
                #
                # Our 2-channel SAR test TIFF is:
                #   1 = VH
                #   2 = VV
                #
                # Use VH as the grayscale display channel.
                # -------------------------------------------------
                else:

                    band = src.read(
                        1
                    )

                    gray = normalize_for_display(
                        band
                    )

                    image = Image.fromarray(
                        gray,
                        mode="L",
                    )

                    display_mode = "GRAYSCALE"
                    display_bands = [
                        1,
                    ]

                output = BytesIO()

                image.save(
                    output,
                    format="PNG",
                    optimize=True,
                )

                metadata = {
                    "width": width,
                    "height": height,
                    "channels": band_count,
                    "display_mode": display_mode,
                    "display_bands": display_bands,
                    "format": "PNG",
                }

                return (
                    output.getvalue(),
                    metadata,
                )

    except Exception as exc:

        raise ValueError(
            f"Unable to render TIFF preview: {exc}"
        ) from exc


@router.post("")
async def create_preview(
    image: UploadFile = File(...),
) -> dict[str, Any]:

    filename = (
        image.filename or ""
    ).lower()

    if not (
        filename.endswith(".tif")
        or filename.endswith(".tiff")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Preview currently supports "
                "TIFF/GeoTIFF files only."
            ),
        )

    try:

        raw_bytes = await image.read()

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded TIFF is empty.",
            )

        png_bytes, metadata = render_preview(
            raw_bytes
        )

        encoded = (
            base64.b64encode(
                png_bytes
            )
            .decode(
                "ascii"
            )
        )

        return {
            "status": "success",
            "filename": image.filename,
            "visual_preview": {
                "available": True,
                "mime_type": "image/png",
                "data_url": (
                    "data:image/png;base64,"
                    + encoded
                ),
                **metadata,
            },
        }

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "TIFF preview generation failed: "
                f"{exc}"
            ),
        ) from exc