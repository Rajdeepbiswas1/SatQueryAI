import os

import torch
import torch.nn.functional as F
import rasterio
import numpy as np

from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights


FOLDER = (
    r"D:\SatQueryAI\data\processed\images_raw"
    r"\BigEarthNet-S2\BigEarthNet-S2"
    r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
    r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90"
)

BANDS = [
    "B04", "B03", "B02",
    "B05", "B06", "B07",
    "B08", "B11", "B12",
]

WAVELENGTHS = [
    0.665, 0.560, 0.490,
    0.705, 0.740, 0.783,
    0.842, 1.610, 2.190,
]

MEAN = torch.tensor([
    114.1099739,
    114.81779093,
    126.63977424,
    84.33539309,
    97.84789168,
    103.94461911,
    101.435633,
    72.32804172,
    56.66528851,
], dtype=torch.float32)

STD = torch.tensor([
    77.84352553,
    69.96844919,
    67.42465279,
    64.57022983,
    61.72545487,
    61.34187099,
    60.29744676,
    47.88519516,
    42.55886798,
], dtype=torch.float32)


def find_band(band):
    for name in os.listdir(FOLDER):
        if name.endswith(f"_{band}.tif"):
            return os.path.join(FOLDER, name)
    raise FileNotFoundError(f"{band} TIFF not found")


def main():
    print("=" * 70)
    print("DOFA RAW S2 INFERENCE TEST")
    print("=" * 70)

    channels = []

    for band in BANDS:
        path = find_band(band)

        with rasterio.open(path) as src:
            image = src.read(1).astype("float32")

        print(
            f"{band}: "
            f"original_shape={image.shape}, "
            f"min={image.min():.2f}, "
            f"max={image.max():.2f}, "
            f"mean={image.mean():.2f}"
        )

        # Convert all selected bands to the common 120x120 grid.
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        if image.shape != (120, 120):
            image_tensor = F.interpolate(
                image_tensor,
                size=(120, 120),
                mode="bilinear",
                align_corners=False,
            )

        image_tensor = image_tensor.squeeze(0).squeeze(0)

        channels.append(image_tensor)

    x = torch.stack(channels)

    print("\nRaw tensor:")
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)

    # Spatial preparation.
    # The released DOFA checkpoint requires a 224x224 input.
    x = F.interpolate(
        x.unsqueeze(0),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    # DOFA normalization on the RAW channel scale.
    x = (
        x
        - MEAN.view(1, -1, 1, 1)
    ) / STD.view(1, -1, 1, 1)

    print("\nFinal DOFA input:")
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", float(x.min()))
    print("max:", float(x.max()))
    print("mean:", float(x.mean()))
    print("std:", float(x.std()))

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )
    model.eval()

    wavelengths = torch.tensor(
        WAVELENGTHS,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        tokens, _ = model.patch_embed(x, wavelengths)

        print("\nPatch embedding:", tuple(tokens.shape))

        tokens = tokens + model.pos_embed[:, 1:, :]

        cls = model.cls_token + model.pos_embed[:, :1, :]
        cls = cls.expand(tokens.shape[0], -1, -1)

        tokens = torch.cat([cls, tokens], dim=1)

        for block in model.blocks:
            tokens = block(tokens)

        spatial_tokens = tokens[:, 1:, :]
        cls_token = tokens[:, 0, :]

        print("Transformer output:", tuple(tokens.shape))
        print("Spatial tokens:", tuple(spatial_tokens.shape))
        print("CLS token:", tuple(cls_token.shape))

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()