import torch
import torch.nn.functional as F


S2_INDICES = [5, 4, 3, 6, 7, 8, 9, 11, 12]

S2_BANDS = [
    "B04", "B03", "B02",
    "B05", "B06", "B07",
    "B08", "B11", "B12",
]

S2_WAVELENGTHS = [
    0.665, 0.560, 0.490,
    0.705, 0.740, 0.783,
    0.842, 1.610, 2.190,
]

S2_MEAN = torch.tensor([
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

S2_STD = torch.tensor([
    77.84352553,
    69.96844919,
    67.42465279,
    64.57022983,
    61.72545487,
    61.34187099,
    60.29744676,
    47.88519516,
    42.55886798,
])


def main():
    path = (
        r"D:\SatQueryAI\data\processed\preprocessed"
        r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90.pt"
    )

    data = torch.load(path, map_location="cpu")
    image = data["image"]

    print("=" * 70)
    print("DOFA S2 ADAPTER TEST")
    print("=" * 70)

    print("\nOriginal shape:", tuple(image.shape))
    print("Original channels:", data["channel_names"])

    # Select the nine Sentinel-2 bands used by DOFA.
    s2 = image[S2_INDICES]

    print("\nSelected bands:")
    for i, band in enumerate(S2_BANDS):
        print(
            f"{i}: {band:>3} | "
            f"input min={s2[i].min():.4f} "
            f"max={s2[i].max():.4f} "
            f"mean={s2[i].mean():.4f}"
        )

    # Our existing tensor is [0,1], so temporarily convert it back
    # using the dataset-specific P05/P95 values before applying the
    # DOFA statistics.
    #
    # IMPORTANT:
    # This is only a numerical test. We are NOT declaring this
    # reconstruction the final DOFA preprocessing yet.
    p05 = torch.tensor([
        52,   # B04
        101,  # B03
        89,   # B02
        43,   # B05
        30,   # B06
        32,   # B07
        30,   # B08
        17,   # B11
        17,   # B12
    ], dtype=torch.float32)

    p95 = torch.tensor([
        1678, # B04
        1360, # B03
        1056, # B02
        2000, # B05
        3423, # B06
        4095, # B07
        4380, # B08
        3382, # B11
        2629, # B12
    ], dtype=torch.float32)

    reconstructed = s2 * (p95 - p05).view(-1, 1, 1) + p05.view(-1, 1, 1)

    dofa_input = (
        reconstructed - S2_MEAN.view(-1, 1, 1)
    ) / S2_STD.view(-1, 1, 1)

    print("\nAfter DOFA normalization:")
    print("shape:", tuple(dofa_input.shape))
    print("min:", float(dofa_input.min()))
    print("max:", float(dofa_input.max()))
    print("mean:", float(dofa_input.mean()))
    print("std:", float(dofa_input.std()))

    # DOFA requires 224x224 spatial input.
    dofa_input = F.interpolate(
        dofa_input.unsqueeze(0),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    print("\nFinal DOFA input:")
    print("shape:", tuple(dofa_input.shape))
    print("dtype:", dofa_input.dtype)
    print("min:", float(dofa_input.min()))
    print("max:", float(dofa_input.max()))
    print("mean:", float(dofa_input.mean()))
    print("std:", float(dofa_input.std()))

    print("\nWavelengths:", S2_WAVELENGTHS)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()