import glob
import os

import torch
import torch.nn.functional as F

from torchgeo.models import (
    dofa_base_patch16_224,
    DOFABase16_Weights,
)


S1_INDICES = [0, 1]

S1_BANDS = [
    "VH",
    "VV",
]

# DOFA v1 released weights use these as Sentinel-1
# modality placeholder wavelengths.
S1_WAVELENGTHS = [
    3.75,
    3.75,
]


DATA_DIR = r"D:\SatQueryAI\data\processed\preprocessed"

STATS_PATH = (
    r"D:\SatQueryAI\data\processed\dofa_s1_stats.pt"
)


def main():
    paths = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.pt",
            )
        )
    )[:10]

    if len(paths) < 10:
        raise RuntimeError(
            f"Expected at least 10 samples, found {len(paths)}"
        )

    print("=" * 70)
    print("DOFA S1 10-SAMPLE VALIDATION")
    print("=" * 70)

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
    )

    mean = stats["mean"].float().view(
        1,
        2,
        1,
        1,
    )

    std = stats["std"].float().view(
        1,
        2,
        1,
        1,
    )

    print("\nLoading DOFA...")

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    model.eval()

    wavelengths = torch.tensor(
        S1_WAVELENGTHS,
        dtype=torch.float32,
    )

    print("DOFA loaded.\n")

    all_features = []

    with torch.inference_mode():

        for i, path in enumerate(paths, 1):

            data = torch.load(
                path,
                map_location="cpu",
            )

            image = data["image"].float()

            # Select VH and VV.
            x = image[S1_INDICES].unsqueeze(0)

            # Channel-wise normalization using statistics
            # computed from all 10,000 training representations.
            x = (x - mean) / std

            input_mean = float(x.mean())
            input_std = float(x.std())

            # Prepare the spatial size expected by DOFA.
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
            )

            # DOFA dynamic patch embedding.
            tokens, _ = model.patch_embed(
                x,
                wavelengths,
            )

            print(
                f"{i:02d}/10 "
                f"{os.path.basename(path)}"
            )

            print(
                f"    input mean/std: "
                f"{input_mean:.4f} / "
                f"{input_std:.4f}"
            )

            print(
                f"    patch embedding: "
                f"{tuple(tokens.shape)}"
            )

            # Add positional embeddings.
            tokens = (
                tokens
                + model.pos_embed[:, 1:, :]
            )

            # Add CLS token.
            cls = (
                model.cls_token
                + model.pos_embed[:, :1, :]
            )

            cls = cls.expand(
                tokens.shape[0],
                -1,
                -1,
            )

            tokens = torch.cat(
                [cls, tokens],
                dim=1,
            )

            # Transformer.
            for block in model.blocks:
                tokens = block(tokens)

            # Keep spatial tokens.
            spatial_tokens = tokens[:, 1:, :]

            all_features.append(
                spatial_tokens.cpu()
            )

            print(
                f"    feature shape: "
                f"{tuple(spatial_tokens.shape)}"
            )

            print(
                f"    feature min/max: "
                f"{float(spatial_tokens.min()):.4f} / "
                f"{float(spatial_tokens.max()):.4f}"
            )

            print(
                f"    feature mean/std: "
                f"{float(spatial_tokens.mean()):.4f} / "
                f"{float(spatial_tokens.std()):.4f}"
            )

    features = torch.cat(
        all_features,
        dim=0,
    )

    print("\n" + "=" * 70)
    print("10-SAMPLE SUMMARY")
    print("=" * 70)

    print(
        "Combined feature tensor:",
        tuple(features.shape),
    )

    print(
        "Global min:",
        float(features.min()),
    )

    print(
        "Global max:",
        float(features.max()),
    )

    print(
        "Global mean:",
        float(features.mean()),
    )

    print(
        "Global std:",
        float(features.std()),
    )

    finite = torch.isfinite(
        features
    ).all().item()

    print(
        "\nAll features finite:",
        finite,
    )

    expected_shape = (
        10,
        196,
        768,
    )

    print(
        "Expected shape:",
        expected_shape,
    )

    shape_correct = (
        tuple(features.shape)
        == expected_shape
    )

    print(
        "Shape correct:",
        shape_correct,
    )

    if not finite:
        raise RuntimeError(
            "Validation failed: "
            "NaN or Inf detected."
        )

    if not shape_correct:
        raise RuntimeError(
            "Validation failed: "
            "incorrect feature shape."
        )

    print("\n" + "=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()