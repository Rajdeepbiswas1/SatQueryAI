from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224


PROJECT_ROOT = Path(r"D:\SatQueryAI")

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_raw"
    / "im1"
    / "00003.png"
)

SHARD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
    / "shard_0000.pt"
)

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_rgb_stats.pt"
)

WAVELENGTHS = [0.665, 0.560, 0.490]


def load_image(path):
    image = Image.open(path).convert("RGB").copy()
    arr = np.asarray(image)

    x = torch.from_numpy(arr).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()

    return x.unsqueeze(0)


def extract(model, x, wavelengths):
    wave = torch.tensor(
        wavelengths,
        dtype=x.dtype,
        device=x.device,
    )

    tokens, _ = model.patch_embed(x, wave)

    tokens = tokens + model.pos_embed[:, 1:, :].to(x.dtype)

    cls = (
        model.cls_token.to(x.dtype)
        + model.pos_embed[:, :1, :].to(x.dtype)
    )

    cls = cls.expand(x.shape[0], -1, -1)

    tokens = torch.cat((cls, tokens), dim=1)

    for block in model.blocks:
        tokens = block(tokens)

    return tokens[:, 1:, :]


def compare(name, actual, expected):
    a = actual.float().cpu()
    b = expected.float().cpu()

    diff = torch.abs(a - b)

    print(
        f"{name}: "
        f"max_abs={diff.max().item():.8f} "
        f"mean_abs={diff.mean().item():.8f}"
    )


def main():
    print("SATQUERYAI TEMPORAL DOFA FP16 DIAGNOSTIC")
    print("=" * 60)

    stats_data = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    mean = stats_data["t1_mean"].float().view(1, 3, 1, 1)
    std = stats_data["t1_std"].float().view(1, 3, 1, 1)

    image = load_image(IMAGE_PATH)

    image = (image - mean) / std

    image = F.interpolate(
        image,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    cache = torch.load(
        SHARD_PATH,
        map_location="cpu",
        weights_only=False,
    )

    pair_ids = [
        str(x.item() if torch.is_tensor(x) else x)
        for x in cache["pair_ids"]
    ]

    row = pair_ids.index("00003")

    expected = (
        cache["t1_features"][row]
        .float()
        .unsqueeze(0)
    )

    print("\nExpected cache:")
    print("shape:", tuple(expected.shape))
    print("dtype:", cache["t1_features"].dtype)

    # --------------------------------------------------------------
    # FP32 reference
    # --------------------------------------------------------------

    print("\nLoading FP32 DOFA")

    fp32_model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    fp32_model.eval()

    for p in fp32_model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        fp32_features = extract(
            fp32_model,
            image.float(),
            WAVELENGTHS,
        )

    compare(
        "FP32 -> cache",
        fp32_features,
        expected,
    )

    # --------------------------------------------------------------
    # FP16 model
    # --------------------------------------------------------------

    print("\nLoading FP16 DOFA")

    fp16_model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    fp16_model = fp16_model.half()
    fp16_model.eval()

    for p in fp16_model.parameters():
        p.requires_grad_(False)

    image_fp16 = image.half()

    with torch.no_grad():
        fp16_features = extract(
            fp16_model,
            image_fp16,
            WAVELENGTHS,
        )

    compare(
        "FP16 -> cache",
        fp16_features,
        expected,
    )

    compare(
        "FP32 -> FP16",
        fp32_features,
        fp16_features,
    )

    print("\nFP16 output statistics:")
    x = fp16_features.float()

    print(
        "min:",
        x.min().item(),
        "max:",
        x.max().item(),
        "mean:",
        x.mean().item(),
        "std:",
        x.std().item(),
    )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()