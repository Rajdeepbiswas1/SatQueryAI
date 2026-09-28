from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224


PROJECT_ROOT = Path(r"D:\SatQueryAI")

T1_PATH = (
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


def load_raw(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)

    if arr.shape != (512, 512, 3):
        raise RuntimeError(f"Unexpected image shape: {arr.shape}")

    x = torch.from_numpy(arr).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()

    return x


def spatial_extract(model, x, wavelengths):
    wave = torch.tensor(
        wavelengths,
        dtype=x.dtype,
        device=x.device,
    )

    tokens, _ = model.patch_embed(x, wave)

    tokens = tokens + model.pos_embed[:, 1:, :]

    cls = model.cls_token + model.pos_embed[:, :1, :]
    cls = cls.expand(x.shape[0], -1, -1)

    tokens = torch.cat((cls, tokens), dim=1)

    for block in model.blocks:
        tokens = block(tokens)

    # Keep spatial tokens only.
    return tokens[:, 1:, :]


def stats(name, x):
    y = x.detach().float().cpu()

    print(
        f"{name}: "
        f"shape={tuple(y.shape)} "
        f"min={y.min().item():.6f} "
        f"max={y.max().item():.6f} "
        f"mean={y.mean().item():.6f} "
        f"std={y.std().item():.6f}"
    )


def compare(name, actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()

    diff = torch.abs(actual - expected)

    print(
        f"{name}: "
        f"max_abs={diff.max().item():.8f} "
        f"mean_abs={diff.mean().item():.8f}"
    )


def main():
    print("SATQUERYAI TEMPORAL DOFA VARIANT DIAGNOSTIC")
    print("=" * 60)

    raw = load_raw(T1_PATH)

    stats_data = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    mean = stats_data["t1_mean"].float().view(1, 3, 1, 1)
    std = stats_data["t1_std"].float().view(1, 3, 1, 1)

    cached = torch.load(
        SHARD_PATH,
        map_location="cpu",
        weights_only=False,
    )

    pair_ids = [
        str(x.item() if torch.is_tensor(x) else x)
        for x in cached["pair_ids"]
    ]

    row = pair_ids.index("00003")

    expected = cached["t1_features"][row].float().unsqueeze(0)

    print("\nAuthoritative cached feature:")
    stats("cached T1", expected)

    # ------------------------------------------------------------------
    # Build candidate inputs.
    # ------------------------------------------------------------------

    raw_b = raw.unsqueeze(0)

    # Variant A:
    # normalize first, then resize.
    a = (raw_b - mean) / std
    a = F.interpolate(
        a,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    # Variant B:
    # resize first, then normalize.
    b = F.interpolate(
        raw_b,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )
    b = (b - mean) / std

    # Variant C:
    # resize only, no dataset normalization.
    c = F.interpolate(
        raw_b,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    print("\nCandidate input statistics:")
    stats("A normalize -> resize", a)
    stats("B resize -> normalize", b)
    stats("C resize only", c)

    # ------------------------------------------------------------------
    # Load DOFA once.
    # ------------------------------------------------------------------

    print("\nLoading DOFA...")

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # One batched forward pass for all variants.
    batch = torch.cat([a, b, c], dim=0)

    with torch.no_grad():
        features = spatial_extract(
            model,
            batch,
            WAVELENGTHS,
        )

    fa = features[0:1]
    fb = features[1:2]
    fc = features[2:3]

    print("\nDOFA outputs:")
    stats("A features", fa)
    stats("B features", fb)
    stats("C features", fc)

    print("\nComparison against authoritative cache:")

    compare("A normalize -> resize", fa, expected)
    compare("B resize -> normalize", fb, expected)
    compare("C resize only", fc, expected)

    print("\nDirect variant comparison:")

    compare("A vs B", fa, fb)
    compare("A vs C", fa, fc)
    compare("B vs C", fb, fc)

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()