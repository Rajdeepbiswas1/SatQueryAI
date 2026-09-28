from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224


PROJECT_ROOT = Path(r"D:\SatQueryAI")

PAIR_ID = "00003"

T1_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_preprocessed"
    / f"{PAIR_ID}.pt"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_rgb_stats.pt"
)

EXPECTED_SHAPE = (1, 196, 768)

WAVELENGTHS = [
    0.665,
    0.560,
    0.490,
]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def load_rgb(path: Path) -> torch.Tensor:
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    image = data["t1"].float().contiguous()

    if tuple(image.shape) != (3, 512, 512):
        raise RuntimeError(
            f"Unexpected T1 shape in {path}: "
            f"{tuple(image.shape)}"
        )

    return image


def prepare_image(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    antialias: bool,
) -> torch.Tensor:

    image = (
        image - mean.view(3, 1, 1)
    ) / std.view(3, 1, 1)

    image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
        antialias=antialias,
    )

    return image.contiguous()


def load_cache_t1() -> torch.Tensor:

    files = sorted(
        FEATURE_ROOT.glob("shard_*.pt")
    )

    if not files:
        files = sorted(
            FEATURE_ROOT.glob("*.pt")
        )

    if not files:
        raise FileNotFoundError(
            f"No temporal feature shards found under {FEATURE_ROOT}"
        )

    for path in files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids = data.get("pair_ids")

        if pair_ids is None:
            continue

        ids = [
            str(x.item() if torch.is_tensor(x) else x)
            for x in pair_ids
        ]

        if PAIR_ID not in ids:
            continue

        index = ids.index(PAIR_ID)

        features = data.get("t1_features")

        if features is None:
            features = data.get("t1")

        if features is None:
            raise RuntimeError(
                f"{path}: no t1 feature tensor found"
            )

        result = (
            features[index]
            .float()
            .unsqueeze(0)
            .contiguous()
        )

        if tuple(result.shape) != EXPECTED_SHAPE:
            raise RuntimeError(
                f"Unexpected cached feature shape: "
                f"{tuple(result.shape)}"
            )

        return result

    raise KeyError(
        f"Temporal pair {PAIR_ID} not found"
    )


def manual_spatial_features(
    model,
    x: torch.Tensor,
    wavelengths: list[float],
) -> torch.Tensor:

    tokens, _ = model.patch_embed(
        x,
        wavelengths,
    )

    tokens = (
        tokens
        + model.pos_embed[:, 1:, :]
    )

    cls = (
        model.cls_token
        + model.pos_embed[:, :1, :]
    )

    tokens = torch.cat(
        [cls.expand(tokens.shape[0], -1, -1), tokens],
        dim=1,
    )

    for block in model.blocks:
        tokens = block(tokens)

    return tokens[:, 1:, :].contiguous()


def direct_forward_features(
    model,
    x: torch.Tensor,
) -> torch.Tensor:

    signature = inspect.signature(
        model.forward_features
    )

    params = signature.parameters

    kwargs = {}

    if "wavelengths" in params:
        kwargs["wavelengths"] = WAVELENGTHS
    elif "wave_list" in params:
        kwargs["wave_list"] = WAVELENGTHS
    else:
        raise RuntimeError(
            "Could not determine wavelength argument "
            "for DOFA.forward_features"
        )

    if "global_pool" in params:
        kwargs["global_pool"] = False

    result = model.forward_features(
        x,
        **kwargs,
    )

    if isinstance(result, tuple):
        result = result[0]

    # Some implementations may return the CLS token.
    if result.ndim == 3 and result.shape[1] == 197:
        result = result[:, 1:, :]

    if tuple(result.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            "Unexpected direct forward_features shape: "
            f"{tuple(result.shape)}"
        )

    return result.contiguous()


def compare(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:

    diff = (
        actual.float()
        - expected.float()
    ).abs()

    print()
    print(label)
    print(
        f"  max_abs={diff.max().item():.8f}"
    )
    print(
        f"  mean_abs={diff.mean().item():.8f}"
    )
    is_close = torch.allclose(
        actual.float(),
        expected.float(),
        atol=1e-3,
        rtol=1e-3,
    )

    print(
        f"  allclose={is_close}"
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("SATQUERYAI TEMPORAL DOFA PATH DIAGNOSTIC")
    print("=" * 80)

    for path, label in [
        (T1_PATH, "T1 image"),
        (STATS_PATH, "Temporal statistics"),
        (FEATURE_ROOT, "Temporal feature root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    cache = load_cache_t1()

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    mean = torch.tensor(
        stats["t1_mean"],
        dtype=torch.float32,
    )

    std = torch.tensor(
        stats["t1_std"],
        dtype=torch.float32,
    )

    raw = load_rgb(T1_PATH)

    print()
    print("Input:")
    print("  raw shape:", tuple(raw.shape))
    print("  raw mean :", raw.mean().item())
    print("  raw std  :", raw.std().item())

    x_noaa = prepare_image(
        raw,
        mean,
        std,
        antialias=False,
    )

    x_aa = prepare_image(
        raw,
        mean,
        std,
        antialias=True,
    )

    print()
    print("Prepared inputs:")
    print(
        "  no-antialias:",
        tuple(x_noaa.shape),
        x_noaa.mean().item(),
        x_noaa.std().item(),
    )
    print(
        "  antialias:",
        tuple(x_aa.shape),
        x_aa.mean().item(),
        x_aa.std().item(),
    )

    print()
    print("Loading DOFA...")

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    model.eval()

    with torch.no_grad():

        # -------------------------------------------------------------
        # Official API, eval mode
        # -------------------------------------------------------------

        direct_noaa = direct_forward_features(
            model,
            x_noaa,
        )

        direct_aa = direct_forward_features(
            model,
            x_aa,
        )

        # -------------------------------------------------------------
        # Manual extraction, eval mode
        # -------------------------------------------------------------

        manual_noaa = manual_spatial_features(
            model,
            x_noaa,
            WAVELENGTHS,
        )

        # -------------------------------------------------------------
        # Train mode
        # -------------------------------------------------------------

        model.train()

        train_1 = direct_forward_features(
            model,
            x_noaa,
        )

        train_2 = direct_forward_features(
            model,
            x_noaa,
        )

        model.eval()

    print()
    print("Model path comparison:")

    compare(
        "Official forward_features (no antialias) -> cache",
        direct_noaa,
        cache,
    )

    compare(
        "Official forward_features (antialias) -> cache",
        direct_aa,
        cache,
    )

    compare(
        "Manual extraction (no antialias) -> cache",
        manual_noaa,
        cache,
    )

    compare(
        "Official forward_features vs manual extraction",
        direct_noaa,
        manual_noaa,
    )

    compare(
        "Train mode run 1 -> cache",
        train_1,
        cache,
    )

    compare(
        "Train mode run 2 -> cache",
        train_2,
        cache,
    )

    compare(
        "Train mode run 1 vs run 2",
        train_1,
        train_2,
    )

    print()
    print("Cache statistics:")
    print(
        "  min :",
        cache.min().item(),
    )
    print(
        "  max :",
        cache.max().item(),
    )
    print(
        "  mean:",
        cache.mean().item(),
    )
    print(
        "  std :",
        cache.std().item(),
    )

    print()
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()