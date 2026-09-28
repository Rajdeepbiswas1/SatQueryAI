import torch
import torch.nn.functional as F

from torchgeo.models import (
    dofa_base_patch16_224,
    DOFABase16_Weights,
)


PAIR_PATH = (
    r"D:\SatQueryAI\data\processed\temporal_preprocessed\00003.pt"
)

STATS_PATH = (
    r"D:\SatQueryAI\data\processed\temporal_rgb_stats.pt"
)


# RGB wavelengths in micrometres.
RGB_WAVELENGTHS = [
    0.665,  # Red
    0.560,  # Green
    0.490,  # Blue
]


def extract_spatial_features(model, x, wavelengths):
    wavelengths = torch.tensor(
        wavelengths,
        dtype=x.dtype,
        device=x.device,
    )

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

    cls = cls.expand(
        tokens.shape[0],
        -1,
        -1,
    )

    tokens = torch.cat(
        [cls, tokens],
        dim=1,
    )

    for block in model.blocks:
        tokens = block(tokens)

    return tokens[:, 1:, :], tokens[:, 0, :]


def prepare_image(
    image,
    mean,
    std,
):
    # image: [3,512,512]
    x = image.unsqueeze(0).float()

    mean = mean.view(
        1, 3, 1, 1
    )

    std = std.view(
        1, 3, 1, 1
    )

    x = (x - mean) / std

    x = F.interpolate(
        x,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    return x


def main():
    print("=" * 70)
    print("DOFA TEMPORAL ONE-PAIR TEST")
    print("=" * 70)

    data = torch.load(
        PAIR_PATH,
        map_location="cpu",
    )

    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
    )

    print("\nPair:", data["pair_id"])

    t1 = data["t1"].float()
    t2 = data["t2"].float()

    print("T1 shape:", tuple(t1.shape))
    print("T2 shape:", tuple(t2.shape))

    t1_mean = stats["t1_mean"].float()
    t1_std = stats["t1_std"].float()

    t2_mean = stats["t2_mean"].float()
    t2_std = stats["t2_std"].float()

    # --------------------------------------------------------
    # Prepare T1
    # --------------------------------------------------------

    x1 = prepare_image(
        t1,
        t1_mean,
        t1_std,
    )

    # --------------------------------------------------------
    # Prepare T2
    # --------------------------------------------------------

    x2 = prepare_image(
        t2,
        t2_mean,
        t2_std,
    )

    print("\nPrepared T1:")
    print("shape:", tuple(x1.shape))
    print("mean:", float(x1.mean()))
    print("std :", float(x1.std()))
    print("min :", float(x1.min()))
    print("max :", float(x1.max()))

    print("\nPrepared T2:")
    print("shape:", tuple(x2.shape))
    print("mean:", float(x2.mean()))
    print("std :", float(x2.std()))
    print("min :", float(x2.min()))
    print("max :", float(x2.max()))

    # --------------------------------------------------------
    # Load DOFA
    # --------------------------------------------------------

    print("\nLoading DOFA...")

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )

    model.eval()

    print("DOFA loaded.")

    # --------------------------------------------------------
    # Extract T1 and T2
    # --------------------------------------------------------

    with torch.inference_mode():

        f1, cls1 = extract_spatial_features(
            model,
            x1,
            RGB_WAVELENGTHS,
        )

        f2, cls2 = extract_spatial_features(
            model,
            x2,
            RGB_WAVELENGTHS,
        )

    print("\nT1 features:")
    print("shape:", tuple(f1.shape))
    print("min :", float(f1.min()))
    print("max :", float(f1.max()))
    print("mean:", float(f1.mean()))
    print("std :", float(f1.std()))

    print("\nT2 features:")
    print("shape:", tuple(f2.shape))
    print("min :", float(f2.min()))
    print("max :", float(f2.max()))
    print("mean:", float(f2.mean()))
    print("std :", float(f2.std()))

    print("\nCLS features:")
    print("T1:", tuple(cls1.shape))
    print("T2:", tuple(cls2.shape))

    # --------------------------------------------------------
    # Simple temporal differences
    # --------------------------------------------------------

    absolute_difference = torch.abs(
        f2 - f1
    )

    signed_difference = f2 - f1

    print("\nTemporal representation:")
    print("|F2-F1| shape:", tuple(absolute_difference.shape))
    print(
        "|F2-F1| mean:",
        float(absolute_difference.mean()),
    )
    print(
        "|F2-F1| std :",
        float(absolute_difference.std()),
    )

    print("\nF2-F1 shape:", tuple(signed_difference.shape))

    print(
        "All finite:",
        (
            torch.isfinite(f1).all()
            and torch.isfinite(f2).all()
            and torch.isfinite(absolute_difference).all()
        ).item(),
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()