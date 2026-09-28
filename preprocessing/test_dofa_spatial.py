import torch
from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights


DOFA_WAVELENGTHS_14 = [
    3.75,   # VH
    3.75,   # VV
    0.443,  # B01
    0.490,  # B02
    0.560,  # B03
    0.665,  # B04
    0.705,  # B05
    0.740,  # B06
    0.783,  # B07
    0.842,  # B08
    0.945,  # B09
    1.610,  # B11
    2.190,  # B12
    0.865,  # B8A
]


def main():
    path = (
        r"D:\SatQueryAI\data\processed\preprocessed"
        r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90.pt"
    )

    sample = torch.load(path, map_location="cpu")
    x = sample["image"].unsqueeze(0)

    # Temporary compatibility preparation for the test.
    x = torch.nn.functional.interpolate(
        x,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    wavelengths = torch.tensor(
        DOFA_WAVELENGTHS_14,
        dtype=x.dtype,
    )

    print("=" * 70)
    print("DOFA SPATIAL TOKEN TEST")
    print("=" * 70)

    model = dofa_base_patch16_224(
        weights=DOFABase16_Weights.DOFA_MAE
    )
    model.eval()

    with torch.inference_mode():
        # Same sequence used internally by DOFA.forward_features()
        wavelist = torch.tensor(
            wavelengths,
            device=x.device,
            dtype=x.dtype,
        )

        tokens, _ = model.patch_embed(x, wavelist)

        print("\nAfter patch embedding:", tuple(tokens.shape))

        tokens = tokens + model.pos_embed[:, 1:, :]

        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)

        tokens = torch.cat((cls_tokens, tokens), dim=1)

        print("After adding CLS:", tuple(tokens.shape))

        for block in model.blocks:
            tokens = block(tokens)

        print("After transformer blocks:", tuple(tokens.shape))

        spatial_tokens = tokens[:, 1:, :]
        cls_token = tokens[:, 0, :]

        print("\nSpatial tokens:", tuple(spatial_tokens.shape))
        print("CLS token:", tuple(cls_token.shape))

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()