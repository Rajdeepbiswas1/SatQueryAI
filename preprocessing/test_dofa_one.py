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
    device = torch.device("cpu")

    # Use one existing preprocessed BigEarthNet sample.
    tensor_path = (
        r"D:\SatQueryAI\data\processed\preprocessed"
        r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90.pt"
    )

    print("=" * 70)
    print("DOFA ONE-SAMPLE TEST")
    print("=" * 70)

    print("\nLoading sample:")
    print(tensor_path)

    sample = torch.load(tensor_path, map_location="cpu")

    print("\nLoaded object type:", type(sample))

    # Our preprocessing files may contain the tensor directly
    # or a dictionary containing the tensor.
    if isinstance(sample, torch.Tensor):
        x = sample
    elif isinstance(sample, dict):
        print("Dictionary keys:", list(sample.keys()))

        if "image" in sample:
            x = sample["image"]
        elif "tensor" in sample:
            x = sample["tensor"]
        else:
            raise KeyError(
                "Could not find image tensor. "
                "Expected key 'image' or 'tensor'."
            )
    else:
        raise TypeError(f"Unsupported object type: {type(sample)}")

    print("\nInput shape before batch dimension:", tuple(x.shape))
    print("Input dtype:", x.dtype)
    print("Input min:", float(x.min()))
    print("Input max:", float(x.max()))

    # Add batch dimension.
    x = x.unsqueeze(0).to(device)
    x = torch.nn.functional.interpolate(
    x,
    size=(224, 224),
    mode="bicubic",
    align_corners=False,
    )

    print("\nInput shape sent to DOFA:", tuple(x.shape))
    print("Number of wavelengths:", len(DOFA_WAVELENGTHS_14))

    wavelengths = torch.tensor(
        DOFA_WAVELENGTHS_14,
        dtype=torch.float32,
        device=device,
    )

    print("Wavelength tensor:", wavelengths)

    print("\nLoading pretrained DOFA...")
    weights = DOFABase16_Weights.DOFA_MAE

    model = dofa_base_patch16_224(weights=weights)
    model = model.to(device)
    model.eval()

    print("DOFA loaded successfully.")

    with torch.inference_mode():
        print("\nRunning forward_features()...")
        features = model.forward_features(x, wavelengths)

        print("forward_features output type:", type(features))
        print("forward_features shape:", tuple(features.shape))

        print("\nRunning full forward()...")
        output = model(x, DOFA_WAVELENGTHS_14)

        print("forward output shape:", tuple(output.shape))

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()