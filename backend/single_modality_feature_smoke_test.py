from pathlib import Path

import torch

from backend.single_modality_feature_service import (
    SingleModalityFeatureService,
)


DATA_ROOT = Path(
    r"D:\SatQueryAI\data\processed\preprocessed"
)

S1_INDICES = [0, 1]

S2_INDICES = [
    5,
    4,
    3,
    6,
    7,
    8,
    9,
    11,
    12,
]


def find_14_channel_tensor(value):
    """
    Recursively find the first tensor with one of the expected
    14-channel spatial shapes.
    """

    if isinstance(value, torch.Tensor):

        if tuple(value.shape) in {
            (14, 120, 120),
            (1, 14, 120, 120),
        }:
            return value

    if isinstance(value, dict):

        for item in value.values():

            found = find_14_channel_tensor(item)

            if found is not None:
                return found

    if isinstance(value, (list, tuple)):

        for item in value:

            found = find_14_channel_tensor(item)

            if found is not None:
                return found

    return None


def main() -> None:

    print("===================================================")
    print("SATQUERYAI SINGLE-MODALITY FEATURE SMOKE TEST")
    print("===================================================")

    files = sorted(
        DATA_ROOT.rglob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No .pt files found under {DATA_ROOT}"
        )

    print("Candidate .pt files:", len(files))

    selected_file = None
    image = None

    for path in files:

        try:

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            candidate = find_14_channel_tensor(
                payload
            )

            if candidate is not None:

                selected_file = path
                image = candidate
                break

        except Exception:
            continue

    if selected_file is None:
        raise RuntimeError(
            "Could not find a real 14-channel tensor "
            "under the preprocessed dataset."
        )

    print("Selected sample:")
    print(selected_file)

    if image.ndim == 4:
        image = image[0]

    if tuple(image.shape) != (
        14,
        120,
        120,
    ):
        raise RuntimeError(
            f"Unexpected source shape: {tuple(image.shape)}"
        )

    image = image.float()

    if not torch.isfinite(image).all().item():
        raise RuntimeError(
            "Source image contains NaN or Inf."
        )

    print(
        "Source shape:",
        list(image.shape),
    )

    s1_image = image[
        S1_INDICES,
        :, :
    ].unsqueeze(0)

    s2_image = image[
        S2_INDICES,
        :, :
    ].unsqueeze(0)

    print(
        "S1 input shape:",
        list(s1_image.shape),
    )

    print(
        "S2 input shape:",
        list(s2_image.shape),
    )

    service = SingleModalityFeatureService(
        device="cpu"
    )

    print(
        "DOFA parameter count:",
        service.parameter_count,
    )

    s1_features, s1_meta = service.extract_s1(
        s1_image
    )

    s2_features, s2_meta = service.extract_s2(
        s2_image
    )

    print(
        "S1 feature shape:",
        list(s1_features.shape),
    )

    print(
        "S2 feature shape:",
        list(s2_features.shape),
    )

    print(
        "S1 feature dtype:",
        s1_features.dtype,
    )

    print(
        "S2 feature dtype:",
        s2_features.dtype,
    )

    assert tuple(
        s1_features.shape
    ) == (
        1,
        196,
        768,
    )

    assert tuple(
        s2_features.shape
    ) == (
        1,
        196,
        768,
    )

    assert torch.isfinite(
        s1_features
    ).all().item()

    assert torch.isfinite(
        s2_features
    ).all().item()

    assert s1_meta["modality"] == "s1"
    assert s2_meta["modality"] == "s2"

    assert s1_meta["frozen"] is True
    assert s2_meta["frozen"] is True

    assert s1_meta["eval_mode"] is True
    assert s2_meta["eval_mode"] is True

    print("S1 extraction: PASS")
    print("S2 extraction: PASS")

    print(
        "S1 wavelengths:",
        s1_meta["wavelengths"],
    )

    print(
        "S2 wavelengths:",
        s2_meta["wavelengths"],
    )

    print(
        "SINGLE-MODALITY FEATURE SERVICE: PASS"
    )


if __name__ == "__main__":
    main()