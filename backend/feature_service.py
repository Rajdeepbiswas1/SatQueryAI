from typing import Any

import torch

from agent.dofa_extractor import SatQueryAIDofaExtractor


class FeatureExtractionError(Exception):
    """Raised when DOFA feature extraction fails."""


class DofaFeatureService:
    """
    Application-level wrapper around the frozen SatQueryAI DOFA
    feature extractor.

    Input:
        [14,120,120] or [1,14,120,120]

    Output:
        S1: [1,196,768]
        S2: [1,196,768]
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = device
        self.extractor = SatQueryAIDofaExtractor(
            device=device
        )

    def extract(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:

        if not isinstance(image, torch.Tensor):
            raise FeatureExtractionError(
                "Input must be a torch.Tensor."
            )

        if tuple(image.shape) not in {
            (14, 120, 120),
            (1, 14, 120, 120),
        }:
            raise FeatureExtractionError(
                "Expected input shape "
                "[14,120,120] or [1,14,120,120], got "
                f"{tuple(image.shape)}"
            )

        if image.dtype != torch.float32:
            raise FeatureExtractionError(
                "Expected torch.float32 input, got "
                f"{image.dtype}"
            )

        if not torch.isfinite(image).all().item():
            raise FeatureExtractionError(
                "Input contains NaN or Inf."
            )

        try:
            s1_features, s2_features = (
                self.extractor.extract(image)
            )

        except Exception as exc:
            raise FeatureExtractionError(
                f"DOFA feature extraction failed: {exc}"
            ) from exc

        expected = (
            1,
            196,
            768,
        )

        if tuple(s1_features.shape) != expected:
            raise FeatureExtractionError(
                "Unexpected S1 feature shape: "
                f"{tuple(s1_features.shape)}"
            )

        if tuple(s2_features.shape) != expected:
            raise FeatureExtractionError(
                "Unexpected S2 feature shape: "
                f"{tuple(s2_features.shape)}"
            )

        metadata = {
            "device": str(self.device),
            "input_shape": list(image.shape),
            "s1_shape": list(s1_features.shape),
            "s2_shape": list(s2_features.shape),
            "feature_dtype": str(s1_features.dtype),
        }

        return (
            s1_features,
            s2_features,
            metadata,
        )


if __name__ == "__main__":
    from preprocessing.preprocess_dataset import (
        build_s1_paths,
        build_s2_paths,
    )

    from backend.upload_service import (
        preprocess_upload_bundle,
    )

    import pandas as pd

    print(
        "SATQUERYAI — DOFA FEATURE SERVICE TEST"
    )
    print("=" * 80)
    print()

    selected = pd.read_csv(
        r"data\splits\selected_pairs.csv"
    )

    metadata = pd.read_parquet(
        r"data\raw\BigEarthNet_LMDB\metadata.parquet",
        columns=[
            "patch_id",
            "s1_name",
        ],
    )

    row = (
        selected
        .merge(
            metadata,
            on="patch_id",
            how="inner",
        )
        .iloc[0]
    )

    patch_id = row.patch_id
    s1_name = row.s1_name

    s1_paths = build_s1_paths(
        s1_name
    )

    s2_paths = build_s2_paths(
        patch_id
    )

    all_paths = (
        list(s1_paths.values())
        + list(s2_paths.values())
    )

    image, preprocessing_metadata, _ = (
        preprocess_upload_bundle(
            all_paths,
            patch_id=patch_id,
        )
    )

    print(
        f"patch_id: {patch_id}"
    )

    print(
        f"input shape: "
        f"{preprocessing_metadata['shape']}"
    )

    service = DofaFeatureService(
        device="cpu"
    )

    print()
    print(
        "Running frozen DOFA extraction..."
    )

    s1, s2, feature_metadata = (
        service.extract(image)
    )

    print()
    print(
        "FEATURE OUTPUT"
    )
    print(
        f"  S1 shape: {feature_metadata['s1_shape']}"
    )
    print(
        f"  S2 shape: {feature_metadata['s2_shape']}"
    )
    print(
        f"  dtype: {feature_metadata['feature_dtype']}"
    )
    print(
        f"  device: {feature_metadata['device']}"
    )

    if tuple(s1.shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "S1 feature validation failed."
        )

    if tuple(s2.shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "S2 feature validation failed."
        )

    if not torch.isfinite(s1).all().item():
        raise RuntimeError(
            "S1 features contain NaN or Inf."
        )

    if not torch.isfinite(s2).all().item():
        raise RuntimeError(
            "S2 features contain NaN or Inf."
        )

    print()
    print(
        "DOFA FEATURE SERVICE TEST: PASS"
    )