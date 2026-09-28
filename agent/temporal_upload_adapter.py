from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchgeo.models import (
    DOFABase16_Weights,
    dofa_base_patch16_224,
)


PROJECT_ROOT = Path(r"D:\SatQueryAI")

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_rgb_stats.pt"
)

RGB_WAVELENGTHS = [
    0.665,  # Red
    0.560,  # Green
    0.490,  # Blue
]

EXPECTED_IMAGE_SHAPE = (
    3,
    512,
    512,
)

EXPECTED_FEATURE_SHAPE = (
    1,
    196,
    768,
)


class TemporalUploadAdapter:
    """
    Convert two raw SECOND-style temporal RGB images into the
    exact feature representation required by Temporal Change V3.

    Runtime contract:

        t1            [1,196,768]
        t2            [1,196,768]
        diff          [1,196,768]

    Processing follows the verified temporal preprocessing path:
        RGB image
            -> float32 [3,512,512]
            -> T1/T2-specific normalization
            -> bicubic resize to [224,224]
            -> DOFA spatial token extraction
            -> [1,196,768]

        diff = abs(t2 - t1)
    """

    def __init__(
        self,
        stats_path: Path = STATS_PATH,
        device: str | torch.device = "cpu",
    ) -> None:
        self.stats_path = Path(stats_path)
        self.device = torch.device(device)

        self._validate_paths()

        stats = torch.load(
            self.stats_path,
            map_location="cpu",
            weights_only=False,
        )

        required_stats = {
            "t1_mean",
            "t1_std",
            "t2_mean",
            "t2_std",
        }

        missing = required_stats - set(stats.keys())

        if missing:
            raise RuntimeError(
                "Temporal RGB statistics are missing required keys: "
                f"{sorted(missing)}"
            )

        self.t1_mean = self._validate_stats(
            stats["t1_mean"],
            "t1_mean",
        )

        self.t1_std = self._validate_stats(
            stats["t1_std"],
            "t1_std",
        )

        self.t2_mean = self._validate_stats(
            stats["t2_mean"],
            "t2_mean",
        )

        self.t2_std = self._validate_stats(
            stats["t2_std"],
            "t2_std",
        )

        self.model = (
            dofa_base_patch16_224(
                weights=DOFABase16_Weights.DOFA_MAE
            )
            .to(self.device)
            .eval()
        )

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self.metadata = {
            "backbone": "DOFA",
            "weights": "DOFA_MAE",
            "device": str(self.device),
            "rgb_wavelengths": list(RGB_WAVELENGTHS),
            "input_shape": list(EXPECTED_IMAGE_SHAPE),
            "resize_shape": [3, 224, 224],
            "output_feature_shape": list(
                EXPECTED_FEATURE_SHAPE
            ),
            "difference": "abs(t2_features - t1_features)",
            "parameter_count": parameter_count,
            "checkpoint_frozen": True,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        if not self.stats_path.is_file():
            raise FileNotFoundError(
                "Temporal RGB statistics file not found: "
                f"{self.stats_path}"
            )

    @staticmethod
    def _validate_stats(
        value: Any,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
        )

        if tuple(tensor.shape) != (3,):
            raise RuntimeError(
                f"{name} must have shape (3,), "
                f"got {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

        return tensor.contiguous()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgb_image(
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(
                f"Temporal image not found: {path}"
            )

        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                array = np.asarray(rgb)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read temporal image: {path}"
            ) from exc

        if tuple(array.shape) != (
            512,
            512,
            3,
        ):
            raise ValueError(
                f"Temporal image must be 512x512 RGB: "
                f"{path}; got {tuple(array.shape)}"
            )

        if array.dtype != np.uint8:
            array = array.astype(
                np.uint8,
                copy=False,
            )

        tensor = torch.from_numpy(
            array.copy()
        ).float()

        tensor = tensor / 255.0

        tensor = tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        if tuple(tensor.shape) != EXPECTED_IMAGE_SHAPE:
            raise RuntimeError(
                "Unexpected RGB tensor shape after loading "
                f"{path}: {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite values found in image: {path}"
            )

        return tensor

    # ------------------------------------------------------------------
    # Exact verified temporal preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(
        image: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(image.shape) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                "Expected image tensor shape "
                f"{EXPECTED_IMAGE_SHAPE}, "
                f"got {tuple(image.shape)}"
            )

        x = image.unsqueeze(0).float()

        mean = mean.view(
            1,
            3,
            1,
            1,
        )

        std = std.view(
            1,
            3,
            1,
            1,
        )

        if torch.any(std <= 0):
            raise RuntimeError(
                "Temporal RGB standard deviation "
                "contains non-positive values."
            )

        x = (x - mean) / std

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )

        if tuple(x.shape) != (
            1,
            3,
            224,
            224,
        ):
            raise RuntimeError(
                "Unexpected prepared temporal image shape: "
                f"{tuple(x.shape)}"
            )

        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Prepared temporal image contains "
                "non-finite values."
            )

        return x

    # ------------------------------------------------------------------
    # Exact DOFA spatial feature extraction
    # ------------------------------------------------------------------

    def _extract_spatial_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(x.shape) != (
            1,
            3,
            224,
            224,
        ):
            raise ValueError(
                "DOFA temporal input must have shape "
                f"(1,3,224,224), got {tuple(x.shape)}"
            )

        x = x.to(self.device)

        wavelengths = torch.tensor(
            RGB_WAVELENGTHS,
            dtype=x.dtype,
            device=x.device,
        )

        tokens, _ = self.model.patch_embed(
            x,
            wavelengths,
        )

        if tuple(tokens.shape) != (
            1,
            196,
            768,
        ):
            raise RuntimeError(
                "Unexpected DOFA patch embedding shape: "
                f"{tuple(tokens.shape)}"
            )

        tokens = (
            tokens
            + self.model.pos_embed[:, 1:, :]
        )

        cls = (
            self.model.cls_token
            + self.model.pos_embed[:, :1, :]
        )

        cls = cls.expand(
            tokens.shape[0],
            -1,
            -1,
        )

        tokens = torch.cat(
            [
                cls,
                tokens,
            ],
            dim=1,
        )

        if tuple(tokens.shape) != (
            1,
            197,
            768,
        ):
            raise RuntimeError(
                "Unexpected DOFA token sequence shape: "
                f"{tuple(tokens.shape)}"
            )

        for block in self.model.blocks:
            tokens = block(tokens)

        spatial_tokens = tokens[:, 1:, :]

        if tuple(spatial_tokens.shape) != (
            1,
            196,
            768,
        ):
            raise RuntimeError(
                "Unexpected DOFA spatial feature shape: "
                f"{tuple(spatial_tokens.shape)}"
            )

        spatial_tokens = (
            spatial_tokens
            .float()
            .contiguous()
        )

        if not torch.isfinite(spatial_tokens).all():
            raise RuntimeError(
                "DOFA spatial features contain "
                "non-finite values."
            )

        return spatial_tokens

    # ------------------------------------------------------------------
    # Public pair transformation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transform_pair(
        self,
        t1_path: str | Path,
        t2_path: str | Path,
    ) -> dict[str, Any]:
        t1_path = Path(t1_path)
        t2_path = Path(t2_path)

        if t1_path.resolve() == t2_path.resolve():
            raise ValueError(
                "T1 and T2 must be different image files."
            )

        t1_image = self._load_rgb_image(
            t1_path
        )

        t2_image = self._load_rgb_image(
            t2_path
        )

        t1_prepared = self._prepare_image(
            t1_image,
            self.t1_mean,
            self.t1_std,
        )

        t2_prepared = self._prepare_image(
            t2_image,
            self.t2_mean,
            self.t2_std,
        )

        t1_features = self._extract_spatial_features(
            t1_prepared
        )

        t2_features = self._extract_spatial_features(
            t2_prepared
        )

        diff_features = torch.abs(
            t2_features - t1_features
        )

        if tuple(t1_features.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                "Final T1 feature shape mismatch: "
                f"{tuple(t1_features.shape)}"
            )

        if tuple(t2_features.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                "Final T2 feature shape mismatch: "
                f"{tuple(t2_features.shape)}"
            )

        if tuple(diff_features.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                "Final difference feature shape mismatch: "
                f"{tuple(diff_features.shape)}"
            )

        if not torch.isfinite(diff_features).all():
            raise RuntimeError(
                "Temporal difference features contain "
                "non-finite values."
            )

        return {
            "t1": t1_features.float().cpu().contiguous(),
            "t2": t2_features.float().cpu().contiguous(),
            "diff": diff_features.float().cpu().contiguous(),
            "metadata": {
                **self.metadata,
                "t1_path": str(t1_path),
                "t2_path": str(t2_path),
                "t1_input_shape": list(t1_image.shape),
                "t2_input_shape": list(t2_image.shape),
            },
        }


# ----------------------------------------------------------------------
# Direct smoke construction
# ----------------------------------------------------------------------

if __name__ == "__main__":
    adapter = TemporalUploadAdapter()

    print("=" * 70)
    print("SATQUERYAI — TEMPORAL UPLOAD ADAPTER")
    print("=" * 70)
    print("Adapter initialization: PASS")
    print("Device:", adapter.device)
    print("DOFA parameter count:", adapter.metadata["parameter_count"])
    print(
        "Checkpoint frozen:",
        adapter.metadata["checkpoint_frozen"],
    )