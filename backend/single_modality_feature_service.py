from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agent.dofa_extractor import (
    S1_WAVELENGTHS,
    S2_WAVELENGTHS,
    SatQueryAIDofaExtractor,
)


class SingleModalityFeatureExtractionError(
    RuntimeError
):
    """Raised when single-modality DOFA extraction fails."""


class SingleModalityFeatureService:
    """
    Raw single-modality image -> DOFA spatial features.

    This service is intentionally separate from the existing
    DofaFeatureService, which is a frozen 14-channel paired-input
    service.

    Supported single-modality tensors:

        S1: [2, H, W] or [1, 2, H, W]
        S2: [9, H, W] or [1, 9, H, W]

    Output:

        [1, 196, 768]

    The same frozen pretrained DOFA model and the same normalization
    statistics used by the existing project extractor are reused.
    """

    EXPECTED_FEATURE_SHAPE = (
        1,
        196,
        768,
    )

    MODALITY_CHANNELS = {
        "s1": 2,
        "s2": 9,
    }

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = torch.device(device)

        try:
            self.extractor = SatQueryAIDofaExtractor(
                device=device
            )
        except Exception as exc:
            raise SingleModalityFeatureExtractionError(
                f"Failed to initialize pretrained DOFA: {exc}"
            ) from exc

        self.model = self.extractor.model

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self.model_name = (
            "DOFA_MAE_SingleModality"
        )

    def _validate_image(
        self,
        image: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:

        modality = modality.strip().lower()

        if modality not in self.MODALITY_CHANNELS:
            raise SingleModalityFeatureExtractionError(
                "Unsupported modality. Expected 's1' or 's2', "
                f"got '{modality}'."
            )

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise SingleModalityFeatureExtractionError(
                "Input image must be a torch.Tensor."
            )

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise SingleModalityFeatureExtractionError(
                "Expected image shape "
                "[C,H,W] or [1,C,H,W], got "
                f"{tuple(image.shape)}"
            )

        expected_channels = (
            self.MODALITY_CHANNELS[modality]
        )

        if image.shape[0] != 1:
            raise SingleModalityFeatureExtractionError(
                "Single-modality inference currently supports "
                f"batch size 1, got {image.shape[0]}."
            )

        if image.shape[1] != expected_channels:
            raise SingleModalityFeatureExtractionError(
                f"{modality.upper()} input requires "
                f"{expected_channels} bands, got "
                f"{image.shape[1]}."
            )

        image = image.float().to(
            self.device
        )

        if not torch.isfinite(
            image
        ).all().item():
            raise SingleModalityFeatureExtractionError(
                "Input image contains NaN or Inf."
            )

        if (
            image.shape[2] < 1
            or image.shape[3] < 1
        ):
            raise SingleModalityFeatureExtractionError(
                "Input image has invalid spatial dimensions."
            )

        return image

    def _normalization_parameters(
        self,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

        if modality == "s1":
            mean = self.extractor.s1_mean
            std = self.extractor.s1_std
            wavelengths = list(
                S1_WAVELENGTHS
            )
        else:
            mean = self.extractor.s2_mean
            std = self.extractor.s2_std
            wavelengths = list(
                S2_WAVELENGTHS
            )

        return (
            mean,
            std,
            wavelengths,
        )

    @torch.inference_mode()
    def extract(
        self,
        image: torch.Tensor,
        modality: str,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
    ]:

        modality = modality.strip().lower()

        image = self._validate_image(
            image,
            modality,
        )

        (
            mean,
            std,
            wavelengths,
        ) = self._normalization_parameters(
            modality
        )

        mean = mean.view(
            1,
            -1,
            1,
            1,
        )

        std = std.view(
            1,
            -1,
            1,
            1,
        )

        x = (
            image - mean
        ) / std

        if not torch.isfinite(
            x
        ).all().item():
            raise SingleModalityFeatureExtractionError(
                "Normalized image contains NaN or Inf."
            )

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )

        wavelength_tensor = torch.tensor(
            wavelengths,
            dtype=x.dtype,
            device=x.device,
        )

        try:

            tokens, _ = (
                self.model.patch_embed(
                    x,
                    wavelength_tensor,
                )
            )

            tokens = (
                tokens
                + self.model.pos_embed[
                    :,
                    1:,
                    :,
                ]
            )

            cls = (
                self.model.cls_token
                + self.model.pos_embed[
                    :,
                    :1,
                    :,
                ]
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

            for block in self.model.blocks:
                tokens = block(tokens)

            spatial_tokens = (
                tokens[:, 1:, :]
                .float()
                .contiguous()
            )

        except Exception as exc:

            raise SingleModalityFeatureExtractionError(
                f"DOFA single-modality extraction failed: {exc}"
            ) from exc

        if tuple(
            spatial_tokens.shape
        ) != self.EXPECTED_FEATURE_SHAPE:

            raise SingleModalityFeatureExtractionError(
                "Unexpected single-modality DOFA feature shape: "
                f"{tuple(spatial_tokens.shape)}; expected "
                f"{self.EXPECTED_FEATURE_SHAPE}"
            )

        if not torch.isfinite(
            spatial_tokens
        ).all().item():
            raise SingleModalityFeatureExtractionError(
                "DOFA features contain NaN or Inf."
            )

        metadata = {
            "status": "success",
            "model": self.model_name,
            "modality": modality,
            "device": str(self.device),
            "input_shape": list(
                image.shape
            ),
            "resized_shape": [
                image.shape[0],
                image.shape[1],
                224,
                224,
            ],
            "feature_shape": list(
                spatial_tokens.shape
            ),
            "feature_dtype": str(
                spatial_tokens.dtype
            ),
            "wavelengths": wavelengths,
            "parameter_count": self.parameter_count,
            "frozen": True,
            "eval_mode": not self.model.training,
        }

        return (
            spatial_tokens.cpu(),
            metadata,
        )

    def extract_s1(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
    ]:
        return self.extract(
            image,
            "s1",
        )

    def extract_s2(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
    ]:
        return self.extract(
            image,
            "s2",
        )