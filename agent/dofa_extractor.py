from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from torchgeo.models import (
    dofa_base_patch16_224,
    DOFABase16_Weights,
)


PROJECT_ROOT = Path(r"D:\SatQueryAI")

S1_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_s1_stats.pt"
)

S2_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_s2_stats.pt"
)


# Original 14-channel tensor order:
#
# 0  VH
# 1  VV
# 2  B01
# 3  B02
# 4  B03
# 5  B04
# 6  B05
# 7  B06
# 8  B07
# 9  B08
# 10 B09
# 11 B11
# 12 B12
# 13 B8A

S1_INDICES = [0, 1]

S2_INDICES = [
    5,   # B04
    4,   # B03
    3,   # B02
    6,   # B05
    7,   # B06
    8,   # B07
    9,   # B08
    11,  # B11
    12,  # B12
]

S1_WAVELENGTHS = [
    3.75,
    3.75,
]

S2_WAVELENGTHS = [
    0.665,
    0.560,
    0.490,
    0.705,
    0.740,
    0.783,
    0.842,
    1.610,
    2.190,
]


class SatQueryAIDofaExtractor:
    """
    Frozen DOFA inference-only feature extractor.

    Input:
        [1, 14, 120, 120]

    Output:
        S1: [1, 196, 768]
        S2: [1, 196, 768]

    DOFA weights are never trained or modified.
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = torch.device(device)

        if not S1_STATS_PATH.is_file():
            raise FileNotFoundError(
                f"Missing S1 statistics: {S1_STATS_PATH}"
            )

        if not S2_STATS_PATH.is_file():
            raise FileNotFoundError(
                f"Missing S2 statistics: {S2_STATS_PATH}"
            )

        # ---------------------------------------------------------
        # Load the exact statistics used by the existing DOFA
        # feature-generation pipeline.
        # ---------------------------------------------------------

        s1_stats = torch.load(
            S1_STATS_PATH,
            map_location="cpu",
        )

        s2_stats = torch.load(
            S2_STATS_PATH,
            map_location="cpu",
        )

        self.s1_mean = (
            s1_stats["mean"]
            .float()
            .to(self.device)
        )

        self.s1_std = (
            s1_stats["std"]
            .float()
            .to(self.device)
        )

        self.s2_mean = (
            s2_stats["mean"]
            .float()
            .to(self.device)
        )

        self.s2_std = (
            s2_stats["std"]
            .float()
            .to(self.device)
        )

        # ---------------------------------------------------------
        # Load pretrained DOFA.
        #
        # This loads the pretrained weights but does NOT train them.
        # ---------------------------------------------------------

        self.model = (
            dofa_base_patch16_224(
                weights=DOFABase16_Weights.DOFA_MAE
            )
            .to(self.device)
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    # -----------------------------------------------------------------
    # Internal feature extraction
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def _extract(
        self,
        images: torch.Tensor,
        indices,
        mean: torch.Tensor,
        std: torch.Tensor,
        wavelengths,
    ) -> torch.Tensor:

        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape "
                "[B,14,120,120], got "
                f"{tuple(images.shape)}"
            )

        if images.shape[1:] != (
            14,
            120,
            120,
        ):
            raise ValueError(
                "Expected input shape "
                "[B,14,120,120], got "
                f"{tuple(images.shape)}"
            )

        x = images[:, indices, :, :]

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

        x = (x - mean) / std

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

        tokens, _ = self.model.patch_embed(
            x,
            wavelength_tensor,
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
            [cls, tokens],
            dim=1,
        )

        for block in self.model.blocks:
            tokens = block(tokens)

        spatial_tokens = tokens[:, 1:, :]

        return spatial_tokens.float()

    # -----------------------------------------------------------------
    # Public extraction method
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def extract(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        image = image.float().to(
            self.device
        )

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape != (
            1,
            14,
            120,
            120,
        ):
            raise ValueError(
                "Expected one image with shape "
                "[14,120,120] or [1,14,120,120], "
                f"got {tuple(image.shape)}"
            )

        if not torch.isfinite(image).all().item():
            raise ValueError(
                "Input image contains NaN or Inf."
            )

        s1_features = self._extract(
            image,
            S1_INDICES,
            self.s1_mean,
            self.s1_std,
            S1_WAVELENGTHS,
        )

        s2_features = self._extract(
            image,
            S2_INDICES,
            self.s2_mean,
            self.s2_std,
            S2_WAVELENGTHS,
        )

        expected = (
            1,
            196,
            768,
        )

        if tuple(s1_features.shape) != expected:
            raise RuntimeError(
                "Unexpected S1 feature shape: "
                f"{tuple(s1_features.shape)}"
            )

        if tuple(s2_features.shape) != expected:
            raise RuntimeError(
                "Unexpected S2 feature shape: "
                f"{tuple(s2_features.shape)}"
            )

        if not torch.isfinite(
            s1_features
        ).all().item():
            raise RuntimeError(
                "S1 features contain NaN or Inf."
            )

        if not torch.isfinite(
            s2_features
        ).all().item():
            raise RuntimeError(
                "S2 features contain NaN or Inf."
            )

        return (
            s1_features,
            s2_features,
        )