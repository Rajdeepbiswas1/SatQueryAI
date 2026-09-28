from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Fuse Sentinel-1 and Sentinel-2 DOFA spatial features.

    Input:
        s1: [B, 196, 768]
        s2: [B, 196, 768]

    Output:
        fused: [B, 196, 256]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.s1_proj = nn.Linear(input_dim, hidden_dim)
        self.s2_proj = nn.Linear(input_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> torch.Tensor:

        if s1.ndim != 3 or s2.ndim != 3:
            raise ValueError(
                f"Expected [B,196,768], got "
                f"S1={tuple(s1.shape)}, S2={tuple(s2.shape)}"
            )

        if s1.shape != s2.shape:
            raise ValueError(
                f"S1 and S2 shapes must match, got "
                f"{tuple(s1.shape)} and {tuple(s2.shape)}"
            )

        s1 = self.s1_proj(s1.float())
        s2 = self.s2_proj(s2.float())

        x = torch.cat([s1, s2], dim=-1)

        return self.fusion(x)


class TemporalFusion(nn.Module):
    """
    Fuse T1, T2 and temporal-difference DOFA features.

    Input:
        t1:   [B, 196, 768]
        t2:   [B, 196, 768]
        diff: [B, 196, 768]

    Output:
        fused: [B, 196, 256]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.t1_proj = nn.Linear(input_dim, hidden_dim)
        self.t2_proj = nn.Linear(input_dim, hidden_dim)
        self.diff_proj = nn.Linear(input_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:

        if t1.ndim != 3 or t2.ndim != 3 or diff.ndim != 3:
            raise ValueError(
                "Expected all inputs to have shape [B,196,768]"
            )

        if not (t1.shape == t2.shape == diff.shape):
            raise ValueError(
                f"Temporal input shapes must match: "
                f"T1={tuple(t1.shape)}, "
                f"T2={tuple(t2.shape)}, "
                f"Diff={tuple(diff.shape)}"
            )

        t1 = self.t1_proj(t1.float())
        t2 = self.t2_proj(t2.float())
        diff = self.diff_proj(diff.float())

        x = torch.cat([t1, t2, diff], dim=-1)

        return self.fusion(x)