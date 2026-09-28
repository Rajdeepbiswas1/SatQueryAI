"""
SatQueryAI - Cross-Attention Reference Grounding Model

Purpose
-------
Query-conditioned visual grounding for remote-sensing reference expressions.

Input:
    visual_tokens: [B, 196, 768]
        Frozen DOFA spatial tokens from a 14x14 grid.

    text_embedding: [B, 768]
        Frozen DistilBERT embedding of the reference expression.

Output:
    heatmap_logits: [B, 14, 14]
    heatmap:        [B, 14, 14]
    box:            [B, 4]   -> normalized [x1, y1, x2, y2]

Architecture
------------
DOFA tokens
    -> visual projection
    -> query-conditioned cross-attention
    -> spatial positional embedding
    -> token refinement
    -> heatmap
    -> differentiable geometry
    -> bounding box

The model does not modify DOFA weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Small feed-forward network used inside the grounding model."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialPositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for the 14x14 DOFA spatial grid.
    """

    def __init__(
        self,
        height: int = 14,
        width: int = 14,
        dim: int = 256,
    ) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.dim = dim

        self.embedding = nn.Parameter(
            torch.randn(1, height * width, dim) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.expand(batch_size, -1, -1)


class GroundingCrossAttentionModel(nn.Module):
    """
    Cross-attention reference grounding model.

    Inputs
    ------
    visual_tokens:
        [B, 196, 768]

    text_embedding:
        [B, 768]

    Outputs
    -------
    dict containing:
        heatmap_logits : [B, 14, 14]
        heatmap        : [B, 14, 14]
        box            : [B, 4]
        geometry       : [B, 4]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        grid_size: int = 14,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # ------------------------------------------------------------
        # 1. Project frozen DOFA visual tokens
        # ------------------------------------------------------------

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------
        # 2. Project frozen text embedding
        # ------------------------------------------------------------

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------
        # 3. Spatial positional information
        # ------------------------------------------------------------

        self.spatial_position = SpatialPositionalEmbedding(
            height=grid_size,
            width=grid_size,
            dim=hidden_dim,
        )

        # ------------------------------------------------------------
        # 4. Query -> visual cross-attention
        #
        # Query:
        #     [B, 1, D]
        #
        # Key/Value:
        #     [B, 196, D]
        #
        # This lets the textual reference actively select visual
        # tokens.
        # ------------------------------------------------------------

        self.query_to_visual = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------
        # 5. Token refinement after cross-attention
        # ------------------------------------------------------------

        self.token_mlp = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.token_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------
        # 6. Additional text-to-token compatibility branch
        #
        # This provides an explicit similarity signal between every
        # visual token and the reference expression.
        # ------------------------------------------------------------

        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------
        # 7. Heatmap prediction
        # ------------------------------------------------------------

        self.heatmap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ------------------------------------------------------------
        # 8. Geometry head
        #
        # Receives differentiable heatmap geometry:
        #     center_x
        #     center_y
        #     std_x
        #     std_y
        #
        # plus global visual summary and query representation.
        # ------------------------------------------------------------

        geometry_input_dim = (
            hidden_dim   # visual summary
            + hidden_dim # text
            + 4          # heatmap geometry
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(geometry_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        # ------------------------------------------------------------
        # Coordinate grids are buffers, not trainable parameters.
        # ------------------------------------------------------------

        coords = torch.linspace(0.0, 1.0, grid_size)

        yy, xx = torch.meshgrid(
            coords,
            coords,
            indexing="ij",
        )

        self.register_buffer(
            "grid_x",
            xx.reshape(-1),
            persistent=False,
        )

        self.register_buffer(
            "grid_y",
            yy.reshape(-1),
            persistent=False,
        )

    # ==================================================================
    # Heatmap geometry
    # ==================================================================

    def heatmap_geometry(
        self,
        heatmap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a spatial probability map into differentiable
        geometric statistics.

        Parameters
        ----------
        heatmap:
            [B, H, W]

        Returns
        -------
        geometry:
            [B, 4]

        Order:
            center_x,
            center_y,
            std_x,
            std_y
        """

        batch_size = heatmap.shape[0]

        weights = heatmap.reshape(
            batch_size,
            self.num_tokens,
        )

        weights = weights / (
            weights.sum(dim=1, keepdim=True) + 1e-8
        )

        x = self.grid_x.to(
            dtype=heatmap.dtype,
            device=heatmap.device,
        ).unsqueeze(0)

        y = self.grid_y.to(
            dtype=heatmap.dtype,
            device=heatmap.device,
        ).unsqueeze(0)

        center_x = (weights * x).sum(dim=1)
        center_y = (weights * y).sum(dim=1)

        var_x = (
            weights * (x - center_x.unsqueeze(1)).pow(2)
        ).sum(dim=1)

        var_y = (
            weights * (y - center_y.unsqueeze(1)).pow(2)
        ).sum(dim=1)

        std_x = torch.sqrt(var_x + 1e-6)
        std_y = torch.sqrt(var_y + 1e-6)

        return torch.stack(
            [
                center_x,
                center_y,
                std_x,
                std_y,
            ],
            dim=1,
        )

    # ==================================================================
    # Geometry -> box
    # ==================================================================

    def geometry_to_box(
        self,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert:
            center_x, center_y, std_x, std_y

        into normalized:
            x1, y1, x2, y2

        The standard deviation is converted into a spatial extent.
        """

        cx = geometry[:, 0]
        cy = geometry[:, 1]
        sx = geometry[:, 2]
        sy = geometry[:, 3]

        # 2 standard deviations gives the main heatmap mass.
        half_width = 2.0 * sx
        half_height = 2.0 * sy

        x1 = cx - half_width
        y1 = cy - half_height
        x2 = cx + half_width
        y2 = cy + half_height

        x1 = x1.clamp(0.0, 1.0)
        y1 = y1.clamp(0.0, 1.0)
        x2 = x2.clamp(0.0, 1.0)
        y2 = y2.clamp(0.0, 1.0)

        # Explicitly enforce valid ordering.
        x_low = torch.minimum(x1, x2)
        x_high = torch.maximum(x1, x2)

        y_low = torch.minimum(y1, y2)
        y_high = torch.maximum(y1, y2)

        return torch.stack(
            [
                x_low,
                y_low,
                x_high,
                y_high,
            ],
            dim=1,
        )

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # ------------------------------------------------------------
        # Shape checks
        # ------------------------------------------------------------

        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [B, 196, 768]. "
                f"Received {tuple(visual_tokens.shape)}."
            )

        if text_embedding.ndim != 2:
            raise ValueError(
                "text_embedding must have shape [B, 768]. "
                f"Received {tuple(text_embedding.shape)}."
            )

        batch_size, num_tokens, _ = visual_tokens.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} visual tokens, "
                f"received {num_tokens}."
            )

        # ------------------------------------------------------------
        # Visual projection
        # ------------------------------------------------------------

        visual = self.visual_projection(
            visual_tokens
        )

        # Add spatial information.
        visual = visual + self.spatial_position(batch_size)

        # ------------------------------------------------------------
        # Text projection
        # ------------------------------------------------------------

        query = self.text_projection(
            text_embedding
        ).unsqueeze(1)  # [B, 1, D]

        # ------------------------------------------------------------
        # Query-conditioned cross-attention
        # ------------------------------------------------------------

        attended_query, attention_weights = self.query_to_visual(
            query=query,
            key=visual,
            value=visual,
            need_weights=True,
        )

        # attended_query:
        # [B, 1, D]

        attended_query = self.cross_attn_norm(
            query + attended_query
        )

        query_vector = attended_query.squeeze(1)

        # ------------------------------------------------------------
        # Broadcast query-conditioned information back to every
        # spatial token.
        # ------------------------------------------------------------

        query_expanded = query_vector.unsqueeze(1).expand(
            -1,
            self.num_tokens,
            -1,
        )

        combined = torch.cat(
            [
                visual,
                query_expanded,
            ],
            dim=-1,
        )

        gate = self.query_gate(combined)

        conditioned_tokens = (
            visual * gate
            + query_expanded * (1.0 - gate)
        )

        # ------------------------------------------------------------
        # Residual token refinement
        # ------------------------------------------------------------

        refined = self.token_mlp(
            conditioned_tokens
        )

        conditioned_tokens = self.token_norm(
            conditioned_tokens + refined
        )

        # ------------------------------------------------------------
        # Heatmap logits
        # ------------------------------------------------------------

        heatmap_logits = self.heatmap_head(
            conditioned_tokens
        ).squeeze(-1)

        heatmap_logits = heatmap_logits.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # ------------------------------------------------------------
        # Spatial softmax
        # ------------------------------------------------------------

        heatmap = F.softmax(
            heatmap_logits.reshape(
                batch_size,
                self.num_tokens,
            ),
            dim=1,
        ).reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # ------------------------------------------------------------
        # Differentiable geometry
        # ------------------------------------------------------------

        geometry = self.heatmap_geometry(
            heatmap
        )

        # ------------------------------------------------------------
        # Global visual summary
        # ------------------------------------------------------------

        visual_summary = conditioned_tokens.mean(
            dim=1
        )

        geometry_input = torch.cat(
            [
                visual_summary,
                query_vector,
                geometry,
            ],
            dim=1,
        )

        box_delta = self.geometry_head(
            geometry_input
        )

        # Keep geometry head output bounded.
        #
        # This produces a learned adjustment around the
        # heatmap-derived geometry rather than an unrestricted
        # absolute coordinate prediction.

        box_delta = 0.20 * torch.tanh(
            box_delta
        )

        base_box = self.geometry_to_box(
            geometry
        )

        box = base_box + box_delta

        # ------------------------------------------------------------
        # Normalize box ordering and bounds
        # ------------------------------------------------------------

        x1 = box[:, 0].clamp(0.0, 1.0)
        y1 = box[:, 1].clamp(0.0, 1.0)
        x2 = box[:, 2].clamp(0.0, 1.0)
        y2 = box[:, 3].clamp(0.0, 1.0)

        x_low = torch.minimum(x1, x2)
        x_high = torch.maximum(x1, x2)

        y_low = torch.minimum(y1, y2)
        y_high = torch.maximum(y1, y2)

        box = torch.stack(
            [
                x_low,
                y_low,
                x_high,
                y_high,
            ],
            dim=1,
        )

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "geometry": geometry,
            "base_box": base_box,
            "box": box,
            "cross_attention": attention_weights,
        }


# ======================================================================
# Simple smoke test
# ======================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("GROUNDING CROSS-ATTENTION MODEL SMOKE TEST")
    print("=" * 70)

    model = GroundingCrossAttentionModel().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {total_params:,}")

    visual = torch.randn(
        8,
        196,
        768,
        device=device,
    )

    text = torch.randn(
        8,
        768,
        device=device,
    )

    with torch.no_grad():
        output = model(
            visual_tokens=visual,
            text_embedding=text,
        )

    for key, value in output.items():
        print(
            f"{key:20s}: "
            f"{tuple(value.shape)}"
        )

    boxes = output["box"]

    assert boxes.shape == (8, 4)
    assert output["heatmap"].shape == (8, 14, 14)
    assert output["geometry"].shape == (8, 4)

    assert torch.isfinite(boxes).all()
    assert torch.isfinite(output["heatmap"]).all()

    assert (boxes >= 0.0).all()
    assert (boxes <= 1.0).all()

    assert (boxes[:, 0] <= boxes[:, 2]).all()
    assert (boxes[:, 1] <= boxes[:, 3]).all()

    heatmap_sums = output["heatmap"].sum(dim=(1, 2))

    assert torch.allclose(
        heatmap_sums,
        torch.ones_like(heatmap_sums),
        atol=1e-5,
    )

    print()
    print("All smoke-test checks passed.")
    print("=" * 70)