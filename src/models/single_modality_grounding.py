import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def boxes_to_heatmaps(
    boxes,
    height=56,
    width=56,
):
    """
    Convert normalized [x1,y1,x2,y2] boxes into
    normalized area-overlap heatmaps.

    Each cell receives the area of intersection
    between the cell and the target box.
    """

    device = boxes.device
    dtype = boxes.dtype

    batch_size = boxes.shape[0]

    x1 = boxes[:, 0].view(-1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1)

    ys = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / height

    xs = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / width

    cy = ys.view(1, height, 1)
    cx = xs.view(1, 1, width)

    cell_x1 = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        )
        / width
    ).view(1, 1, width)

    cell_x2 = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 1
    ).view(1, 1, width) / width

    cell_y1 = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        )
        / height
    ).view(1, height, 1)

    cell_y2 = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 1
    ).view(1, height, 1) / height

    overlap_w = (
        torch.minimum(x2, cell_x2)
        - torch.maximum(x1, cell_x1)
    ).clamp_min(0.0)

    overlap_h = (
        torch.minimum(y2, cell_y2)
        - torch.maximum(y1, cell_y1)
    ).clamp_min(0.0)

    heatmap = overlap_w * overlap_h

    heatmap = heatmap.reshape(
        batch_size,
        height,
        width
    )

    heatmap_sum = (
        heatmap.sum(dim=(1, 2), keepdim=True)
        .clamp_min(1e-8)
    )

    return heatmap / heatmap_sum

def heatmap_moments(
    heatmap,
):
    """
    Differentiable centroid and spread.

    Returns:
        cx, cy, std_x, std_y
    """

    batch_size, height, width = heatmap.shape

    device = heatmap.device
    dtype = heatmap.dtype

    ys = (
        torch.arange(
            height,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / height

    xs = (
        torch.arange(
            width,
            device=device,
            dtype=dtype
        ) + 0.5
    ) / width

    yy = ys.view(1, height, 1)
    xx = xs.view(1, 1, width)

    mass = heatmap.sum(
        dim=(1, 2)
    ).clamp_min(1e-8)

    cx = (
        heatmap * xx
    ).sum(
        dim=(1, 2)
    ) / mass

    cy = (
        heatmap * yy
    ).sum(
        dim=(1, 2)
    ) / mass

    variance_x = (
        heatmap
        * (xx - cx.view(-1, 1, 1)) ** 2
    ).sum(
        dim=(1, 2)
    ) / mass

    variance_y = (
        heatmap
        * (yy - cy.view(-1, 1, 1)) ** 2
    ).sum(
        dim=(1, 2)
    ) / mass

    std_x = torch.sqrt(
        variance_x + 1e-8
    )

    std_y = torch.sqrt(
        variance_y + 1e-8
    )

    return cx, cy, std_x, std_y

def heatmap_to_box(
    heatmap,
):
    """
    Convert heatmap moments into a normalized
    bounding box.

    For a roughly uniform rectangle:
        std = width / sqrt(12)
    """

    cx, cy, std_x, std_y = heatmap_moments(
        heatmap
    )

    width = (
        std_x * math.sqrt(12.0)
    ).clamp(
        min=0.02,
        max=1.0
    )

    height = (
        std_y * math.sqrt(12.0)
    ).clamp(
        min=0.02,
        max=1.0
    )

    x1 = (
        cx - width / 2.0
    ).clamp(0.0, 1.0)

    y1 = (
        cy - height / 2.0
    ).clamp(0.0, 1.0)

    x2 = (
        cx + width / 2.0
    ).clamp(0.0, 1.0)

    y2 = (
        cy + height / 2.0
    ).clamp(0.0, 1.0)

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1
    )

class SingleModalityGrounding(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
        highres_grid=56,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.highres_grid = highres_grid
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Visual projection
        # ----------------------------------------------------

        self.visual_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Learnable spatial encoding
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.randn(
                1,
                grid_size * grid_size,
                hidden_dim
            ) * 0.02
        )

        # ----------------------------------------------------
        # Query-conditioned token representation
        # ----------------------------------------------------

        self.match = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Coarse 14x14 localization score
        # ----------------------------------------------------

        self.coarse_head = nn.Linear(
            hidden_dim,
            1
        )

        # ----------------------------------------------------
        # High-resolution decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            ),
        )

        # ----------------------------------------------------
        # Global query-conditioned refinement
        # ----------------------------------------------------

        self.refinement = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        visual,
        text,
    ):

        if visual.ndim != 3:
            raise ValueError(
                f"Expected visual [B,196,768], "
                f"got {tuple(visual.shape)}"
            )

        if text.ndim != 2:
            raise ValueError(
                f"Expected text [B,768], "
                f"got {tuple(text.shape)}"
            )

        b = visual.shape[0]

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        v = self.visual_proj(
            visual.float()
        )

        q = self.query_proj(
            text.float()
        )

        v = v + self.spatial_pos

        # ----------------------------------------------------
        # Question-conditioned matching
        # ----------------------------------------------------

        q_expanded = q.unsqueeze(1).expand(
            -1,
            v.shape[1],
            -1
        )

        combined = torch.cat(
            [v, q_expanded],
            dim=-1
        )

        conditioned = self.match(
            combined
        )

        # ----------------------------------------------------
        # 14x14 coarse feature map
        # ----------------------------------------------------

        coarse_logits = self.coarse_head(
            conditioned
        ).squeeze(-1)

        coarse_map = conditioned.transpose(
            1,
            2
        ).reshape(
            b,
            self.hidden_dim,
            self.grid_size,
            self.grid_size
        )

        # ----------------------------------------------------
        # High-resolution heatmap
        # ----------------------------------------------------

        heatmap_logits = self.decoder(
            coarse_map
        ).squeeze(1)

        # ----------------------------------------------------
        # Query-conditioned refinement
        # ----------------------------------------------------

        global_visual = conditioned.mean(
            dim=1
        )

        refinement = self.refinement(
            torch.cat(
                [global_visual, q],
                dim=-1
            )
        )

        heatmap_logits = (
            heatmap_logits
            + refinement.mean(
                dim=-1
            ).view(
                b, 1, 1
            )
        )

        heatmap = F.softmax(
            heatmap_logits.flatten(1),
            dim=1
        ).reshape(
            b,
            self.highres_grid,
            self.highres_grid
        )

        predicted_box = heatmap_to_box(
            heatmap
        )

        cx, cy, std_x, std_y = heatmap_moments(
            heatmap
        )

        return {
            "coarse_logits": coarse_logits,
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "cx": cx,
            "cy": cy,
            "std_x": std_x,
            "std_y": std_y,
            "predicted_box": predicted_box,
        }
