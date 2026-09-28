class PointGroundingModel(
    nn.Module
):

    """
    Point-conditioned remote-sensing grounding model.

    Inputs:
        S1 DOFA:    [B,196,768]
        S2 DOFA:    [B,196,768]
        text:       [B,768]
        point:      [B,2]

    Design:

        S1 + S2
            ↓
        visual fusion
            ↓
        14x14 spatial tokens
            ↓
        point-aware local feature
            +
        query embedding
            +
        point embedding
            +
        global visual feature
            ↓
        box-distance regression

    The box is parameterized relative to the supplied point:

        left distance
        top distance
        right distance
        bottom distance

    This guarantees that the predicted box contains
    the supplied point.
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        # ----------------------------------------------------
        # S1 + S2 fusion
        # ----------------------------------------------------

        self.visual_fusion = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Spatial position
        # ----------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.zeros(
                1,
                grid_size * grid_size,
                hidden_dim,
            )
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02,
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = nn.Sequential(
            nn.Linear(
                768,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Point projection
        # ----------------------------------------------------

        self.point_proj = nn.Sequential(
            nn.Linear(
                2,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        # ----------------------------------------------------
        # Spatial relevance head
        # ----------------------------------------------------

        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.condition_match = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.match_norm = nn.LayerNorm(
            hidden_dim
        )

        self.heatmap_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1,
            ),
        )

        # ----------------------------------------------------
        # Box regression
        # ----------------------------------------------------

        self.box_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 4,
                512,
            ),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.Linear(
                256,
                4,
            ),
        )

    def forward(
        self,
        s1,
        s2,
        text,
        point,
    ):

        B = s1.size(0)

        # ----------------------------------------------------
        # 1. Visual fusion
        # ----------------------------------------------------

        visual = torch.cat(
            [s1, s2],
            dim=-1,
        )

        visual = self.visual_fusion(
            visual
        )

        # ----------------------------------------------------
        # 2. Spatial position
        # ----------------------------------------------------

        visual = (
            visual +
            self.spatial_pos
        )

        # ----------------------------------------------------
        # 3. Feature map
        # ----------------------------------------------------

        fmap = (
            visual
            .transpose(1, 2)
            .reshape(
                B,
                self.hidden_dim,
                self.grid_size,
                self.grid_size,
            )
        )

        # ----------------------------------------------------
        # 4. Query + point condition
        # ----------------------------------------------------

        query = self.query_proj(
            text
        )

        point_feature = self.point_proj(
            point
        )

        condition = (
            query +
            point_feature
        )

        # ----------------------------------------------------
        # 5. Local feature sampled AT POINT
        # ----------------------------------------------------

        grid = point * 2.0 - 1.0

        grid = grid.view(
            B,
            1,
            1,
            2,
        )

        local_feature = F.grid_sample(
            fmap,
            grid,
            mode="bilinear",
            align_corners=False,
        )

        local_feature = (
            local_feature
            .squeeze(-1)
            .squeeze(-1)
        )

        # ----------------------------------------------------
        # 6. Global visual feature
        # ----------------------------------------------------

        global_visual = (
            visual.mean(dim=1)
        )

        # ----------------------------------------------------
        # 7. Spatial relevance
        # ----------------------------------------------------

        v = self.visual_match(
            visual
        )

        c = self.condition_match(
            condition
        ).unsqueeze(1)

        matched = self.match_norm(
            v + c
        )

        heatmap_logits = (
            self.heatmap_head(
                matched
            )
            .squeeze(-1)
            .reshape(
                B,
                self.grid_size,
                self.grid_size,
            )
        )

        heatmap = F.softmax(
            heatmap_logits.reshape(
                B,
                -1,
            ),
            dim=-1,
        ).reshape(
            B,
            self.grid_size,
            self.grid_size,
        )

        # ----------------------------------------------------
        # 8. Heatmap context
        # ----------------------------------------------------

        context = (
            visual *
            heatmap.reshape(
                B,
                -1,
                1,
            )
        ).sum(
            dim=1
        )

        # ----------------------------------------------------
        # 9. Box distances from point
        # ----------------------------------------------------

        box_input = torch.cat(
            [
                local_feature,
                context,
                global_visual,
                condition,
            ],
            dim=-1,
        )

        distances = F.softplus(
            self.box_head(
                box_input
            )
        )

        # Prevent excessively large early predictions.
        distances = distances.clamp(
            min=1e-4,
            max=1.0,
        )

        left = distances[:, 0]
        top = distances[:, 1]
        right = distances[:, 2]
        bottom = distances[:, 3]

        px = point[:, 0]
        py = point[:, 1]

        left = torch.minimum(
            left,
            px,
        )

        top = torch.minimum(
            top,
            py,
        )

        right = torch.minimum(
            right,
            1.0 - px,
        )

        bottom = torch.minimum(
            bottom,
            1.0 - py,
        )

        pred_box = torch.stack(
            [
                px - left,
                py - top,
                px + right,
                py + bottom,
            ],
            dim=-1,
        )

        return {
            "heatmap_logits":
                heatmap_logits,

            "heatmap":
                heatmap,

            "pred_box":
                pred_box,

            "distances":
                distances,

            "local_feature":
                local_feature,

            "global_visual":
                global_visual,
        }
