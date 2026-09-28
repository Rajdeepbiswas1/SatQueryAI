class GroundingHeatmapModel(nn.Module):
    """
    C1:
        Cached DOFA S1/S2 tokens
            ↓
        visual fusion
            ↓
        14x14 spatial representation
            ↓
        learned high-resolution decoder
        14x14 -> 28x28 -> 56x56
            ↓
        query-conditioned spatial matching
            ↓
        56x56 heatmap
            ↓
        spatial geometry
            ↓
        box correction
            ↓
        final box
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.highres_grid_size = 56

        # --------------------------------------------------------------
        # VISUAL FUSION
        # S1 + S2 = 1536 dimensions
        # --------------------------------------------------------------
        self.visual_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # --------------------------------------------------------------
        # ORIGINAL 14x14 SPATIAL POSITION EMBEDDING
        # --------------------------------------------------------------
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, grid_size * grid_size, hidden_dim)
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02
        )

        # --------------------------------------------------------------
        # C1 HIGH-RESOLUTION DECODER
        #
        # 14x14 -> 28x28
        # 28x28 -> 56x56
        # --------------------------------------------------------------
        self.highres1 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.GroupNorm(32, hidden_dim)
        )

        self.highres2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.GroupNorm(32, hidden_dim)
        )

        # --------------------------------------------------------------
        # QUERY PROJECTION
        # --------------------------------------------------------------
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # --------------------------------------------------------------
        # VISUAL / QUERY MATCHING
        # --------------------------------------------------------------
        self.visual_match = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.query_match = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.match_norm = nn.LayerNorm(
            hidden_dim
        )

        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # --------------------------------------------------------------
        # SPATIAL SUMMARY
        # --------------------------------------------------------------
        self.spatial_summary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # --------------------------------------------------------------
        # GEOMETRY + CORRECTION
        # --------------------------------------------------------------
        #
        # spatial summary (256)
        # + query (256)
        # + heatmap-derived geometry (8)
        # = 520
        #
        # However, the validated baseline correction path used the
        # concatenated global visual/query representation and geometry.
        # We construct it explicitly below.
        #
        self.correction_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + 8,
                256
            ),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )

    # ------------------------------------------------------------------
    # SPATIAL GEOMETRY
    # ------------------------------------------------------------------
    def extract_spatial_geometry(
        self,
        heatmap
    ):
        """
        Convert heatmap into:
            center_x
            center_y
            std_x
            std_y
            x1
            y1
            x2
            y2
        """

        B, H, W = heatmap.shape

        device = heatmap.device
        dtype = heatmap.dtype

        xs = (
            torch.arange(
                W,
                device=device,
                dtype=dtype
            ) + 0.5
        ) / W

        ys = (
            torch.arange(
                H,
                device=device,
                dtype=dtype
            ) + 0.5
        ) / H

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij"
        )

        xx = xx.reshape(1, -1)
        yy = yy.reshape(1, -1)

        p = heatmap.reshape(B, -1)

        cx = (p * xx).sum(dim=1)
        cy = (p * yy).sum(dim=1)

        var_x = (
            p * (xx - cx.unsqueeze(1)).pow(2)
        ).sum(dim=1)

        var_y = (
            p * (yy - cy.unsqueeze(1)).pow(2)
        ).sum(dim=1)

        std_x = torch.sqrt(
            var_x.clamp_min(1e-8)
        )

        std_y = torch.sqrt(
            var_y.clamp_min(1e-8)
        )

        width = (
            std_x * (12.0 ** 0.5)
        ).clamp(
            min=0.02,
            max=1.0
        )

        height = (
            std_y * (12.0 ** 0.5)
        ).clamp(
            min=0.02,
            max=1.0
        )

        x1 = (
            cx - width / 2
        ).clamp(
            0.0,
            1.0
        )

        y1 = (
            cy - height / 2
        ).clamp(
            0.0,
            1.0
        )

        x2 = (
            cx + width / 2
        ).clamp(
            0.0,
            1.0
        )

        y2 = (
            cy + height / 2
        ).clamp(
            0.0,
            1.0
        )

        geometry = torch.stack(
            [
                cx,
                cy,
                std_x,
                std_y,
                x1,
                y1,
                x2,
                y2
            ],
            dim=1
        )

        base_box = torch.stack(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dim=1
        )

        return geometry, base_box

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(
        self,
        s1,
        s2,
        text_embedding
    ):
        """
        Inputs:
            s1:
                [B, 196, 768]

            s2:
                [B, 196, 768]

            text_embedding:
                [B, 768]
        """

        B = s1.size(0)

        # --------------------------------------------------------------
        # 1. FUSE S1 + S2
        # --------------------------------------------------------------
        visual = torch.cat(
            [s1, s2],
            dim=-1
        )

        visual = self.visual_fusion(
            visual
        )

        # --------------------------------------------------------------
        # 2. ADD 14x14 POSITION INFORMATION
        # --------------------------------------------------------------
        visual = visual + self.spatial_pos

        # --------------------------------------------------------------
        # 3. RESHAPE INTO 2D
        # --------------------------------------------------------------
        feature_map = visual.transpose(
            1,
            2
        ).reshape(
            B,
            self.hidden_dim,
            self.grid_size,
            self.grid_size
        )

        # --------------------------------------------------------------
        # 4. C1 HIGH-RESOLUTION UPSAMPLING
        # --------------------------------------------------------------
        feature_map = F.interpolate(
            feature_map,
            size=(28, 28),
            mode="bilinear",
            align_corners=False
        )

        feature_map = self.highres1(
            feature_map
        )

        feature_map = F.interpolate(
            feature_map,
            size=(56, 56),
            mode="bilinear",
            align_corners=False
        )

        feature_map = self.highres2(
            feature_map
        )

        # --------------------------------------------------------------
        # 5. BACK TO TOKENS
        # --------------------------------------------------------------
        visual_tokens = feature_map.flatten(
            2
        ).transpose(
            1,
            2
        )
        # [B, 3136, 256]

        # --------------------------------------------------------------
        # 6. QUERY
        # --------------------------------------------------------------
        query = self.query_proj(
            text_embedding
        )
        # [B, 256]

        # --------------------------------------------------------------
        # 7. QUERY-CONDITIONED MATCHING
        # --------------------------------------------------------------
        v = self.visual_match(
            visual_tokens
        )

        q = self.query_match(
            query
        ).unsqueeze(1)

        matched = self.match_norm(
            v + q
        )

        logits = self.match_head(
            matched
        ).squeeze(-1)

        # --------------------------------------------------------------
        # 8. 56x56 HEATMAP
        # --------------------------------------------------------------
        heatmap_logits = logits.reshape(
            B,
            56,
            56
        )

        heatmap = F.softmax(
            heatmap_logits.reshape(B, -1),
            dim=-1
        ).reshape(
            B,
            56,
            56
        )

        # --------------------------------------------------------------
        # 9. GLOBAL / SPATIAL VISUAL SUMMARY
        # --------------------------------------------------------------
        spatial_context = (
            visual_tokens * heatmap.reshape(
                B,
                -1,
                1
            )
        ).sum(
            dim=1
        )

        spatial_context = self.spatial_summary(
            spatial_context
        )

        # --------------------------------------------------------------
        # 10. HEATMAP -> GEOMETRY
        # --------------------------------------------------------------
        geometry, base_box = (
            self.extract_spatial_geometry(
                heatmap
            )
        )

        # --------------------------------------------------------------
        # 11. LEARNED BOX CORRECTION
        # --------------------------------------------------------------
        correction_input = torch.cat(
            [
                spatial_context,
                query,
                geometry
            ],
            dim=1
        )

        correction = (
            torch.tanh(
                self.correction_head(
                    correction_input
                )
            ) * 0.2
        )

        box = base_box + correction

        # --------------------------------------------------------------
        # 12. ENSURE VALID BOX
        # --------------------------------------------------------------
        x1 = torch.minimum(
            box[:, 0],
            box[:, 2]
        )

        y1 = torch.minimum(
            box[:, 1],
            box[:, 3]
        )

        x2 = torch.maximum(
            box[:, 0],
            box[:, 2]
        )

        y2 = torch.maximum(
            box[:, 1],
            box[:, 3]
        )

        box = torch.stack(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dim=1
        ).clamp(
            0.0,
            1.0
        )

        return {
            "heatmap_logits": heatmap_logits,
            "heatmap": heatmap,
            "geometry": geometry,
            "base_box": base_box,
            "correction": correction,
            "box": box
        }
