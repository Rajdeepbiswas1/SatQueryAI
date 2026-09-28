class C3QueryConditionedGrounding(
    nn.Module
):
    """
    C3 v1.

    Input
    -----
    s1:
        [B,196,768]

    s2:
        [B,196,768]

    text_embedding:
        [B,768]

    Architecture
    ------------
    S1 + S2
        ↓
    visual fusion
        ↓
    14×14 visual representation
        ↓
    query-conditioned semantic attention
        ↓
    semantic auxiliary supervision
        ↓
    semantic prior -> 56×56
        ↓
    C2-style high-resolution grounding
        ↓
    query-conditioned 56×56 heatmap
        ↓
    moments
        ↓
    bounding box
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        grid_size=14,
        highres_grid_size=56,
    ):

        super().__init__()

        self.input_dim = (
            input_dim
        )

        self.hidden_dim = (
            hidden_dim
        )

        self.grid_size = (
            grid_size
        )

        self.highres_grid_size = (
            highres_grid_size
        )

        # ----------------------------------------------------
        # C2 visual fusion
        # ----------------------------------------------------

        self.visual_fusion = (
            nn.Sequential(
                nn.Linear(
                    input_dim * 2,
                    hidden_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    hidden_dim
                ),
            )
        )

        # ----------------------------------------------------
        # Spatial position
        # ----------------------------------------------------

        self.spatial_pos = (
            nn.Parameter(
                torch.zeros(
                    1,
                    grid_size * grid_size,
                    hidden_dim,
                )
            )
        )

        nn.init.normal_(
            self.spatial_pos,
            std=0.02,
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.query_proj = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    hidden_dim
                ),
            )
        )

        # ====================================================
        # SEMANTIC AUXILIARY PATH
        # ====================================================
        #
        # The query itself determines the semantic attention.
        #
        # There is NO ground-truth family index fed into the
        # model. The family index is used only to select the
        # supervision target during training.
        # ====================================================

        self.semantic_visual = (
            nn.Linear(
                hidden_dim,
                hidden_dim,
            )
        )

        self.semantic_query = (
            nn.Linear(
                hidden_dim,
                hidden_dim,
            )
        )

        self.semantic_norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )

        # Converts semantic attention to a learned
        # 256-channel spatial prior.
        self.semantic_prior = (
            nn.Sequential(
                nn.Conv2d(
                    1,
                    hidden_dim,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.GroupNorm(
                    32,
                    hidden_dim,
                ),
            )
        )

        # ====================================================
        # C2 HIGH-RESOLUTION PATH
        # ====================================================

        self.highres1 = (
            nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.GroupNorm(
                    32,
                    hidden_dim,
                ),
            )
        )

        self.highres2 = (
            nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.GroupNorm(
                    32,
                    hidden_dim,
                ),
            )
        )

        # ----------------------------------------------------
        # Query-conditioned grounding matching
        # ----------------------------------------------------

        self.visual_match = (
            nn.Linear(
                hidden_dim,
                hidden_dim,
            )
        )

        self.query_match = (
            nn.Linear(
                hidden_dim,
                hidden_dim,
            )
        )

        # Semantic-prior contribution.
        self.semantic_match = (
            nn.Linear(
                hidden_dim,
                hidden_dim,
            )
        )

        self.match_norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.match_head = (
            nn.Sequential(
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
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        s1,
        s2,
        text_embedding,
    ):

        B = s1.size(0)

        # ----------------------------------------------------
        # 1. Fuse S1 + S2
        # ----------------------------------------------------

        visual = torch.cat(
            [
                s1,
                s2,
            ],
            dim=-1,
        )

        visual = (
            self.visual_fusion(
                visual
            )
        )

        # ----------------------------------------------------
        # 2. Add spatial position
        # ----------------------------------------------------

        visual = (
            visual
            +
            self.spatial_pos
        )

        # ----------------------------------------------------
        # 3. 14×14 visual feature map
        # ----------------------------------------------------

        feature14 = (
            visual
            .transpose(
                1,
                2,
            )
            .reshape(
                B,
                self.hidden_dim,
                self.grid_size,
                self.grid_size,
            )
        )

        # ----------------------------------------------------
        # 4. Query
        # ----------------------------------------------------

        query = (
            self.query_proj(
                text_embedding
            )
        )

        # ====================================================
        # 5. QUERY-CONDITIONED SEMANTIC ATTENTION
        # ====================================================

        semantic_visual = (
            self.semantic_visual(
                visual
            )
        )

        semantic_query = (
            self.semantic_query(
                query
            )
            .unsqueeze(1)
        )

        semantic_joint = (
            semantic_visual
            +
            semantic_query
        )

        semantic_joint = (
            self.semantic_norm(
                F.gelu(
                    semantic_joint
                )
            )
        )

        # ----------------------------------------------------
        # Scalar semantic scores at 14×14.
        #
        # Each query produces one spatial distribution.
        # ----------------------------------------------------

        semantic_logits = (
            semantic_joint
            .mul(
                semantic_query
                +
                0.0
            )
            .sum(
                dim=-1
            )
            /
            math.sqrt(
                self.hidden_dim
            )
        )

        semantic_logits = (
            semantic_logits
            .reshape(
                B,
                self.grid_size,
                self.grid_size,
            )
        )

        semantic_attention = (
            F.softmax(
                semantic_logits.reshape(
                    B,
                    -1,
                ),
                dim=-1,
            )
            .reshape(
                B,
                self.grid_size,
                self.grid_size,
            )
        )

        # ====================================================
        # 6. HIGH-RESOLUTION VISUAL PATH
        # ====================================================

        feature_map = F.interpolate(
            feature14,
            size=(
                28,
                28,
            ),
            mode="bilinear",
            align_corners=False,
        )

        feature_map = (
            self.highres1(
                feature_map
            )
        )

        feature_map = F.interpolate(
            feature_map,
            size=(
                self.highres_grid_size,
                self.highres_grid_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        feature_map = (
            self.highres2(
                feature_map
            )
        )

        # ----------------------------------------------------
        # 7. Semantic prior at 56×56
        # ----------------------------------------------------

        semantic_prior_map = (
            semantic_attention
            .unsqueeze(1)
        )

        semantic_prior_map = (
            F.interpolate(
                semantic_prior_map,
                size=(
                    self.highres_grid_size,
                    self.highres_grid_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
        )

        semantic_prior_features = (
            self.semantic_prior(
                semantic_prior_map
            )
        )

        # ----------------------------------------------------
        # 8. Convert visual / semantic maps to tokens
        # ----------------------------------------------------

        visual_tokens = (
            feature_map
            .flatten(2)
            .transpose(
                1,
                2,
            )
        )

        semantic_tokens = (
            semantic_prior_features
            .flatten(2)
            .transpose(
                1,
                2,
            )
        )

        # ----------------------------------------------------
        # 9. Query-conditioned grounding
        # ----------------------------------------------------

        v = (
            self.visual_match(
                visual_tokens
            )
        )

        q = (
            self.query_match(
                query
            )
            .unsqueeze(1)
        )

        s = (
            self.semantic_match(
                semantic_tokens
            )
        )

        joint = (
            v
            +
            q
            +
            s
        )

        joint = (
            self.match_norm(
                F.gelu(
                    joint
                )
            )
        )

        # ----------------------------------------------------
        # 10. Grounding heatmap
        # ----------------------------------------------------

        grounding_logits = (
            self.match_head(
                joint
            )
            .squeeze(-1)
        )

        grounding_logits = (
            grounding_logits
            .reshape(
                B,
                self.highres_grid_size,
                self.highres_grid_size,
            )
        )

        grounding_heatmap = (
            F.softmax(
                grounding_logits.reshape(
                    B,
                    -1,
                ),
                dim=-1,
            )
            .reshape(
                B,
                self.highres_grid_size,
                self.highres_grid_size,
            )
        )

        # ----------------------------------------------------
        # 11. Geometry
        # ----------------------------------------------------

        (
            cx,
            cy,
            std_x,
            std_y,
        ) = heatmap_moments(
            grounding_heatmap
        )

        predicted_box = (
            heatmap_to_box(
                grounding_heatmap
            )
        )

        return {

            "semantic_logits":
                semantic_logits,

            "semantic_attention":
                semantic_attention,

            "grounding_logits":
                grounding_logits,

            "grounding_heatmap":
                grounding_heatmap,

            "cx":
                cx,

            "cy":
                cy,

            "std_x":
                std_x,

            "std_y":
                std_y,

            "predicted_box":
                predicted_box,
        }
