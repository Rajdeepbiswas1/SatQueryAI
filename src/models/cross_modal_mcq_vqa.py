import torch
import torch.nn as nn


class CrossModalMCQVQA(nn.Module):
    """
    SatQueryAI Cross-Modal Optical + SAR MCQ VQA specialist.

    Inputs:
        s1:
            [B, 196, 768]
        s2:
            [B, 196, 768]
        question:
            [B, 768]
        options:
            [B, 5, 768]

    Output:
        logits:
            [B, 5]

    Evidence:
        s1_to_s2_attention
        s2_to_s1_attention
        query_attention
        option_scores
        fused_tokens
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_heads=4,
        num_options=5,
    ):
        super().__init__()

        self.num_options = num_options

        # --------------------------------------------------
        # Modality projections
        # --------------------------------------------------

        self.s1_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.s2_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # --------------------------------------------------
        # Explicit cross-modal interaction
        # --------------------------------------------------

        self.s1_to_s2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.s2_to_s1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # --------------------------------------------------
        # Joint optical + SAR fusion
        # --------------------------------------------------

        fusion_dim = hidden_dim * 4

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # --------------------------------------------------
        # Question representation
        # --------------------------------------------------

        self.question_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # --------------------------------------------------
        # Question-guided spatial attention
        # --------------------------------------------------

        self.visual_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.question_query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        # --------------------------------------------------
        # Option representation
        # --------------------------------------------------

        self.option_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # --------------------------------------------------
        # Shared option scoring network
        #
        # Each option receives:
        #
        # visual summary
        # question
        # option
        # visual × question
        # visual × option
        # question × option
        # --------------------------------------------------

        scoring_dim = hidden_dim * 6

        self.option_scorer = nn.Sequential(
            nn.Linear(
                scoring_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        s1,
        s2,
        question,
        options,
        return_evidence=False,
    ):
        # --------------------------------------------------
        # Project both modalities
        # --------------------------------------------------

        s1_proj = self.s1_projection(s1)
        s2_proj = self.s2_projection(s2)

        # --------------------------------------------------
        # S1 attends to S2
        # --------------------------------------------------

        s1_cross, s1_attention = self.s1_to_s2(
            query=s1_proj,
            key=s2_proj,
            value=s2_proj,
            need_weights=True,
        )

        # --------------------------------------------------
        # S2 attends to S1
        # --------------------------------------------------

        s2_cross, s2_attention = self.s2_to_s1(
            query=s2_proj,
            key=s1_proj,
            value=s1_proj,
            need_weights=True,
        )

        # --------------------------------------------------
        # Joint cross-modal representation
        # --------------------------------------------------

        fused_input = torch.cat(
            [
                s1_proj,
                s2_proj,
                s1_cross,
                s2_cross,
            ],
            dim=-1,
        )

        fused = self.fusion(
            fused_input
        )

        # --------------------------------------------------
        # Question projection
        # --------------------------------------------------

        question_proj = self.question_projection(
            question
        )

        # --------------------------------------------------
        # Question-guided spatial attention
        # --------------------------------------------------

        keys = self.visual_key(
            fused
        )

        query = self.question_query(
            question_proj
        )

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = scores / (
            fused.size(-1) ** 0.5
        )

        query_attention = torch.softmax(
            scores,
            dim=1,
        )

        visual_summary = torch.bmm(
            query_attention.unsqueeze(1),
            fused,
        ).squeeze(1)

        # --------------------------------------------------
        # Option representations
        # --------------------------------------------------

        option_proj = self.option_projection(
            options
        )

        # --------------------------------------------------
        # Expand visual/question representation
        # for each answer option
        # --------------------------------------------------

        visual_expand = (
            visual_summary
            .unsqueeze(1)
            .expand(
                -1,
                self.num_options,
                -1,
            )
        )

        question_expand = (
            question_proj
            .unsqueeze(1)
            .expand(
                -1,
                self.num_options,
                -1,
            )
        )

        # --------------------------------------------------
        # Explicit multimodal interactions
        # --------------------------------------------------

        visual_question = (
            visual_expand
            * question_expand
        )

        visual_option = (
            visual_expand
            * option_proj
        )

        question_option = (
            question_expand
            * option_proj
        )

        # --------------------------------------------------
        # Option scoring features
        # --------------------------------------------------

        option_features = torch.cat(
            [
                visual_expand,
                question_expand,
                option_proj,
                visual_question,
                visual_option,
                question_option,
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # Shared scorer produces one score
        # per option
        # --------------------------------------------------

        logits = self.option_scorer(
            option_features
        ).squeeze(-1)

        if not return_evidence:
            return logits

        evidence = {
            "s1_to_s2_attention": s1_attention,
            "s2_to_s1_attention": s2_attention,
            "query_attention": query_attention,
            "fused_tokens": fused,
            "s1_tokens": s1_proj,
            "s2_tokens": s2_proj,
            "option_features": option_features,
            "option_scores": logits,
        }

        return logits, evidence