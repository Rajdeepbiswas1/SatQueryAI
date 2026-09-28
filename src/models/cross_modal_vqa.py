import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalVQA(nn.Module):
    """
    SatQueryAI Cross-Modal Optical + SAR VQA specialist.

    Inputs:
        s1       [B, 196, 768]  -> SAR / Sentinel-1
        s2       [B, 196, 768]  -> Optical / Sentinel-2
        question [B, 768]

    Output:
        logits   [B, 2]
        evidence dictionary containing modality/joint attention.
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_heads=4,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Separate modality projections
        # ----------------------------------------------------

        self.s1_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.s2_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Bidirectional cross-modal interaction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Explicit multimodal fusion
        #
        # original S1
        # original S2
        # cross-attended S1
        # cross-attended S2
        # ----------------------------------------------------

        fusion_dim = hidden_dim * 4

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Question projection
        # ----------------------------------------------------

        self.question_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Query-conditioned visual attention
        # ----------------------------------------------------

        self.visual_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.question_query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Final binary VQA classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        s1,
        s2,
        question,
        return_evidence=False,
    ):
        # ----------------------------------------------------
        # Project each modality independently
        # ----------------------------------------------------

        s1_proj = self.s1_projection(s1)
        s2_proj = self.s2_projection(s2)

        # ----------------------------------------------------
        # Bidirectional cross-modal reasoning
        #
        # S1 asks:
        # "Which S2 tokens are relevant to this SAR token?"
        #
        # S2 asks:
        # "Which S1 tokens are relevant to this optical token?"
        # ----------------------------------------------------

        s1_cross, s1_attention = self.s1_to_s2(
            query=s1_proj,
            key=s2_proj,
            value=s2_proj,
            need_weights=True,
        )

        s2_cross, s2_attention = self.s2_to_s1(
            query=s2_proj,
            key=s1_proj,
            value=s1_proj,
            need_weights=True,
        )

        # ----------------------------------------------------
        # Fuse all four representations
        # ----------------------------------------------------

        fused_input = torch.cat(
            [
                s1_proj,
                s2_proj,
                s1_cross,
                s2_cross,
            ],
            dim=-1,
        )

        fused = self.fusion(fused_input)

        # ----------------------------------------------------
        # Question conditioning
        # ----------------------------------------------------

        question_proj = self.question_projection(
            question
        )

        keys = self.visual_key(fused)
        query = self.question_query(question_proj)

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = scores / (fused.size(-1) ** 0.5)

        attention = torch.softmax(
            scores,
            dim=1,
        )

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            fused,
        ).squeeze(1)

        # ----------------------------------------------------
        # Question/visual interaction
        # ----------------------------------------------------

        interaction = (
            visual_summary
            * question_proj
        )

        combined = torch.cat(
            [
                visual_summary,
                question_proj,
                interaction,
            ],
            dim=-1,
        )

        logits = self.classifier(
            combined
        )

        if not return_evidence:
            return logits

        evidence = {
            "s1_to_s2_attention": s1_attention,
            "s2_to_s1_attention": s2_attention,
            "query_attention": attention,
            "fused_tokens": fused,
            "s1_tokens": s1_proj,
            "s2_tokens": s2_proj,
        }

        return logits, evidence