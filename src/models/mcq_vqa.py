import torch
import torch.nn as nn

from models.feature_fusion import CrossModalFusion


class MCQVQAModel(nn.Module):
    """
    BigEarthNet MCQ VQA specialist.

    Inputs:
        s1:       [B, 196, 768]
        s2:       [B, 196, 768]
        question: [B, 768]

    Output:
        logits:   [B, 4]

    Class mapping:
        0 -> A
        1 -> B
        2 -> C
        3 -> D
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        num_classes=4,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Visual fusion
        # ----------------------------------------------------

        self.visual_fusion = CrossModalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        # ----------------------------------------------------
        # Question projection
        # ----------------------------------------------------

        self.question_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Question-guided visual attention
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
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        s1,
        s2,
        question,
    ):

        # [B, 196, 256]
        visual = self.visual_fusion(
            s1,
            s2,
        )

        # [B, 256]
        question = self.question_projection(
            question
        )

        # [B, 196, 256]
        keys = self.visual_key(
            visual
        )

        # [B, 256]
        query = self.question_query(
            question
        )

        # Calculate similarity between the
        # question and each spatial image token.
        #
        # Result: [B, 196]
        attention_scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        attention_scores = (
            attention_scores
            / (self.question_query.out_features ** 0.5)
        )

        # [B, 196]
        attention = torch.softmax(
            attention_scores,
            dim=1,
        )

        # Weighted sum of the 196 visual tokens.
        #
        # Result: [B, 256]
        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)

        # Question/visual interaction.
        #
        # Result: [B, 256]
        interaction = (
            visual_summary
            * question
        )

        # [B, 768]
        combined = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=1,
        )

        # [B, 4]
        logits = self.classifier(
            combined
        )

        return logits