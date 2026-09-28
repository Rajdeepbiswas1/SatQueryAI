import torch
import torch.nn as nn

from models.feature_fusion import CrossModalFusion


class MCQOptionAwareVQA(nn.Module):
    """
    Option-aware BigEarthNet MCQ VQA specialist.

    Inputs:
        s1:        [B, 196, 768]
        s2:        [B, 196, 768]
        question:  [B, 768]
        options:   [B, 4, 768]

    Output:
        logits:    [B, 4]

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
        num_options=4,
    ):
        super().__init__()

        self.num_options = num_options

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
        # Option projection
        # ----------------------------------------------------

        self.option_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ----------------------------------------------------
        # Visual attention
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
        # Option scoring
        #
        # Each option gets a score based on:
        #   visual summary
        #   question
        #   option
        #   visual-option interaction
        #   question-option interaction
        # ----------------------------------------------------

        scoring_dim = hidden_dim * 5

        self.option_classifier = nn.Sequential(
            nn.Linear(
                scoring_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
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
    ):
        """
        Args:
            s1:
                [B, 196, 768]

            s2:
                [B, 196, 768]

            question:
                [B, 768]

            options:
                [B, 4, 768]

        Returns:
            [B, 4]
        """

        batch_size = s1.shape[0]

        # ----------------------------------------------------
        # Visual representation
        # ----------------------------------------------------

        visual = self.visual_fusion(
            s1,
            s2,
        )
        # [B, 196, 256]

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        question = self.question_projection(
            question
        )
        # [B, 256]

        # ----------------------------------------------------
        # Question-guided visual attention
        # ----------------------------------------------------

        keys = self.visual_key(
            visual
        )
        # [B, 196, 256]

        query = self.question_query(
            question
        )
        # [B, 256]

        attention_scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)
        # [B, 196]

        attention_scores = (
            attention_scores
            / (256 ** 0.5)
        )

        attention = torch.softmax(
            attention_scores,
            dim=1,
        )
        # [B, 196]

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)
        # [B, 256]

        # ----------------------------------------------------
        # Option representations
        # ----------------------------------------------------

        options = self.option_projection(
            options
        )
        # [B, 4, 256]

        # ----------------------------------------------------
        # Expand question and visual summary
        # for four options.
        # ----------------------------------------------------

        question_expanded = (
            question
            .unsqueeze(1)
            .expand(
                batch_size,
                self.num_options,
                -1,
            )
        )
        # [B, 4, 256]

        visual_expanded = (
            visual_summary
            .unsqueeze(1)
            .expand(
                batch_size,
                self.num_options,
                -1,
            )
        )
        # [B, 4, 256]

        # ----------------------------------------------------
        # Interactions
        # ----------------------------------------------------

        visual_option = (
            visual_expanded
            * options
        )
        # [B, 4, 256]

        question_option = (
            question_expanded
            * options
        )
        # [B, 4, 256]

        # ----------------------------------------------------
        # Combine information for each option
        #
        # visual
        # question
        # option
        # visual × option
        # question × option
        # ----------------------------------------------------

        option_features = torch.cat(
            [
                visual_expanded,
                question_expanded,
                options,
                visual_option,
                question_option,
            ],
            dim=-1,
        )
        # [B, 4, 1280]

        # ----------------------------------------------------
        # Score each option independently
        # ----------------------------------------------------

        scores = self.option_classifier(
            option_features
        )
        # [B, 4, 1]

        logits = scores.squeeze(-1)
        # [B, 4]

        return logits