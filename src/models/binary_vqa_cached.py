import torch
import torch.nn as nn

from models.feature_fusion import CrossModalFusion


class CachedBinaryVQA(nn.Module):
    """
    Frozen Binary VQA inference model.

    This architecture is identical to the model used by
    train_binary_vqa_cached.py.

    Inputs:
        s1:       [B, 196, 768]
        s2:       [B, 196, 768]
        question: [B, 768]

    Output:
        logits:   [B, 2]

    Class mapping:
        0 = no
        1 = yes
    """

    def __init__(self):
        super().__init__()

        self.visual_fusion = CrossModalFusion(
            input_dim=768,
            hidden_dim=256,
        )

        self.question_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        self.visual_key = nn.Linear(
            256,
            256,
        )

        self.question_query = nn.Linear(
            256,
            256,
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        s1,
        s2,
        question,
    ):
        visual = self.visual_fusion(
            s1,
            s2,
        )

        question = self.question_projection(
            question
        )

        keys = self.visual_key(
            visual
        )

        query = self.question_query(
            question
        )

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = scores / (256 ** 0.5)

        attention = torch.softmax(
            scores,
            dim=1,
        )

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)

        interaction = (
            visual_summary
            * question
        )

        combined = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=1,
        )

        return self.classifier(
            combined
        )