from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel

from models.feature_fusion import CrossModalFusion


class BinaryVQAModel(nn.Module):
    """
    First real SatQueryAI VQA model.

    Inputs
    ------
    s1:
        [B, 196, 768]

    s2:
        [B, 196, 768]

    input_ids:
        [B, sequence_length]

    attention_mask:
        [B, sequence_length]

    Output
    ------
    logits:
        [B, 2]

    Class mapping
    -------------
    0 -> no
    1 -> yes
    """

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        visual_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()

        if num_classes != 2:
            raise ValueError(
                "BinaryVQAModel requires exactly 2 classes."
            )

        # ----------------------------------------------------
        # Vision side
        # ----------------------------------------------------
        self.visual_fusion = CrossModalFusion(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
        )

        # ----------------------------------------------------
        # Language side
        # ----------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        text_dim = self.text_encoder.config.hidden_size

        self.question_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
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
        # Final multimodal classifier
        #
        # visual = 256
        # question = 256
        # interaction = 256
        #
        # total = 768
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        # ----------------------------------------------------
        # Optionally freeze DistilBERT
        # ----------------------------------------------------
        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(False)

    def encode_question(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert tokenized question into one vector.

        Returns:
            [B, hidden_dim]
        """

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = text_output.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        masked_hidden = hidden_states * mask

        summed = masked_hidden.sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        question = summed / lengths

        return self.question_projection(question)

    def attend_to_visual_features(
        self,
        visual_tokens: torch.Tensor,
        question: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the question to determine which visual tokens matter.

        visual_tokens:
            [B, 196, 256]

        question:
            [B, 256]

        Returns:
            visual_summary:
                [B, 256]
        """

        keys = self.visual_key(visual_tokens)

        query = self.question_query(question)

        # [B, 196, 256] × [B, 256]
        # -> [B, 196]
        scores = (
            keys * query.unsqueeze(1)
        ).sum(dim=-1)

        scores = scores / math.sqrt(
            visual_tokens.shape[-1]
        )

        attention = torch.softmax(
            scores,
            dim=1,
        )

        # Weighted sum of 196 spatial locations
        visual_summary = (
            visual_tokens
            * attention.unsqueeze(-1)
        ).sum(dim=1)

        return visual_summary

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Visual representation
        # ----------------------------------------------------
        visual_tokens = self.visual_fusion(
            s1,
            s2,
        )

        # ----------------------------------------------------
        # Question representation
        # ----------------------------------------------------
        question = self.encode_question(
            input_ids,
            attention_mask,
        )

        # ----------------------------------------------------
        # Question-guided visual attention
        # ----------------------------------------------------
        visual_summary = self.attend_to_visual_features(
            visual_tokens,
            question,
        )

        # ----------------------------------------------------
        # Multimodal interaction
        # ----------------------------------------------------
        interaction = (
            visual_summary * question
        )

        multimodal = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Binary answer logits
        # ----------------------------------------------------
        logits = self.classifier(multimodal)

        return logits