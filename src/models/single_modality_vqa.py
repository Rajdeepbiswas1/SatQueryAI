"""
SatQueryAI — True Single-Modality VQA Models

Checkpoint-compatible inference models for:
    S1 Binary VQA
    S2 Binary VQA
    S1 MCQ VQA
    S2 MCQ VQA

Expected checkpoint architecture:
    Visual projection: 768 -> 256
    Question projection: 768 -> 256
    Cosine attention over 196 visual tokens
    Fusion: 512 -> 256 + GELU + LayerNorm

Binary head:
    256 -> 128 -> 2

Option-aware MCQ head:
    Option projection: 768 -> 256
    Option scorer: 768 -> 256 -> 1

The trained checkpoints are modality-specific through the selected
checkpoint; the architecture itself is shared.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleModalityEncoder(nn.Module):
    """
    Shared encoder used by the trained single-modality VQA models.

    Inputs:
        visual_tokens: [B, 196, 768]
        question_embedding: [B, 768]

    Returns:
        fused_features: [B, 256]
        attention: [B, 196]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        question_dim: int = 768,
        hidden_dim: int = 256,
        attention_temperature: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.attention_temperature = float(attention_temperature)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_embedding: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"visual_tokens must have shape [B, N, D], got {tuple(visual_tokens.shape)}"
            )

        if question_embedding.ndim != 2:
            raise ValueError(
                "question_embedding must have shape [B, D], "
                f"got {tuple(question_embedding.shape)}"
            )

        if visual_tokens.shape[0] != question_embedding.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"visual={visual_tokens.shape[0]}, "
                f"question={question_embedding.shape[0]}"
            )

        visual = self.visual_proj(visual_tokens)
        question = self.question_proj(question_embedding)

        # Query-conditioned cosine attention over spatial visual tokens.
        visual_norm = F.normalize(visual, p=2, dim=-1)
        question_norm = F.normalize(question, p=2, dim=-1)

        attention_logits = torch.sum(
            visual_norm * question_norm.unsqueeze(1),
            dim=-1,
        )

        attention_logits = attention_logits * self.attention_temperature
        attention = torch.softmax(attention_logits, dim=1)

        # Weighted visual context.
        context = torch.sum(
            visual * attention.unsqueeze(-1),
            dim=1,
        )

        fused = self.fusion(torch.cat([context, question], dim=-1))
        fused = self.dropout(fused)

        if return_attention:
            return fused, attention

        return fused, None


class SingleModalityBinaryVQA(nn.Module):
    """
    Binary VQA model for one modality.

    Output:
        logits: [B, 2]
        attention: [B, 196]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        question_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = SingleModalityEncoder(
            visual_dim=visual_dim,
            question_dim=question_dim,
            hidden_dim=hidden_dim,
            attention_temperature=16.0,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_embedding: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        fused, attention = self.encoder(
            visual_tokens,
            question_embedding,
            return_attention=return_attention,
        )

        logits = self.classifier(fused)

        result = {"logits": logits}

        if return_attention and attention is not None:
            result["attention"] = attention

        return result


class SingleModalityOptionAwareMCQVQA(nn.Module):
    """
    Option-aware MCQ model for one modality.

    Inputs:
        visual_tokens:       [B, 196, 768]
        question_embedding:  [B, 768]
        option_embeddings:   [B, 4, 768]

    Outputs:
        logits:    [B, 4]
        attention: [B, 196]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        question_dim: int = 768,
        hidden_dim: int = 256,
        num_options: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_options <= 0:
            raise ValueError("num_options must be positive")

        self.num_options = num_options

        self.encoder = SingleModalityEncoder(
            visual_dim=visual_dim,
            question_dim=question_dim,
            hidden_dim=hidden_dim,
            attention_temperature=16.0,
            dropout=dropout,
        )

        self.option_proj = nn.Linear(visual_dim, hidden_dim)

        self.option_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_embedding: torch.Tensor,
        option_embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if option_embeddings.ndim != 3:
            raise ValueError(
                "option_embeddings must have shape [B, K, D], "
                f"got {tuple(option_embeddings.shape)}"
            )

        if option_embeddings.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "Batch size mismatch between visual_tokens and option_embeddings"
            )

        if option_embeddings.shape[1] != self.num_options:
            raise ValueError(
                f"Expected {self.num_options} options, "
                f"got {option_embeddings.shape[1]}"
            )

        fused, attention = self.encoder(
            visual_tokens,
            question_embedding,
            return_attention=return_attention,
        )

        question = self.encoder.question_proj(question_embedding)
        options = self.option_proj(option_embeddings)

        batch_size, num_options, _ = options.shape

        fused_expanded = fused.unsqueeze(1).expand(
            -1,
            num_options,
            -1,
        )

        question_expanded = question.unsqueeze(1).expand(
            -1,
            num_options,
            -1,
        )

        scorer_input = torch.cat(
            [
                fused_expanded,
                question_expanded,
                options,
            ],
            dim=-1,
        )

        logits = self.option_scorer(scorer_input).squeeze(-1)

        result = {"logits": logits}

        if return_attention and attention is not None:
            result["attention"] = attention

        return result


def load_single_modality_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> Dict:
    """
    Load a trained single-modality checkpoint strictly.

    Returns checkpoint metadata.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a valid state dict"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable + non-trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())