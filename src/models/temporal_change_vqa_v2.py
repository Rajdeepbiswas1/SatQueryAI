from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.feature_fusion import TemporalFusion
except ImportError:
    from .feature_fusion import TemporalFusion


class TemporalChangeVQAv2(nn.Module):
    """
    Temporal Change VQA v2.

    Evidence-based change from the frozen v1/balanced temporal model:
      1. Remove the learned attention_query bottleneck.
      2. Use question_projection output directly as the spatial query.
      3. L2-normalize query and key.
      4. Use a fixed temperature of 48.0 for the cosine attention scores.

    Input:
        t1:       [B, 196, 768]
        t2:       [B, 196, 768]
        diff:     [B, 196, 768]
        question: [B, 768]

    Output:
        answer_logits:
            binary              [B, 2]
            change_ratio        [B, 11]
            change_ratio_types  [B, 9]
            land_cover          [B, 6]

        query_attention: [B, 14, 14]
        change_logits:   [B, 14, 14]
        fused_tokens:    [B, 196, 256]
        joint:           [B, 256]

    Expected trainable parameter count:
        1,679,133

    This model intentionally has no trainable attention-query layer.
    """

    ATTENTION_TEMPERATURE = 48.0

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_spatial_tokens: int = 196,
        grid_size: int = 14,
    ) -> None:
        super().__init__()

        if num_spatial_tokens != grid_size * grid_size:
            raise ValueError(
                "num_spatial_tokens must equal grid_size * grid_size: "
                f"{num_spatial_tokens} != {grid_size}^2"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.grid_size = int(grid_size)

        self.temporal_fusion = TemporalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.question_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Deliberately removed in v2.
        # The v1 diagnostic showed that the learned 256->256
        # attention_query made question directions nearly identical.

        self.attention_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.attention_value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.joint_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 11),
        )

        self.ratio_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 9),
        )

        self.land_cover_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

        self.change_evidence_head = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                128,
                1,
                kernel_size=1,
            ),
        )

    def _validate_inputs(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
        question: torch.Tensor,
    ) -> None:
        expected_spatial = self.num_spatial_tokens
        expected_feature = self.input_dim

        for name, tensor in (
            ("t1", t1),
            ("t2", t2),
            ("diff", diff),
        ):
            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} must have shape "
                    f"[B,{expected_spatial},{expected_feature}], "
                    f"got {tuple(tensor.shape)}"
                )

            if tensor.shape[1] != expected_spatial:
                raise ValueError(
                    f"{name} has {tensor.shape[1]} spatial tokens; "
                    f"expected {expected_spatial}"
                )

            if tensor.shape[2] != expected_feature:
                raise ValueError(
                    f"{name} has feature dimension {tensor.shape[2]}; "
                    f"expected {expected_feature}"
                )

        if not (t1.shape == t2.shape == diff.shape):
            raise ValueError(
                "Temporal input shapes must match: "
                f"t1={tuple(t1.shape)}, "
                f"t2={tuple(t2.shape)}, "
                f"diff={tuple(diff.shape)}"
            )

        if question.ndim != 2:
            raise ValueError(
                f"question must have shape [B,{expected_feature}], "
                f"got {tuple(question.shape)}"
            )

        if question.shape[0] != t1.shape[0]:
            raise ValueError(
                "Batch size mismatch between temporal inputs and question: "
                f"temporal={t1.shape[0]}, question={question.shape[0]}"
            )

        if question.shape[1] != expected_feature:
            raise ValueError(
                f"question has feature dimension {question.shape[1]}; "
                f"expected {expected_feature}"
            )

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
        question: torch.Tensor,
    ) -> dict[str, object]:
        self._validate_inputs(
            t1=t1,
            t2=t2,
            diff=diff,
            question=question,
        )

        batch_size = t1.shape[0]

        # --------------------------------------------------------
        # Temporal visual representation
        # --------------------------------------------------------

        fused_tokens = self.temporal_fusion(
            t1,
            t2,
            diff,
        )
        # [B,196,256]

        # --------------------------------------------------------
        # Question representation
        # --------------------------------------------------------

        question_feature = self.question_projection(
            question.float()
        )
        # [B,256]

        # --------------------------------------------------------
        # v2 question-conditioned spatial attention
        #
        # No learned attention_query projection.
        # The question projection is used directly.
        # --------------------------------------------------------

        query = F.normalize(
            question_feature,
            p=2,
            dim=-1,
            eps=1e-8,
        )
        # [B,256]

        key = self.attention_key(
            fused_tokens
        )
        # [B,196,256]

        key = F.normalize(
            key,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        value = self.attention_value(
            fused_tokens
        )
        # [B,196,256]

        cosine_scores = (
            key
            * query.unsqueeze(1)
        ).sum(dim=-1)
        # [B,196]

        attention_scores = (
            cosine_scores
            * self.ATTENTION_TEMPERATURE
        )

        query_attention_flat = torch.softmax(
            attention_scores,
            dim=-1,
        )
        # [B,196]

        attended_temporal = torch.sum(
            value
            * query_attention_flat.unsqueeze(-1),
            dim=1,
        )
        # [B,256]

        global_temporal = fused_tokens.mean(
            dim=1
        )
        # [B,256]

        # --------------------------------------------------------
        # Joint representation
        # --------------------------------------------------------

        joint_input = torch.cat(
            [
                attended_temporal,
                global_temporal,
            ],
            dim=-1,
        )
        # [B,512]

        joint = self.joint_projection(
            joint_input
        )
        # [B,256]

        # --------------------------------------------------------
        # Answer heads
        # --------------------------------------------------------

        answer_logits = {
            "binary": self.binary_head(joint),
            "change_ratio": self.ratio_head(joint),
            "change_ratio_types": self.ratio_type_head(joint),
            "land_cover": self.land_cover_head(joint),
        }

        # --------------------------------------------------------
        # Spatial attention output
        # --------------------------------------------------------

        query_attention = query_attention_flat.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # --------------------------------------------------------
        # Spatial evidence map
        # --------------------------------------------------------

        feature_map = fused_tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            self.hidden_dim,
            self.grid_size,
            self.grid_size,
        )

        change_logits = self.change_evidence_head(
            feature_map
        ).squeeze(1)
        # [B,14,14]

        return {
            "answer_logits": answer_logits,
            "query_attention": query_attention,
            "change_logits": change_logits,
            "fused_tokens": fused_tokens,
            "joint": joint,
        }


# Compatibility alias for code that expects the original class name.
TemporalChangeVQA = TemporalChangeVQAv2


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the exact number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "TemporalChangeVQAv2",
    "TemporalChangeVQA",
    "count_trainable_parameters",
]
