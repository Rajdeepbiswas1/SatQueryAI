from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.feature_fusion import TemporalFusion


# =============================================================================
# VERIFIED TEMPORAL ANSWER VOCABULARIES
# =============================================================================

BINARY_CLASSES: Tuple[str, ...] = (
    "no",
    "yes",
)

LAND_COVER_CLASSES: Tuple[str, ...] = (
    "buildings",
    "trees",
    "low_vegetation",
    "NVG_surface",
    "water",
    "playgrounds",
)

RATIO_CLASSES: Tuple[str, ...] = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
)

RATIO_TYPE_CLASSES: Tuple[str, ...] = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
)

QUESTION_TYPES: Tuple[str, ...] = (
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "smallest_change",
    "largest_change",
    "change_to_what",
    "change_ratio",
    "change_ratio_types",
)


# =============================================================================
# TEMPORAL CHANGE VQA
# =============================================================================

class TemporalChangeVQA(nn.Module):
    """
    Lightweight temporal specialist for SatQueryAI.

    Inputs:
        t1       [B,196,768]
        t2       [B,196,768]
        diff     [B,196,768]
        question [B,768]

    Outputs:
        answer_logits
        query_attention [B,14,14]
        change_logits   [B,14,14]
        fused_tokens    [B,196,256]
        joint           [B,256]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        grid_size: int = 14,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        if grid_size * grid_size != 196:
            raise ValueError(
                "TemporalChangeVQA requires a 14x14 grid "
                "containing exactly 196 spatial tokens."
            )

        # ---------------------------------------------------------------------
        # 1. TEMPORAL FUSION
        # ---------------------------------------------------------------------

        self.temporal_fusion = TemporalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        # ---------------------------------------------------------------------
        # 2. QUESTION PROJECTION
        # ---------------------------------------------------------------------

        self.question_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        # ---------------------------------------------------------------------
        # 3. QUESTION-GUIDED ATTENTION
        # ---------------------------------------------------------------------

        self.attention_query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.attention_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.attention_value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        # ---------------------------------------------------------------------
        # 4. JOINT REPRESENTATION
        # ---------------------------------------------------------------------

        self.joint_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        # ---------------------------------------------------------------------
        # 5. TASK-SPECIFIC HEADS
        # ---------------------------------------------------------------------

        self.binary_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                len(BINARY_CLASSES),
            ),
        )

        self.ratio_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                len(RATIO_CLASSES),
            ),
        )

        self.ratio_type_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                len(RATIO_TYPE_CLASSES),
            ),
        )

        self.land_cover_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                len(LAND_COVER_CLASSES),
            ),
        )

        # ---------------------------------------------------------------------
        # 6. CHANGE EVIDENCE MAP
        # ---------------------------------------------------------------------

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

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def _validate_inputs(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
        question: torch.Tensor,
    ) -> None:

        tensors = {
            "t1": t1,
            "t2": t2,
            "diff": diff,
            "question": question,
        }

        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

        expected_feature_shape = (
            t1.shape[0],
            self.grid_size * self.grid_size,
            self.input_dim,
        )

        if tuple(t1.shape) != expected_feature_shape:
            raise ValueError(
                f"Unexpected t1 shape. "
                f"Expected {expected_feature_shape}, "
                f"got {tuple(t1.shape)}"
            )

        if tuple(t2.shape) != expected_feature_shape:
            raise ValueError(
                f"Unexpected t2 shape. "
                f"Expected {expected_feature_shape}, "
                f"got {tuple(t2.shape)}"
            )

        if tuple(diff.shape) != expected_feature_shape:
            raise ValueError(
                f"Unexpected diff shape. "
                f"Expected {expected_feature_shape}, "
                f"got {tuple(diff.shape)}"
            )

        expected_question_shape = (
            t1.shape[0],
            self.input_dim,
        )

        if tuple(question.shape) != expected_question_shape:
            raise ValueError(
                f"Unexpected question shape. "
                f"Expected {expected_question_shape}, "
                f"got {tuple(question.shape)}"
            )

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
        question: torch.Tensor,
    ) -> Dict[str, object]:

        self._validate_inputs(
            t1=t1,
            t2=t2,
            diff=diff,
            question=question,
        )

        batch_size = t1.shape[0]

        # ---------------------------------------------------------------------
        # 1. TEMPORAL FUSION
        # ---------------------------------------------------------------------

        fused_tokens = self.temporal_fusion(
            t1=t1,
            t2=t2,
            diff=diff,
        )

        # [B,196,256]

        # ---------------------------------------------------------------------
        # 2. QUESTION REPRESENTATION
        # ---------------------------------------------------------------------

        question_feature = self.question_projection(
            question.float()
        )

        # [B,256]

        # ---------------------------------------------------------------------
        # 3. ATTENTION
        # ---------------------------------------------------------------------

        query = self.attention_query(
            question_feature
        )

        key = self.attention_key(
            fused_tokens
        )

        value = self.attention_value(
            fused_tokens
        )

        scale = math.sqrt(
            float(self.hidden_dim)
        )

        attention_scores = (
            key
            * query.unsqueeze(1)
        ).sum(
            dim=-1
        ) / scale

        # [B,196]

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

        # ---------------------------------------------------------------------
        # 4. GLOBAL TEMPORAL REPRESENTATION
        # ---------------------------------------------------------------------

        global_temporal = fused_tokens.mean(
            dim=1
        )

        # [B,256]

        # ---------------------------------------------------------------------
        # 5. JOINT REPRESENTATION
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # 6. ANSWER HEADS
        # ---------------------------------------------------------------------

        answer_logits = {
            "binary": self.binary_head(
                joint
            ),

            "change_ratio": self.ratio_head(
                joint
            ),

            "change_ratio_types": self.ratio_type_head(
                joint
            ),

            "land_cover": self.land_cover_head(
                joint
            ),
        }

        # ---------------------------------------------------------------------
        # 7. QUERY ATTENTION MAP
        # ---------------------------------------------------------------------

        query_attention = query_attention_flat.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
        )

        # [B,14,14]

        # ---------------------------------------------------------------------
        # 8. CHANGE EVIDENCE MAP
        # ---------------------------------------------------------------------

        feature_map = fused_tokens.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            self.hidden_dim,
            self.grid_size,
            self.grid_size,
        )

        # [B,256,14,14]

        change_logits = self.change_evidence_head(
            feature_map
        ).squeeze(
            1
        )

        # [B,14,14]

        return {
            "answer_logits": answer_logits,
            "query_attention": query_attention,
            "change_logits": change_logits,
            "fused_tokens": fused_tokens,
            "joint": joint,
        }

    # =========================================================================
    # QUESTION-TYPE ROUTING
    # =========================================================================

    def select_answer_logits(
        self,
        outputs,
        question_type: str,
    ) -> torch.Tensor:
        """
        Select the answer-logit head corresponding to a temporal
        question type.

        Accepts either:
            1. the complete model output dictionary containing
            "answer_logits", or
            2. the answer-logits dictionary itself.
        """

        # ------------------------------------------------------------
        # Resolve the answer-logits dictionary.
        # ------------------------------------------------------------

        if not isinstance(outputs, dict):
            raise TypeError(
                "outputs must be a dictionary."
            )

        if "answer_logits" in outputs:
            answer_logits = outputs["answer_logits"]
        else:
            answer_logits = outputs

        if not isinstance(answer_logits, dict):
            raise TypeError(
                "answer_logits must be a dictionary."
            )

        # ------------------------------------------------------------
        # Map temporal question type -> answer head.
        # ------------------------------------------------------------

        question_type_to_head = {
            "change_or_not": "binary",
            "increase_or_not": "binary",
            "decrease_or_not": "binary",

            "smallest_change": "land_cover",
            "largest_change": "land_cover",
            "change_to_what": "land_cover",

            "change_ratio": "change_ratio",
            "change_ratio_types": "change_ratio_types",
        }

        if question_type not in question_type_to_head:
            raise KeyError(
                f"Unsupported temporal question type: "
                f"{question_type!r}. "
                f"Supported types: "
                f"{tuple(question_type_to_head.keys())}"
            )

        head_name = question_type_to_head[question_type]

        # ------------------------------------------------------------
        # Validate requested head.
        # ------------------------------------------------------------

        if head_name not in answer_logits:
            raise KeyError(
                f"Model outputs do not contain the required "
                f"answer-logit head {head_name!r}."
            )

        logits = answer_logits[head_name]

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"Answer logits for head {head_name!r} "
                f"must be a torch.Tensor."
            )

        return logits
    # =========================================================================
    # ANSWER VOCABULARY
    # =========================================================================

    @staticmethod
    def get_answer_classes(
        question_type: str,
    ) -> Tuple[str, ...]:

        question_type = str(
            question_type
        ).strip().lower()

        binary_types = {
            "change_or_not",
            "increase_or_not",
            "decrease_or_not",
        }

        land_cover_types = {
            "smallest_change",
            "largest_change",
            "change_to_what",
        }

        if question_type in binary_types:
            return BINARY_CLASSES

        if question_type == "change_ratio":
            return RATIO_CLASSES

        if question_type == "change_ratio_types":
            return RATIO_TYPE_CLASSES

        if question_type in land_cover_types:
            return LAND_COVER_CLASSES

        raise ValueError(
            f"Unsupported question type: "
            f"{question_type!r}. "
            f"Supported types: {QUESTION_TYPES}"
        )

    # =========================================================================
    # DECODE PREDICTION
    # =========================================================================

    @classmethod
    def decode_prediction(
        cls,
        question_type: str,
        logits: torch.Tensor,
    ):
        """
        Decode output logits into the exact dataset answer string.
        """

        classes = cls.get_answer_classes(
            question_type
        )

        if not torch.is_tensor(logits):
            raise TypeError(
                "logits must be a torch.Tensor."
            )

        if logits.ndim != 2:
            raise ValueError(
                f"Expected [B,C] logits, "
                f"got {tuple(logits.shape)}"
            )

        if logits.shape[1] != len(classes):
            raise ValueError(
                f"Logits contain {logits.shape[1]} classes, "
                f"but {question_type!r} requires "
                f"{len(classes)} classes."
            )

        indices = torch.argmax(
            logits,
            dim=-1,
        ).detach().cpu().tolist()

        answers = [
            classes[int(index)]
            for index in indices
        ]

        if len(answers) == 1:
            return answers[0]

        return answers


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    "TemporalChangeVQA",
    "BINARY_CLASSES",
    "LAND_COVER_CLASSES",
    "RATIO_CLASSES",
    "RATIO_TYPE_CLASSES",
    "QUESTION_TYPES",
]