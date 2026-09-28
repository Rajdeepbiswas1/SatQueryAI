from typing import Any, Optional

import torch

from agent.bootstrap import create_runtime_agent

from backend.single_modality_feature_service import (
    SingleModalityFeatureExtractionError,
    SingleModalityFeatureService,
)

from backend.text_embedding_service import (
    SatQueryAITextEmbeddingService,
)


class SingleModalityGroundingInferenceError(
    RuntimeError
):
    """Raised when single-modality grounding inference fails."""


class SatQueryAISingleModalityGroundingInferenceService:
    """
    Application-facing single-modality grounding service.

    Pipeline:

        raw S1/S2 image
            ↓
        SingleModalityFeatureService
            ↓
        [1,196,768]
            ↓
        text embedding [1,768]
            ↓
        SatQueryAI Agent
            ↓
        S1 or S2 Grounding V1
            ↓
        bbox + heatmap + confidence + evidence
    """

    SUPPORTED_MODALITIES = {
        "s1",
        "s2",
    }

    EXPECTED_FEATURE_SHAPE = (
        1,
        196,
        768,
    )

    EXPECTED_TEXT_SHAPE = (
        1,
        768,
    )

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = device

        self.feature_service = (
            SingleModalityFeatureService(
                device=device
            )
        )

        self.text_service = (
            SatQueryAITextEmbeddingService(
                device=device
            )
        )

        self.agent, _, _ = create_runtime_agent()

        self.model_name = (
            "SatQueryAISingleModalityGroundingInference"
        )

    def _validate_modality(
        self,
        modality: str,
    ) -> str:

        if not isinstance(
            modality,
            str,
        ):
            raise SingleModalityGroundingInferenceError(
                "Modality must be a string."
            )

        modality = modality.strip().lower()

        if modality not in self.SUPPORTED_MODALITIES:
            raise SingleModalityGroundingInferenceError(
                "Unsupported modality. Expected "
                f"'s1' or 's2', got '{modality}'."
            )

        return modality

    def _validate_point(
        self,
        point: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:

        if point is None:
            return None

        if not isinstance(
            point,
            torch.Tensor,
        ):
            point = torch.tensor(
                point,
                dtype=torch.float32,
            )

        if point.ndim == 1:
            point = point.unsqueeze(0)

        if tuple(point.shape) != (
            1,
            2,
        ):
            raise SingleModalityGroundingInferenceError(
                "Point must have shape [1,2], got "
                f"{tuple(point.shape)}"
            )

        point = point.float().cpu()

        if not torch.isfinite(
            point
        ).all().item():
            raise SingleModalityGroundingInferenceError(
                "Point contains NaN or Inf."
            )

        if (
            (point < 0).any().item()
            or (point > 1).any().item()
        ):
            raise SingleModalityGroundingInferenceError(
                "Point coordinates must be normalized to [0,1]."
            )

        return point

    def execute(
        self,
        query: str,
        image: torch.Tensor,
        modality: str,
        point: Optional[torch.Tensor] = None,
        reference: Optional[str] = None,
    ) -> dict[str, Any]:

        if not isinstance(
            query,
            str,
        ):
            raise SingleModalityGroundingInferenceError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise SingleModalityGroundingInferenceError(
                "Query cannot be empty."
            )

        modality = self._validate_modality(
            modality
        )

        point = self._validate_point(
            point
        )

        if reference is not None:

            if not isinstance(
                reference,
                str,
            ):
                raise SingleModalityGroundingInferenceError(
                    "Reference must be a string."
                )

            reference = reference.strip()

            if not reference:
                reference = None

        try:

            visual_features, feature_metadata = (
                self.feature_service.extract(
                    image=image,
                    modality=modality,
                )
            )

        except (
            SingleModalityFeatureExtractionError
        ) as exc:

            raise SingleModalityGroundingInferenceError(
                f"Single-modality feature extraction failed: {exc}"
            ) from exc

        if tuple(
            visual_features.shape
        ) != self.EXPECTED_FEATURE_SHAPE:
            raise SingleModalityGroundingInferenceError(
                "Unexpected visual feature shape: "
                f"{tuple(visual_features.shape)}"
            )

        try:

            text_embedding, text_metadata = (
                self.text_service.embed(
                    query
                )
            )

        except Exception as exc:

            raise SingleModalityGroundingInferenceError(
                f"Text embedding failed: {exc}"
            ) from exc

        if tuple(
            text_embedding.shape
        ) != self.EXPECTED_TEXT_SHAPE:
            raise SingleModalityGroundingInferenceError(
                "Unexpected text embedding shape: "
                f"{tuple(text_embedding.shape)}"
            )

        agent_result = self.agent.execute(
            query=query,
            input_modalities=[
                modality
            ],
            s1=(
                visual_features
                if modality == "s1"
                else None
            ),
            s2=(
                visual_features
                if modality == "s2"
                else None
            ),
            text_embedding=text_embedding,
            point=point,
            reference=reference,
        )

        if not isinstance(
            agent_result,
            dict,
        ):
            raise SingleModalityGroundingInferenceError(
                "Agent returned an invalid result."
            )

        if agent_result.get(
            "answer_status"
        ) != "success":
            raise SingleModalityGroundingInferenceError(
                "Grounding agent execution failed: "
                f"{agent_result}"
            )

        analysis = agent_result.get(
            "analysis",
            {},
        )

        if not isinstance(
            analysis,
            dict,
        ):
            raise SingleModalityGroundingInferenceError(
                "Agent returned invalid grounding analysis."
            )

        if analysis.get(
            "task"
        ) != "grounding":
            raise SingleModalityGroundingInferenceError(
                "Single-modality grounding request was routed to "
                f"unexpected task: {analysis.get('task')}"
            )

        execution_trace = agent_result.get(
            "execution_trace",
            {},
        )

        if not isinstance(
            execution_trace,
            dict,
        ):
            raise SingleModalityGroundingInferenceError(
                "Agent returned invalid execution trace."
            )

        return {
            "status": "success",
            "query": query,
            "input": {
                "modality": modality,
                "feature_shape": list(
                    visual_features.shape
                ),
                "point": (
                    point.tolist()
                    if point is not None
                    else None
                ),
                "reference": reference,
            },
            "feature_extraction": feature_metadata,
            "text_embedding": text_metadata,
            "analysis": analysis,
            "execution_trace": execution_trace,
            "response": agent_result,
        }


__all__ = [
    "SatQueryAISingleModalityGroundingInferenceService",
    "SingleModalityGroundingInferenceError",
]