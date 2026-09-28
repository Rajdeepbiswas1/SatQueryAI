from typing import Any

import torch

from agent.bootstrap import create_runtime_agent
from agent.core import QueryAnalysis

from backend.single_modality_feature_service import (
    SingleModalityFeatureExtractionError,
    SingleModalityFeatureService,
)

from backend.text_embedding_service import (
    SatQueryAITextEmbeddingService,
)


class SingleModalityVQAInferenceError(
    RuntimeError
):
    """Raised when single-modality VQA inference fails."""


class SatQueryAISingleModalityVQAInferenceService:
    """
    Application-facing single-modality VQA inference service.

    Pipeline:

        raw modality tensor
            ↓
        SingleModalityFeatureService
            ↓
        [1,196,768]
            ↓
        SatQueryAITextEmbeddingService
            ↓
        [1,768]
            ↓
        SatQueryAI Agent
            ↓
        modality-specific VQA specialist
    """

    SUPPORTED_MODALITIES = {
        "s1",
        "s2",
    }

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
            "SatQueryAISingleModalityVQAInference"
        )

    def _validate_modality(
        self,
        modality: str,
    ) -> str:

        if not isinstance(
            modality,
            str,
        ):
            raise SingleModalityVQAInferenceError(
                "Modality must be a string."
            )

        modality = modality.strip().lower()

        if modality not in self.SUPPORTED_MODALITIES:
            raise SingleModalityVQAInferenceError(
                "Unsupported modality. Expected "
                f"'s1' or 's2', got '{modality}'."
            )

        return modality

    def _validate_options(
        self,
        options: list[str] | None,
    ) -> list[str] | None:

        if options is None:
            return None

        if not isinstance(
            options,
            list,
        ):
            raise SingleModalityVQAInferenceError(
                "Options must be a list of four strings."
            )

        if len(options) != 4:
            raise SingleModalityVQAInferenceError(
                "MCQ requests require exactly four options."
            )

        cleaned = []

        for index, option in enumerate(options):

            if not isinstance(
                option,
                str,
            ):
                raise SingleModalityVQAInferenceError(
                    f"Option {index + 1} must be a string."
                )

            option = option.strip()

            if not option:
                raise SingleModalityVQAInferenceError(
                    f"Option {index + 1} cannot be empty."
                )

            cleaned.append(option)

        return cleaned

    def execute(
        self,
        query: str,
        image: torch.Tensor,
        modality: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:

        if not isinstance(
            query,
            str,
        ):
            raise SingleModalityVQAInferenceError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise SingleModalityVQAInferenceError(
                "Query cannot be empty."
            )

        modality = self._validate_modality(
            modality
        )

        options = self._validate_options(
            options
        )

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

            raise SingleModalityVQAInferenceError(
                f"Single-modality feature extraction failed: {exc}"
            ) from exc

        if tuple(
            visual_features.shape
        ) != (
            1,
            196,
            768,
        ):
            raise SingleModalityVQAInferenceError(
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

            raise SingleModalityVQAInferenceError(
                f"Text embedding failed: {exc}"
            ) from exc

        if tuple(
            text_embedding.shape
        ) != (
            1,
            768,
        ):
            raise SingleModalityVQAInferenceError(
                "Unexpected text embedding shape: "
                f"{tuple(text_embedding.shape)}"
            )

        # -----------------------------------------------------------
        # MCQ option embeddings
        #
        # The frozen single-modality MCQ specialists expect:
        #
        #     [B,4,768]
        #
        # Each option is embedded independently using the same
        # project text-embedding service.
        # -----------------------------------------------------------

        option_embeddings = None
        option_embedding_metadata = None

        if options is not None:

            embedded_options = []
            option_metadata = []

            for index, option in enumerate(
                options
            ):

                try:

                    option_embedding, metadata = (
                        self.text_service.embed(
                            option
                        )
                    )

                except Exception as exc:

                    raise SingleModalityVQAInferenceError(
                        f"Option {index + 1} embedding failed: {exc}"
                    ) from exc

                if tuple(
                    option_embedding.shape
                ) != (
                    1,
                    768,
                ):
                    raise SingleModalityVQAInferenceError(
                        "Unexpected option embedding shape for "
                        f"option {index + 1}: "
                        f"{tuple(option_embedding.shape)}"
                    )

                embedded_options.append(
                    option_embedding
                )

                option_metadata.append(
                    metadata
                )

            option_embeddings = torch.cat(
                embedded_options,
                dim=0,
            ).unsqueeze(0)

            if tuple(
                option_embeddings.shape
            ) != (
                1,
                4,
                768,
            ):
                raise SingleModalityVQAInferenceError(
                    "Unexpected MCQ option embedding shape: "
                    f"{tuple(option_embeddings.shape)}"
                )

            option_embedding_metadata = {
                "shape": list(
                    option_embeddings.shape
                ),
                "count": 4,
                "embedding_dtype": str(
                    option_embeddings.dtype
                ),
                "options": options,
                "individual_embeddings": (
                    option_metadata
                ),
            }

        input_modalities = [
            modality
        ]

        agent_result = self.agent.execute(
            query=query,
            input_modalities=input_modalities,
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
            options=option_embeddings,
        )

        if not isinstance(
            agent_result,
            dict,
        ):
            raise SingleModalityVQAInferenceError(
                "Agent returned an invalid result."
            )

        if agent_result.get("answer_status") != "success":
            raise SingleModalityVQAInferenceError(
                "Agent execution did not succeed: "
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
            raise SingleModalityVQAInferenceError(
                "Agent returned invalid analysis."
            )

        if analysis.get("task") != (
            "single_image_vqa"
        ):
            raise SingleModalityVQAInferenceError(
                "Single-modality VQA request was routed to "
                f"unexpected task: {analysis.get('task')}"
            )

        trace = agent_result.get(
            "execution_trace",
            {},
        )

        if not isinstance(
            trace,
            dict,
        ):
            raise SingleModalityVQAInferenceError(
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
            },
            "feature_extraction": feature_metadata,
            "text_embedding": text_metadata,
            "analysis": analysis,
            "execution_trace": trace,
            "response": agent_result,
        }


__all__ = [
    "SatQueryAISingleModalityVQAInferenceService",
    "SingleModalityVQAInferenceError",
]