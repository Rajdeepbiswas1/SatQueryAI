from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from agent.bootstrap import create_runtime_agent
from agent.response_formatter import format_agent_response

from backend.single_modality_feature_service import (
    SingleModalityFeatureService,
)

from backend.single_modality_upload_service import (
    SingleModalityUploadService,
)

from backend.text_embedding_service import (
    SatQueryAITextEmbeddingService,
)


class CrossModalUploadInferenceError(
    RuntimeError
):
    """Raised when cross-modal upload inference fails."""


class SatQueryAICrossModalUploadService:
    """
    Application-facing optical + SAR upload service.

    Supported input:
        exactly two images

    Required modalities:
        one S1 image
        one S2 image

    Each input may be:
        - a combined TIFF with the trained channel count
        - a complete BigEarthNet-style scene directory

    S1:
        VH + VV
        -> [1, 2, H, W]

    S2:
        B04, B03, B02, B05, B06, B07, B08, B11, B12
        -> [1, 9, H, W]

    Frozen feature representation:
        S1 -> [1, 196, 768]
        S2 -> [1, 196, 768]

    The existing agent then performs:
        cross-modal VQA
        or
        cross-modal grounding
    depending on the query.
    """

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

        self.agent, _, _ = (
            create_runtime_agent()
        )

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_modality(
        path: Path,
    ) -> tuple[torch.Tensor, str, dict[str, Any]]:

        try:

            tensor, modality, metadata = (
                SingleModalityUploadService._load_scene(
                    path,
                    None,
                )
            )

        except Exception as exc:

            raise CrossModalUploadInferenceError(
                f"Could not load cross-modal input "
                f"{path}: {exc}"
            ) from exc

        if modality not in {
            "s1",
            "s2",
        }:
            raise CrossModalUploadInferenceError(
                f"Unsupported detected modality "
                f"'{modality}' for {path}"
            )

        return (
            tensor,
            modality,
            metadata,
        )

    # ------------------------------------------------------------------
    # MCQ validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_options(
        options: Optional[Sequence[str]],
    ) -> Optional[list[str]]:

        if options is None:
            return None

        values = [
            str(value).strip()
            for value in options
        ]

        if len(values) != 4:
            raise CrossModalUploadInferenceError(
                "MCQ requests require exactly four options."
            )

        if any(
            not value
            for value in values
        ):
            raise CrossModalUploadInferenceError(
                "MCQ options cannot be empty."
            )

        return values

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        paths: Sequence[str | Path],
        options: Optional[Sequence[str]] = None,
        patch_id: str = "uploaded_optical_sar",
    ) -> dict[str, Any]:

        if not isinstance(
            query,
            str,
        ):
            raise CrossModalUploadInferenceError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise CrossModalUploadInferenceError(
                "Query cannot be empty."
            )

        paths = [
            Path(path)
            for path in paths
        ]

        if len(paths) != 2:
            raise CrossModalUploadInferenceError(
                "Cross-modal optical+SAR analysis requires "
                "exactly 2 inputs: one S1 and one S2."
            )

        validated_options = (
            self._validate_options(
                options
            )
        )

        # --------------------------------------------------------------
        # Load both inputs and detect modalities.
        # --------------------------------------------------------------

        loaded = []

        for path in paths:

            tensor, modality, metadata = (
                self._load_modality(
                    path
                )
            )

            loaded.append(
                (
                    path,
                    tensor,
                    modality,
                    metadata,
                )
            )

        s1_entries = [
            entry
            for entry in loaded
            if entry[2] == "s1"
        ]

        s2_entries = [
            entry
            for entry in loaded
            if entry[2] == "s2"
        ]

        if len(s1_entries) != 1:
            raise CrossModalUploadInferenceError(
                "Cross-modal input must contain exactly "
                "one S1 input."
            )

        if len(s2_entries) != 1:
            raise CrossModalUploadInferenceError(
                "Cross-modal input must contain exactly "
                "one S2 input."
            )

        (
            s1_path,
            s1_image,
            _,
            s1_metadata,
        ) = s1_entries[0]

        (
            s2_path,
            s2_image,
            _,
            s2_metadata,
        ) = s2_entries[0]

        # --------------------------------------------------------------
        # Verify spatial compatibility before feature extraction.
        # --------------------------------------------------------------

        s1_hw = tuple(
            s1_image.shape[-2:]
        )

        s2_hw = tuple(
            s2_image.shape[-2:]
        )

        if s1_hw != s2_hw:
            raise CrossModalUploadInferenceError(
                "S1 and S2 spatial dimensions do not match: "
                f"S1={s1_hw}, S2={s2_hw}. "
                "Cross-modal inputs must represent compatible "
                "co-registered spatial coverage."
            )

        # --------------------------------------------------------------
        # Frozen DOFA feature extraction.
        # --------------------------------------------------------------

        try:

            s1_features, s1_feature_metadata = (
                self.feature_service.extract(
                    image=s1_image,
                    modality="s1",
                )
            )

            s2_features, s2_feature_metadata = (
                self.feature_service.extract(
                    image=s2_image,
                    modality="s2",
                )
            )

        except Exception as exc:

            raise CrossModalUploadInferenceError(
                f"Cross-modal feature extraction failed: {exc}"
            ) from exc

        if tuple(
            s1_features.shape
        ) != self.EXPECTED_FEATURE_SHAPE:

            raise CrossModalUploadInferenceError(
                "Unexpected S1 feature shape: "
                f"{tuple(s1_features.shape)}"
            )

        if tuple(
            s2_features.shape
        ) != self.EXPECTED_FEATURE_SHAPE:

            raise CrossModalUploadInferenceError(
                "Unexpected S2 feature shape: "
                f"{tuple(s2_features.shape)}"
            )

        # --------------------------------------------------------------
        # Query embedding.
        # --------------------------------------------------------------

        try:

            text_embedding, text_metadata = (
                self.text_service.embed(
                    query
                )
            )

        except Exception as exc:

            raise CrossModalUploadInferenceError(
                f"Text embedding failed: {exc}"
            ) from exc

        if tuple(
            text_embedding.shape
        ) != self.EXPECTED_TEXT_SHAPE:

            raise CrossModalUploadInferenceError(
                "Unexpected text embedding shape: "
                f"{tuple(text_embedding.shape)}"
            )

        # --------------------------------------------------------------
        # MCQ option embeddings.
        # --------------------------------------------------------------

        option_embedding = None

        if validated_options is not None:

            option_vectors = []

            for option in validated_options:

                vector, _ = (
                    self.text_service.embed(
                        option
                    )
                )

                if tuple(
                    vector.shape
                ) != self.EXPECTED_TEXT_SHAPE:

                    raise CrossModalUploadInferenceError(
                        "Unexpected option embedding shape "
                        f"for '{option}': "
                        f"{tuple(vector.shape)}"
                    )

                option_vectors.append(
                    vector
                )

            option_embedding = torch.cat(
                option_vectors,
                dim=0,
            ).unsqueeze(0)

            if tuple(
                option_embedding.shape
            ) != (
                1,
                4,
                768,
            ):

                raise CrossModalUploadInferenceError(
                    "Unexpected option embedding shape: "
                    f"{tuple(option_embedding.shape)}"
                )

        # --------------------------------------------------------------
        # Existing agent.
        #
        # Supplying both S1 and S2 explicitly is intentional:
        # the existing analyzer uses this as the paired cross-modal
        # input condition.
        # --------------------------------------------------------------

        agent_kwargs = {
            "query": query,
            "input_modalities": [
                "s1",
                "s2",
            ],
            "s1": s1_features,
            "s2": s2_features,
            "text_embedding": text_embedding,
            "patch_id": patch_id,
        }

        if option_embedding is not None:
            agent_kwargs["options"] = (
                option_embedding
            )

        try:

            agent_result = (
                self.agent.execute(
                    **agent_kwargs
                )
            )

        except Exception as exc:

            raise CrossModalUploadInferenceError(
                f"Cross-modal agent execution failed: {exc}"
            ) from exc

        if not isinstance(
            agent_result,
            dict,
        ):
            raise CrossModalUploadInferenceError(
                "Agent returned an invalid result."
            )

        if agent_result.get(
            "answer_status"
        ) != "success":

            raise CrossModalUploadInferenceError(
                "Cross-modal agent execution did not "
                f"complete successfully: {agent_result}"
            )

        analysis = agent_result.get(
            "analysis",
            {},
        )

        if not isinstance(
            analysis,
            dict,
        ):
            raise CrossModalUploadInferenceError(
                "Agent returned invalid analysis."
            )

        if analysis.get(
            "task"
        ) not in {
            "cross_modal",
            "grounding",
        }:

            raise CrossModalUploadInferenceError(
                "Two-modality request was routed to "
                f"unexpected task: {analysis.get('task')}"
            )

        # --------------------------------------------------------------
        # Standard application response.
        # --------------------------------------------------------------

        formatted_response = (
            format_agent_response(
                agent_result
            )
        )

        return {
            "status": "success",
            "query": query,

            "input": {
                "patch_id": patch_id,
                "s1": {
                    "path": str(s1_path),
                    **s1_metadata,
                },
                "s2": {
                    "path": str(s2_path),
                    **s2_metadata,
                },
            },

            "feature_extraction": {
                "s1": {
                    "shape": list(
                        s1_features.shape
                    ),
                    "metadata": (
                        s1_feature_metadata
                    ),
                },
                "s2": {
                    "shape": list(
                        s2_features.shape
                    ),
                    "metadata": (
                        s2_feature_metadata
                    ),
                },
            },

            "text_embedding": text_metadata,

            "option_embedding": (
                {
                    "shape": list(
                        option_embedding.shape
                    ),
                    "count": 4,
                }
                if option_embedding is not None
                else None
            ),

            "analysis": analysis,

            "execution_trace": (
                agent_result[
                    "execution_trace"
                ]
            ),

            "agent_result": agent_result,

            "response": formatted_response,
        }


__all__ = [
    "SatQueryAICrossModalUploadService",
    "CrossModalUploadInferenceError",
]