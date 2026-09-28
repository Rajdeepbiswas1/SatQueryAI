from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from agent.bootstrap import create_runtime_agent
from agent.response_formatter import format_agent_response
from agent.temporal_upload_adapter import TemporalUploadAdapter

from backend.text_embedding_service import (
    SatQueryAITextEmbeddingService,
)


class TemporalInferenceError(Exception):
    """Raised when temporal application inference fails."""


class SatQueryAITemporalInferenceService:
    """
    Application-level temporal inference pipeline.

    Raw inputs:
        exactly two temporal RGB images
        natural-language query

    Pipeline:
        T1/T2 images
            ->
        TemporalUploadAdapter
            ->
        T1/T2/diff [1,196,768]
            ->
        DistilBERT query embedding [1,768]
            ->
        existing SatQueryAI agent
            ->
        Temporal V3 specialist
            ->
        standardized response
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = device

        # ---------------------------------------------------------
        # Frozen temporal raw-image adapter.
        # ---------------------------------------------------------

        self.temporal_adapter = TemporalUploadAdapter(
            device=device
        )

        # ---------------------------------------------------------
        # Existing frozen text embedding service.
        # ---------------------------------------------------------

        self.text_service = (
            SatQueryAITextEmbeddingService(
                device=device
            )
        )

        # ---------------------------------------------------------
        # Existing runtime agent.
        # ---------------------------------------------------------

        runtime = create_runtime_agent()

        if not runtime:
            raise TemporalInferenceError(
                "Runtime agent initialization returned no objects."
            )

        self.agent = runtime[0]

        if self.agent is None:
            raise TemporalInferenceError(
                "Runtime agent initialization returned None."
            )

    def execute(
        self,
        query: str,
        paths: Iterable[Path],
    ) -> dict[str, Any]:
        """
        Execute temporal inference on exactly two raw images.

        paths[0] = BEFORE / T1
        paths[1] = AFTER / T2
        """

        if not isinstance(query, str):
            raise TemporalInferenceError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise TemporalInferenceError(
                "Query cannot be empty."
            )

        paths = [
            Path(path)
            for path in paths
        ]

        if len(paths) != 2:
            raise TemporalInferenceError(
                "Temporal inference requires exactly two image files."
            )

        t1_path = paths[0]
        t2_path = paths[1]

        # =========================================================
        # Stage 1 — Raw temporal image processing
        # =========================================================

        try:

            temporal_result = (
                self.temporal_adapter.transform_pair(
                    t1_path=t1_path,
                    t2_path=t2_path,
                )
            )

        except Exception as exc:

            raise TemporalInferenceError(
                f"Temporal image processing failed: {exc}"
            ) from exc

        t1_features = temporal_result["t1"]
        t2_features = temporal_result["t2"]
        diff_features = temporal_result["diff"]

        temporal_metadata = temporal_result[
            "metadata"
        ]

        # =========================================================
        # Stage 2 — Query embedding
        # =========================================================

        try:

            text_embedding, text_metadata = (
                self.text_service.embed(
                    query
                )
            )

        except Exception as exc:

            raise TemporalInferenceError(
                f"Temporal query embedding failed: {exc}"
            ) from exc

        # =========================================================
        # Stage 3 — Existing agent orchestration
        # =========================================================

        try:

            agent_result = self.agent.execute(
                query=query,
                input_modalities=[
                    "temporal",
                ],
                t1=t1_features,
                t2=t2_features,
                diff=diff_features,
                text_embedding=text_embedding,
            )

        except Exception as exc:

            raise TemporalInferenceError(
                f"Temporal agent execution failed: {exc}"
            ) from exc

        # =========================================================
        # Stage 4 — Standard response formatting
        # =========================================================

        try:

            formatted = format_agent_response(
                agent_result
            )

        except Exception as exc:

            raise TemporalInferenceError(
                f"Temporal response formatting failed: {exc}"
            ) from exc

        # =========================================================
        # Application-level response
        # =========================================================

        return {
            "status": "success",

            "query": query,

            "input": {
                "file_count": 2,
                "before_path": str(t1_path),
                "after_path": str(t2_path),
                "temporal_processing": temporal_metadata,
            },

            "feature_extraction": {
                "t1_shape": list(
                    t1_features.shape
                ),
                "t2_shape": list(
                    t2_features.shape
                ),
                "diff_shape": list(
                    diff_features.shape
                ),
                "dtype": str(
                    t1_features.dtype
                ),
                "backbone": temporal_metadata.get(
                    "backbone"
                ),
                "weights": temporal_metadata.get(
                    "weights"
                ),
                "parameter_count": temporal_metadata.get(
                    "parameter_count"
                ),
            },

            "text_embedding": text_metadata,

            "agent_result": agent_result,

            "response": formatted,
        }


# =====================================================================
# Direct initialization smoke test
# =====================================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "SATQUERYAI — TEMPORAL INFERENCE SERVICE"
    )
    print("=" * 80)
    print()

    service = SatQueryAITemporalInferenceService(
        device="cpu"
    )

    print(
        "Temporal adapter:",
        type(
            service.temporal_adapter
        ).__name__,
    )

    print(
        "Agent:",
        type(
            service.agent
        ).__name__,
    )

    print(
        "Temporal DOFA parameters:",
        service.temporal_adapter.metadata[
            "parameter_count"
        ],
    )

    print()
    print(
        "TEMPORAL INFERENCE SERVICE INITIALIZATION: PASS"
    )