from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from agent.bootstrap import create_runtime_agent
from agent.response_formatter import format_agent_response

from backend.upload_service import (
    preprocess_upload_bundle,
)

from backend.feature_service import (
    DofaFeatureService,
)

from backend.text_embedding_service import (
    SatQueryAITextEmbeddingService,
)


class AgentInferenceError(Exception):
    """Raised when application-level agent inference fails."""


class SatQueryAIAgentInferenceService:
    """
    Full application inference pipeline.

    Raw inputs:
        14 S1/S2 TIFF files
        natural-language query

    Pipeline:
        TIFFs
            ->
        preprocessing
            ->
        [14,120,120]
            ->
        frozen DOFA
            ->
        S1/S2 feature tokens
            ->
        frozen DistilBERT query embedding
            ->
        existing SatQueryAI agent
            ->
        specialist
            ->
        formatted response
    """

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:

        self.device = device

        # ---------------------------------------------------------
        # These are application-level services around frozen models.
        # ---------------------------------------------------------

        self.feature_service = DofaFeatureService(
            device=device
        )

        self.text_service = (
            SatQueryAITextEmbeddingService(
                device=device
            )
        )

        # ---------------------------------------------------------
        # Existing real runtime agent.
        # ---------------------------------------------------------

        runtime = create_runtime_agent()

        if not runtime:
            raise AgentInferenceError(
                "Runtime agent initialization returned no objects."
            )

        self.agent = runtime[0]

        if self.agent is None:
            raise AgentInferenceError(
                "Runtime agent initialization returned None."
            )

    def execute(
        self,
        query: str,
        paths: Iterable[Path],
        patch_id: str = "uploaded_pair",
        options: Sequence[str] | None = None,
        point=None,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete SatQueryAI pipeline on one uploaded pair.
        """

        if not isinstance(query, str):
            raise AgentInferenceError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise AgentInferenceError(
                "Query cannot be empty."
            )

        paths = [
            Path(path)
            for path in paths
        ]

        # =========================================================
        # Stage 1 — Raw image preprocessing
        # =========================================================

        image_tensor, preprocessing_metadata, file_map = (
            preprocess_upload_bundle(
                paths=paths,
                patch_id=patch_id,
            )
        )

        # =========================================================
        # Stage 2 — Frozen DOFA feature extraction
        # =========================================================

        s1_features, s2_features, feature_metadata = (
            self.feature_service.extract(
                image_tensor
            )
        )

        # =========================================================
        # Stage 3 — Frozen DistilBERT query embedding
        # =========================================================

        text_embedding, text_metadata = (
            self.text_service.embed(
                query
            )
        )

        # =========================================================
        # Stage 4 — Agent orchestration
        # =========================================================
        #
        # Both Sentinel-1 and Sentinel-2 representations are
        # explicitly available. This allows the existing input-aware
        # cross-modal routing logic to distinguish the paired input.
        #
        # Grounding/temporal query signals retain their priority
        # inside the existing agent.
        # =========================================================

        try:

            agent_result = self.agent.execute(
                query=query,
                input_modalities=[
                    "s1",
                    "s2",
                ],
                s1=s1_features,
                s2=s2_features,
                text_embedding=text_embedding,
                options=options,
                point=point,
                reference=reference,
                patch_id=patch_id,
            )

        except Exception as exc:
            raise AgentInferenceError(
                f"Agent execution failed: {exc}"
            ) from exc

        # =========================================================
        # Stage 5 — Standard response formatting
        # =========================================================

        try:

            formatted = format_agent_response(
                agent_result
            )

        except Exception as exc:
            raise AgentInferenceError(
                f"Response formatting failed: {exc}"
            ) from exc

        # =========================================================
        # Application-level audit metadata
        # =========================================================

        return {
            "status": "success",

            "query": query,

            "input": {
                "patch_id": patch_id,
                "file_count": len(file_map),
                "channels": list(file_map.keys()),
                "preprocessing": preprocessing_metadata,
            },

            "feature_extraction": feature_metadata,

            "text_embedding": text_metadata,

            "agent_result": agent_result,

            "response": formatted,
        }


# ============================================================================
# Integration test
# ============================================================================

if __name__ == "__main__":

    import pandas as pd

    from preprocessing.preprocess_dataset import (
        build_s1_paths,
        build_s2_paths,
    )

    print(
        "SATQUERYAI — FULL AGENT INFERENCE SERVICE TEST"
    )
    print("=" * 80)
    print()

    # -------------------------------------------------------------
    # Use one real binary-VQA question and its corresponding
    # S1/S2 image pair.
    #
    # Exactly ONE pair is processed.
    # -------------------------------------------------------------

    binary_annotations = pd.read_parquet(
        r"data\processed\binary_vqa_split.parquet"
    )

    binary_annotations = binary_annotations[
        binary_annotations["type"] == "binary"
    ].reset_index(drop=True)

    if len(binary_annotations) == 0:
        raise RuntimeError(
            "No binary VQA rows were found."
        )

    row = binary_annotations.iloc[0]

    query = str(
        row["input"]
    )

    patch_id = str(
        row["patch_id"]
    )

    metadata = pd.read_parquet(
        r"data\raw\BigEarthNet_LMDB\metadata.parquet",
        columns=[
            "patch_id",
            "s1_name",
        ],
    )

    matched = metadata[
        metadata["patch_id"] == patch_id
    ]

    if len(matched) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve S1 metadata "
            f"for patch_id={patch_id}. "
            f"Matches={len(matched)}"
        )

    s1_name = str(
        matched.iloc[0]["s1_name"]
    )

    s1_paths = build_s1_paths(
        s1_name
    )

    s2_paths = build_s2_paths(
        patch_id
    )

    all_paths = (
        list(s1_paths.values())
        + list(s2_paths.values())
    )

    print(
        f"Patch ID: {patch_id}"
    )

    print(
        f"S1 name: {s1_name}"
    )

    print(
        f"Question: {query}"
    )

    print(
        f"Input files: {len(all_paths)}"
    )

    print()
    print(
        "Initializing full application inference stack..."
    )

    service = SatQueryAIAgentInferenceService(
        device="cpu"
    )

    print()
    print(
        "Executing full pipeline..."
    )

    result = service.execute(
        query=query,
        paths=all_paths,
        patch_id=patch_id,
    )

    print()
    print(
        "PIPELINE RESULT"
    )
    print("-" * 80)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Agent answer status:",
        result["agent_result"].get(
            "answer_status"
        )
    )

    analysis = result[
        "agent_result"
    ].get(
        "analysis",
        {}
    )

    print(
        "Detected task:",
        analysis.get("task")
    )

    print(
        "Grounding mode:",
        analysis.get("grounding_mode")
    )

    trace = result[
        "agent_result"
    ].get(
        "execution_trace",
        {}
    )

    print(
        "Selected tool:",
        trace.get("selected_tool")
    )

    print(
        "Capability available:",
        trace.get("capability_available")
    )

    print(
        "Routing confidence:",
        trace.get("routing_confidence")
    )

    evidence = result[
        "agent_result"
    ].get(
        "evidence"
    )

    if isinstance(
        evidence,
        dict,
    ):

        print(
            "Specialist selected:",
            evidence.get(
                "specialist_selected"
            )
        )

        print(
            "Answer:",
            evidence.get(
                "answer"
            )
        )

        print(
            "Answer confidence:",
            evidence.get(
                "answer_confidence"
            )
        )

    print()
    print(
        "INPUT CONTRACT"
    )

    print(
        "  preprocessing shape:",
        result[
            "input"
        ][
            "preprocessing"
        ][
            "shape"
        ]
    )

    print(
        "  S1 features:",
        result[
            "feature_extraction"
        ][
            "s1_shape"
        ]
    )

    print(
        "  S2 features:",
        result[
            "feature_extraction"
        ][
            "s2_shape"
        ]
    )

    print(
        "  text embedding:",
        result[
            "text_embedding"
        ][
            "shape"
        ]
    )

    print()
    print(
        "FULL AGENT INFERENCE SERVICE TEST: PASS"
    )