# =============================================================================
# SATQUERYAI — APPLICATION BOOTSTRAP
# =============================================================================
#
# Creates the real SatQueryAI agent and registers the currently verified
# specialist capabilities.
#
# This file performs initialization only.
#
# It does NOT:
#   - train models
#   - modify checkpoints
#   - modify specialist architectures
#   - contain dataset/training logic
#
# Final top-level capabilities:
#   grounding
#   single_image_vqa
#   temporal_change
#   cross_modal
# =============================================================================

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from agent.core import (
    QueryAnalysis,
    SatQueryAIAgent,
    create_satqueryai_agent,
)

from src.models.grounding_recovered.point_grounding_recovered import (
    PointGroundingModel,
)

from src.models.grounding_recovered.c3_reference_grounding_recovered import (
    C3QueryConditionedGrounding,
)

from src.models.grounding_recovered.unified_grounding_recovered import (
    UnifiedGroundingModel,
    UnifiedGroundingInference,
    SatQueryAIGroundingRouter,
)

from agent.binary_vqa_tool import (
    SatQueryAIBinaryVQATool,
)

from agent.mcq_vqa_tool import (
    SatQueryAIMCQVQATool,
)

from agent.single_modality_vqa_tools import (
    SatQueryAISingleModalityBinaryVQATool,
    SatQueryAISingleModalityMCQVQATool,
)

from agent.single_modality_grounding_tool import (
    SatQueryAISingleModalityGroundingTool,
)

from agent.temporal_change_vqa_tool import (
    SatQueryAITemporalChangeV3Tool,
)

from agent.cross_modal_vqa_tool import (
    SatQueryAICrossModalBinaryVQATool,
    SatQueryAICrossModalMCQVQATool,
)

from agent.cross_modal_vqa_router import (
    SatQueryAICrossModalVQARouter,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

POINT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "point_grounding"
    / "best.pt"
)

C3_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "c3_v1"
    / "best.pt"
)


# =============================================================================
# FROZEN MODEL INVARIANTS
# =============================================================================

EXPECTED_POINT_PARAMS = 1_531_909
EXPECTED_C3_PARAMS = 2_186_753
EXPECTED_UNIFIED_PARAMS = 3_718_662


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def _load_state_dict(
    checkpoint_path: Path,
) -> Tuple[dict, dict]:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        # Some PyTorch checkpoints are themselves state dictionaries.
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Invalid state_dict in checkpoint: {checkpoint_path}"
        )

    return checkpoint, state_dict


def _parameter_count(
    model: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =============================================================================
# SINGLE / PAIRED GROUNDING APPLICATION DISPATCHER
# =============================================================================

class SatQueryAIGroundingTool:
    """
    Application-level grounding dispatcher.

    Routing:

        one visual modality
            ->
        true single-modality S1/S2 grounding specialist

        both S1 + S2
            ->
        existing UnifiedGroundingRouter

    The specialist models themselves remain independent and frozen.
    """

    def __init__(
        self,
        router: SatQueryAIGroundingRouter,
        single_modality_tool: SatQueryAISingleModalityGroundingTool,
    ) -> None:

        if not isinstance(
            router,
            SatQueryAIGroundingRouter,
        ):
            raise TypeError(
                "router must be a "
                "SatQueryAIGroundingRouter."
            )

        if not isinstance(
            single_modality_tool,
            SatQueryAISingleModalityGroundingTool,
        ):
            raise TypeError(
                "single_modality_tool must be a "
                "SatQueryAISingleModalityGroundingTool."
            )

        self.router = router
        self.single_modality_tool = single_modality_tool

    def __call__(
        self,
        query: str,
        analysis: QueryAnalysis,
        s1=None,
        s2=None,
        text_embedding=None,
        point=None,
        reference: Optional[str] = None,
        patch_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Translate the generic Agent contract into either:

            single-modality grounding
        or
            paired Unified grounding.
        """

        del kwargs

        if analysis.task != "grounding":
            raise ValueError(
                "SatQueryAIGroundingTool received a "
                "non-grounding request."
            )

        mode = analysis.grounding_mode

        if mode not in {
            "point",
            "reference",
        }:
            raise ValueError(
                f"Invalid grounding mode: {mode}"
            )

        s1_present = s1 is not None
        s2_present = s2 is not None

        if not s1_present and not s2_present:
            raise ValueError(
                "Grounding requires at least one visual modality."
            )

        # ---------------------------------------------------------------------
        # Paired cross-modal grounding
        # ---------------------------------------------------------------------

        if s1_present and s2_present:

            return self.router.dispatch(
                mode=mode,
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
                point=point,
                query=query,
                reference=reference,
                patch_id=patch_id,
            )

        # ---------------------------------------------------------------------
        # True single-modality grounding
        # ---------------------------------------------------------------------

        return self.single_modality_tool(
            query=query,
            analysis=analysis,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
            reference=reference,
            patch_id=patch_id,
        )


# =============================================================================
# SINGLE-IMAGE VQA APPLICATION DISPATCHER
# =============================================================================

class SatQueryAISingleImageVQATool:
    """
    Application-level dispatcher for true single-image VQA.

    No S1/S2 fusion occurs here.

    Routing:

        exactly one modality + no options
            ->
        single-modality Binary VQA

        exactly one modality + four options
            ->
        single-modality option-aware MCQ VQA
    """

    def __init__(
        self,
        binary_tool: SatQueryAISingleModalityBinaryVQATool,
        mcq_tool: SatQueryAISingleModalityMCQVQATool,
    ) -> None:

        if not isinstance(
            binary_tool,
            SatQueryAISingleModalityBinaryVQATool,
        ):
            raise TypeError(
                "binary_tool must be a "
                "SatQueryAISingleModalityBinaryVQATool."
            )

        if not isinstance(
            mcq_tool,
            SatQueryAISingleModalityMCQVQATool,
        ):
            raise TypeError(
                "mcq_tool must be a "
                "SatQueryAISingleModalityMCQVQATool."
            )

        self.binary_tool = binary_tool
        self.mcq_tool = mcq_tool

    def __call__(
        self,
        query: str,
        analysis: QueryAnalysis,
        s1=None,
        s2=None,
        text_embedding=None,
        options=None,
        **kwargs: Any,
    ):
        """
        Dispatch to Binary or MCQ single-modality VQA.
        """

        del kwargs

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "SatQueryAISingleImageVQATool received a "
                "non-single-image-VQA request."
            )

        s1_present = s1 is not None
        s2_present = s2 is not None

        if s1_present == s2_present:
            if s1_present:
                raise ValueError(
                    "Single-image VQA requires exactly one visual modality;"
                    " received both s1 and s2."
                )

            raise ValueError(
                "Single-image VQA requires exactly one visual modality:"
                " s1 or s2."
            )

        if options is None:

            return self.binary_tool(
                query=query,
                analysis=analysis,
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
            )

        return self.mcq_tool(
            query=query,
            analysis=analysis,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            options=options,
        )

    dispatch = __call__


# =============================================================================
# LOAD VERIFIED PAIRED GROUNDING SYSTEM
# =============================================================================

def build_grounding_router() -> SatQueryAIGroundingRouter:
    """
    Load the exact verified paired grounding specialists and construct the
    UnifiedGroundingRouter.
    """

    # -------------------------------------------------------------------------
    # Point specialist
    # -------------------------------------------------------------------------

    point_checkpoint, point_state = _load_state_dict(
        POINT_CHECKPOINT
    )

    point_model = PointGroundingModel()

    point_model.load_state_dict(
        point_state,
        strict=True,
    )

    point_params = _parameter_count(
        point_model
    )

    if point_params != EXPECTED_POINT_PARAMS:
        raise RuntimeError(
            "Point parameter-count mismatch: "
            f"expected {EXPECTED_POINT_PARAMS}, "
            f"got {point_params}"
        )

    point_model = point_model.to("cpu")
    point_model.eval()

    for parameter in point_model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------------------
    # Reference specialist
    # -------------------------------------------------------------------------

    c3_checkpoint, c3_state = _load_state_dict(
        C3_CHECKPOINT
    )

    c3_model = C3QueryConditionedGrounding()

    c3_model.load_state_dict(
        c3_state,
        strict=True,
    )

    c3_params = _parameter_count(
        c3_model
    )

    if c3_params != EXPECTED_C3_PARAMS:
        raise RuntimeError(
            "C3 parameter-count mismatch: "
            f"expected {EXPECTED_C3_PARAMS}, "
            f"got {c3_params}"
        )

    c3_model = c3_model.to("cpu")
    c3_model.eval()

    for parameter in c3_model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------------------
    # Unified grounding model
    # -------------------------------------------------------------------------

    unified_model = UnifiedGroundingModel(
        point_model=point_model,
        reference_model=c3_model,
    )

    unified_params = _parameter_count(
        unified_model
    )

    if unified_params != EXPECTED_UNIFIED_PARAMS:
        raise RuntimeError(
            "Unified parameter-count mismatch: "
            f"expected {EXPECTED_UNIFIED_PARAMS}, "
            f"got {unified_params}"
        )

    unified_model = unified_model.to("cpu")
    unified_model.eval()

    for parameter in unified_model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------------------
    # Unified inference + specialist router
    # -------------------------------------------------------------------------

    grounding_inference = UnifiedGroundingInference(
        unified_model
    )

    grounding_router = SatQueryAIGroundingRouter(
        grounding_inference
    )

    # Keep references alive and make initialization auditable.
    grounding_router.point_checkpoint_epoch = (
        point_checkpoint.get("epoch")
    )

    grounding_router.c3_checkpoint_epoch = (
        c3_checkpoint.get("epoch")
    )

    grounding_router.parameter_count = (
        unified_params
    )

    return grounding_router


# =============================================================================
# BUILD COMPLETE AGENT
# =============================================================================

def create_runtime_agent() -> Tuple[
    SatQueryAIAgent,
    SatQueryAIGroundingTool,
    SatQueryAIBinaryVQATool,
]:
    """
    Construct the verified SatQueryAI runtime.

    Final top-level capabilities:

        grounding
        single_image_vqa
        temporal_change
        cross_modal

    Routing:

        single_image_vqa
            ->
        true S1/S2 single-modality VQA

        cross_modal
            ->
        existing paired VQA specialists

        grounding
            ->
        single S1/S2 grounding
        OR
        paired Unified grounding

        temporal_change
            ->
        Temporal Change V3
    """

    agent = create_satqueryai_agent()

    # -------------------------------------------------------------------------
    # Existing paired / cross-modal grounding
    # -------------------------------------------------------------------------

    grounding_router = build_grounding_router()

    single_modality_grounding_tool = (
        SatQueryAISingleModalityGroundingTool()
    )

    grounding_tool = SatQueryAIGroundingTool(
        router=grounding_router,
        single_modality_tool=single_modality_grounding_tool,
    )

    # -------------------------------------------------------------------------
    # Existing paired VQA specialists
    #
    # These remain available for cross_modal and are NOT used by the
    # single_image_vqa capability.
    # -------------------------------------------------------------------------

    binary_vqa_tool = SatQueryAIBinaryVQATool()

    mcq_vqa_tool = SatQueryAIMCQVQATool()

    # -------------------------------------------------------------------------
    # True single-modality VQA specialists
    # -------------------------------------------------------------------------

    single_binary_vqa_tool = (
        SatQueryAISingleModalityBinaryVQATool()
    )

    single_mcq_vqa_tool = (
        SatQueryAISingleModalityMCQVQATool()
    )

    single_image_vqa_tool = SatQueryAISingleImageVQATool(
        binary_tool=single_binary_vqa_tool,
        mcq_tool=single_mcq_vqa_tool,
    )

    # -------------------------------------------------------------------------
    # Grounding
    # -------------------------------------------------------------------------

    agent.registry.register(
        capability="grounding",
        tool_name="GroundingModalityDispatcher",
        handler=grounding_tool,
    )

    # -------------------------------------------------------------------------
    # True single-image VQA
    # -------------------------------------------------------------------------

    agent.registry.register(
        capability="single_image_vqa",
        tool_name="SingleModalityVQATool",
        handler=single_image_vqa_tool,
    )

    # -------------------------------------------------------------------------
    # Temporal Change V3
    # -------------------------------------------------------------------------

    temporal_v3_tool = (
        SatQueryAITemporalChangeV3Tool()
    )

    agent.registry.register(
        capability="temporal_change",
        tool_name="TemporalChangeV3",
        handler=temporal_v3_tool,
    )

    # -------------------------------------------------------------------------
    # Cross-Modal Optical + SAR VQA
    # -------------------------------------------------------------------------

    cross_modal_binary_tool = (
        SatQueryAICrossModalBinaryVQATool()
    )

    cross_modal_mcq_tool = (
        SatQueryAICrossModalMCQVQATool()
    )

    cross_modal_vqa_router = (
        SatQueryAICrossModalVQARouter(
            binary_tool=cross_modal_binary_tool,
            mcq_tool=cross_modal_mcq_tool,
        )
    )

    agent.registry.register(
        capability="cross_modal",
        tool_name="CrossModalVQARouter",
        handler=cross_modal_vqa_router,
    )

    return (
        agent,
        grounding_tool,
        binary_vqa_tool,
    )


# =============================================================================
# DIRECT INITIALIZATION TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("SATQUERYAI — RUNTIME AGENT BOOTSTRAP TEST")
    print("=" * 80)

    (
        runtime_agent,
        grounding_tool,
        binary_vqa_tool,
    ) = create_runtime_agent()

    print()
    print("Registered capabilities:")
    print(
        " ",
        runtime_agent.registry.capabilities(),
    )

    expected_capabilities = [
        "cross_modal",
        "grounding",
        "single_image_vqa",
        "temporal_change",
    ]

    actual_capabilities = (
        runtime_agent.registry.capabilities()
    )

    if actual_capabilities != expected_capabilities:
        raise RuntimeError(
            "Unexpected top-level capabilities: "
            f"{actual_capabilities}"
        )

    print("Capability registry: PASS")

    if not runtime_agent.registry.available(
        "grounding"
    ):
        raise RuntimeError(
            "Grounding capability was not registered."
        )

    if not runtime_agent.registry.available(
        "single_image_vqa"
    ):
        raise RuntimeError(
            "Single-image VQA capability was not registered."
        )

    if not runtime_agent.registry.available(
        "temporal_change"
    ):
        raise RuntimeError(
            "Temporal capability was not registered."
        )

    if not runtime_agent.registry.available(
        "cross_modal"
    ):
        raise RuntimeError(
            "Cross-modal capability was not registered."
        )

    # -------------------------------------------------------------------------
    # Grounding registration
    # -------------------------------------------------------------------------

    print()
    print("Grounding capability: PASS")

    print(
        "Application grounding dispatcher:",
        type(grounding_tool).__name__,
    )

    print(
        "Unified grounding router:",
        type(grounding_tool.router).__name__,
    )

    print(
        "Unified parameters:",
        grounding_tool.router.parameter_count,
    )

    print(
        "Point checkpoint epoch:",
        grounding_tool.router.point_checkpoint_epoch,
    )

    print(
        "C3 checkpoint epoch:",
        grounding_tool.router.c3_checkpoint_epoch,
    )

    print(
        "Single-modality grounding tool:",
        type(
            grounding_tool.single_modality_tool
        ).__name__,
    )

    print(
        "Single-modality grounding parameters:",
        grounding_tool.single_modality_tool.parameter_count,
    )

    print(
        "Single-modality grounding epochs:",
        grounding_tool.single_modality_tool.checkpoint_epoch,
    )

    # -------------------------------------------------------------------------
    # Single-image VQA registration
    # -------------------------------------------------------------------------

    single_image_entry = (
        runtime_agent.registry.get(
            "single_image_vqa"
        )
    )

    single_image_handler = (
        single_image_entry["handler"]
    )

    print()
    print("Single-image VQA capability: PASS")

    print(
        "VQA dispatcher:",
        type(single_image_handler).__name__,
    )

    print(
        "Single binary specialist:",
        type(
            single_image_handler.binary_tool
        ).__name__,
    )

    print(
        "Single binary parameter count:",
        single_image_handler.binary_tool.parameter_count,
    )

    print(
        "Single binary checkpoint epochs:",
        single_image_handler.binary_tool.checkpoint_epoch,
    )

    print(
        "Single MCQ specialist:",
        type(
            single_image_handler.mcq_tool
        ).__name__,
    )

    print(
        "Single MCQ parameter count:",
        single_image_handler.mcq_tool.parameter_count,
    )

    print(
        "Single MCQ checkpoint epochs:",
        single_image_handler.mcq_tool.checkpoint_epoch,
    )

    # -------------------------------------------------------------------------
    # Cross-modal registration
    # -------------------------------------------------------------------------

    cross_modal_handler = (
        runtime_agent.registry.get(
            "cross_modal"
        )["handler"]
    )

    print()
    print("Cross-modal capability: PASS")

    print(
        "Cross-modal router:",
        type(cross_modal_handler).__name__,
    )

    print(
        "Paired Binary VQA:",
        type(
            cross_modal_handler.binary_tool
        ).__name__,
    )

    print(
        "Paired MCQ VQA:",
        type(
            cross_modal_handler.mcq_tool
        ).__name__,
    )

    # -------------------------------------------------------------------------
    # Temporal registration
    # -------------------------------------------------------------------------

    temporal_handler = (
        runtime_agent.registry.get(
            "temporal_change"
        )["handler"]
    )

    print()
    print("Temporal capability: PASS")

    print(
        "Temporal specialist:",
        type(temporal_handler).__name__,
    )

    # -------------------------------------------------------------------------
    # Analyzer routing-only checks
    # -------------------------------------------------------------------------

    tests = [
        (
            "Where is the largest urban area in the image?",
            "grounding",
            "reference",
            None,
        ),
        (
            "Locate the target at the supplied point.",
            "grounding",
            "point",
            None,
        ),
        (
            "How many buildings are present?",
            "single_image_vqa",
            None,
            ["s1"],
        ),
        (
            "What type of land cover is visible?",
            "single_image_vqa",
            None,
            ["s2"],
        ),
        (
            "What changes occurred between the before and after images?",
            "temporal_change",
            None,
            ["t1", "t2", "diff"],
        ),
    ]

    print()
    print("-" * 80)
    print("AGENT ROUTING TESTS")
    print("-" * 80)

    for (
        query,
        expected_task,
        expected_mode,
        modalities,
    ) in tests:

        analysis = runtime_agent.analyze_query(
            query,
            input_modalities=modalities,
        )

        print()
        print("Query:")
        print(" ", query)

        print(
            "Input modalities:",
            modalities,
        )

        print(
            "Task:",
            analysis.task,
        )

        print(
            "Grounding mode:",
            analysis.grounding_mode,
        )

        if analysis.task != expected_task:
            raise RuntimeError(
                f"Unexpected task for query: {query}"
            )

        if analysis.grounding_mode != expected_mode:
            raise RuntimeError(
                f"Unexpected grounding mode for query: {query}"
            )

        print("Routing: PASS")

    print()
    print("=" * 80)
    print("SATQUERYAI RUNTIME AGENT BOOTSTRAP: PASS")
    print("=" * 80)