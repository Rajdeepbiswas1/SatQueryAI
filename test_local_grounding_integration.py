import sys
import time
import traceback
from pathlib import Path

import torch

PROJECT = Path(r"D:\SatQueryAI")
GROUNDING_DIR = (
    PROJECT
    / "src"
    / "models"
    / "grounding_recovered"
)

POINT_CKPT = (
    PROJECT
    / "models"
    / "checkpoints"
    / "point_grounding"
    / "best.pt"
)

C3_CKPT = (
    PROJECT
    / "models"
    / "checkpoints"
    / "c3_v1"
    / "best.pt"
)

# Make recovered modules importable.
sys.path.insert(0, str(GROUNDING_DIR))


from point_grounding_recovered import PointGroundingModel
from c3_reference_grounding_recovered import (
    C3QueryConditionedGrounding,
)
from unified_grounding_recovered import (
    UnifiedGroundingModel,
    UnifiedGroundingInference,
    SatQueryAIGroundingRouter,
)


print("=" * 72)
print("SATQUERYAI — LOCAL UNIFIED GROUNDING INTEGRATION TEST")
print("=" * 72)
print("RUN: LOCAL")
print("DEVICE: CPU")
print("GPU: OFF")
print()


# ------------------------------------------------------------
# 1. Instantiate specialists
# ------------------------------------------------------------

print("[1/7] Instantiating specialists...")

point_model = PointGroundingModel()
reference_model = C3QueryConditionedGrounding()

point_model.eval()
reference_model.eval()

print("Point specialist: PASS")
print("Reference specialist: PASS")
print()


# ------------------------------------------------------------
# 2. Load exact checkpoints
# ------------------------------------------------------------

print("[2/7] Loading checkpoints...")

point_ckpt = torch.load(
    POINT_CKPT,
    map_location="cpu",
    weights_only=False,
)

reference_ckpt = torch.load(
    C3_CKPT,
    map_location="cpu",
    weights_only=False,
)

point_model.load_state_dict(
    point_ckpt["model_state_dict"],
    strict=True,
)

reference_model.load_state_dict(
    reference_ckpt["model_state_dict"],
    strict=True,
)

print(
    "Point checkpoint: PASS "
    f"(epoch {point_ckpt.get('epoch')})"
)

print(
    "C3 checkpoint: PASS "
    f"(epoch {reference_ckpt.get('epoch')})"
)

print()


# ------------------------------------------------------------
# 3. Build unified model
# ------------------------------------------------------------

print("[3/7] Building UnifiedGroundingModel...")

unified_model = UnifiedGroundingModel(
    point_model,
    reference_model,
)

unified_model.eval()

unified_params = sum(
    p.numel()
    for p in unified_model.parameters()
)

print(
    f"Unified model parameters: "
    f"{unified_params:,}"
)

if unified_params != 3_718_662:
    raise RuntimeError(
        "Unexpected unified parameter count: "
        f"{unified_params:,}"
    )

print("Unified model: PASS")
print()


# ------------------------------------------------------------
# 4. Build common inference interface
# ------------------------------------------------------------

print("[4/7] Building UnifiedGroundingInference...")

grounding_inference = UnifiedGroundingInference(
    unified_model,
)

print("Unified inference interface: PASS")
print()


# ------------------------------------------------------------
# 5. Build router
# ------------------------------------------------------------

print("[5/7] Building SatQueryAIGroundingRouter...")

router = SatQueryAIGroundingRouter(
    grounding_inference
)

print("Grounding router: PASS")
print()


# ------------------------------------------------------------
# 6. Synthetic Point route
# ------------------------------------------------------------

print("[6/7] Testing POINT route...")

torch.manual_seed(42)

s1 = torch.randn(1, 196, 768)
s2 = torch.randn(1, 196, 768)
text = torch.randn(1, 768)

point = torch.tensor(
    [[0.50, 0.50]],
    dtype=torch.float32,
)

point_start = time.perf_counter()

try:
    point_result = router.dispatch(
        mode="point",
        s1=s1,
        s2=s2,
        text_embedding=text,
        point=point,
        query="Locate the target at the supplied point.",
        reference=None,
        patch_id="local_synthetic_point",
    )
except Exception:
    print("POINT ROUTE: FAIL")
    traceback.print_exc()
    sys.exit(1)

point_time = time.perf_counter() - point_start

print("POINT ROUTE: PASS")
print(
    "Selected specialist:",
    point_result["execution_trace"]["selected_specialist"],
)
print(
    "Evidence keys:",
    list(point_result.keys()),
)
print(
    f"Point route time: {point_time * 1000:.2f} ms"
)
print()


# ------------------------------------------------------------
# 7. Synthetic Reference route
# ------------------------------------------------------------

print("[7/7] Testing REFERENCE route...")

reference_start = time.perf_counter()

try:
    reference_result = router.dispatch(
        mode="reference",
        s1=s1,
        s2=s2,
        text_embedding=text,
        point=None,
        query="Where is the largest urban area in the image?",
        reference="largest urban area",
        patch_id="local_synthetic_reference",
    )
except Exception:
    print("REFERENCE ROUTE: FAIL")
    traceback.print_exc()
    sys.exit(1)

reference_time = time.perf_counter() - reference_start

print("REFERENCE ROUTE: PASS")
print(
    "Selected specialist:",
    reference_result["execution_trace"]["selected_specialist"],
)
print(
    "Evidence keys:",
    list(reference_result.keys()),
)
print(
    f"Reference route time: "
    f"{reference_time * 1000:.2f} ms"
)
print()


# ------------------------------------------------------------
# Validate standardized evidence
# ------------------------------------------------------------

required_evidence = {
    "mode",
    "bbox",
    "heatmap",
    "evidence_confidence",
    "peak_probability",
    "spatial_entropy",
    "bbox_source",
    "query",
    "reference",
    "patch_id",
    "native_output",
}

for name, result in (
    ("POINT", point_result),
    ("REFERENCE", reference_result),
):
    missing = required_evidence - set(result.keys())

    if missing:
        raise RuntimeError(
            f"{name} standardized evidence missing: "
            f"{sorted(missing)}"
        )

    print(
        f"{name} standardized evidence: PASS"
    )

print()


# ------------------------------------------------------------
# Validate execution trace
# ------------------------------------------------------------

for name, result in (
    ("POINT", point_result),
    ("REFERENCE", reference_result),
):

    trace = result.get("execution_trace")

    if not isinstance(trace, dict):
        raise RuntimeError(
            f"{name} execution trace missing"
        )

    required_trace = {
        "task",
        "requested_mode",
        "selected_specialist",
        "point_used",
        "query_available",
        "reference_available",
        "patch_available",
        "input_shapes",
        "validation_time_ms",
        "inference_time_ms",
        "total_time_ms",
        "status",
    }

    missing = required_trace - set(trace.keys())

    if missing:
        raise RuntimeError(
            f"{name} execution trace missing: "
            f"{sorted(missing)}"
        )

    if trace["status"] != "success":
        raise RuntimeError(
            f"{name} route status: "
            f"{trace['status']}"
        )

    print(
        f"{name} execution trace: PASS"
    )

print()


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 72)
print("LOCAL UNIFIED GROUNDING INTEGRATION: COMPLETE")
print("=" * 72)

print()
print("POINT ROUTE")
print(
    "  Router -> PointGroundingModel -> "
    "Unified Evidence: PASS"
)

print()
print("REFERENCE ROUTE")
print(
    "  Router -> C3QueryConditionedGrounding -> "
    "Unified Evidence: PASS"
)

print()
print("SPECIALIST DISPATCH: PASS")
print("STANDARDIZED EVIDENCE: PASS")
print("EXECUTION TRACE: PASS")
print(
    f"UNIFIED PARAMETERS: {unified_params:,}"
)

print()
print("TRAINING: NOT STARTED")
print("KAGGLE: NOT USED")
print("GPU: OFF")
print("DEVICE: CPU")
print("=" * 72)