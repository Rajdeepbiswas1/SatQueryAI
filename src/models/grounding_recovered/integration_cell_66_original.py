# =============================================================================
# SATQUERYAI
# COMPLETE UNIFIED GROUNDING INTEGRATION + END-TO-END SMOKE TEST
# =============================================================================
#
# RUN ON KAGGLE — GPU OFF
#
# Existing validated specialists:
#
#   PointGroundingModel
#       1,531,909 parameters
#       Best validation mIoU = 0.7395820617675781
#
#   C3QueryConditionedGrounding
#       2,186,753 parameters
#       Best validation mIoU = 0.3281804048423226
#
# Unified:
#       3,718,662 parameters
#
# This cell:
#   1. Verifies existing unified model
#   2. Restores exact C3 helper dependencies
#   3. Tests direct C3 forward
#   4. Rebuilds unified inference interface
#   5. Rebuilds grounding router
#   6. Executes point route
#   7. Executes reference route
#   8. Verifies standardized evidence
#   9. Verifies execution traces
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# NO MODEL WEIGHT MODIFICATION
# =============================================================================

import math
import time
import torch


# =============================================================================
# 1. HEADER
# =============================================================================

print("=" * 80)
print("SATQUERYAI COMPLETE UNIFIED GROUNDING INTEGRATION")
print("=" * 80)

print()
print("Execution environment:")
print("  RUN ON KAGGLE — GPU OFF")


# =============================================================================
# 2. VERIFY UNIFIED MODEL EXISTS
# =============================================================================

print()
print("-" * 80)
print("VERIFYING EXISTING UNIFIED MODEL")
print("-" * 80)

if "unified_model" not in globals():
    raise RuntimeError(
        "unified_model is missing from the current kernel."
    )

unified_model = globals()["unified_model"]

print(
    "Unified class:",
    type(unified_model).__name__
)

print(
    "Point specialist:",
    type(unified_model.point_model).__name__
)

print(
    "Reference specialist:",
    type(unified_model.reference_model).__name__
)


# =============================================================================
# 3. VERIFY SPECIALIST PARAMETER COUNTS
# =============================================================================

point_params = sum(
    p.numel()
    for p in unified_model.point_model.parameters()
)

reference_params = sum(
    p.numel()
    for p in unified_model.reference_model.parameters()
)

unified_params = sum(
    p.numel()
    for p in unified_model.parameters()
)

print()
print("Point parameters:", point_params)
print("Reference parameters:", reference_params)
print("Unified parameters:", unified_params)

if point_params != 1_531_909:
    raise RuntimeError(
        f"Unexpected point parameter count: {point_params}"
    )

if reference_params != 2_186_753:
    raise RuntimeError(
        f"Unexpected C3 parameter count: {reference_params}"
    )

if unified_params != 3_718_662:
    raise RuntimeError(
        f"Unexpected unified parameter count: {unified_params}"
    )

print()
print("Point parameter count: PASS")
print("C3 parameter count: PASS")
print("Unified parameter count: PASS")


# =============================================================================
# 4. GET ACTUAL C3 MODEL
# =============================================================================

c3_model = unified_model.reference_model

actual_c3_class = type(c3_model)
actual_forward = actual_c3_class.forward
actual_globals = actual_forward.__globals__


print()
print("-" * 80)
print("ACTUAL C3 MODEL")
print("-" * 80)

print(
    "Class:",
    actual_c3_class
)

print(
    "Class ID:",
    id(actual_c3_class)
)

print(
    "Forward globals ID:",
    id(actual_globals)
)


# =============================================================================
# 5. EXACT C3 HELPER: heatmap_moments
# =============================================================================

def heatmap_moments(
    heatmap,
):
    """
    Exact grounding helper used by the verified C2/C3 implementation.

    Input:
        [B,H,W]

    Returns:
        cx
        cy
        std_x
        std_y
    """

    B, H, W = heatmap.shape

    xs = (
        torch.arange(
            W,
            device=heatmap.device,
            dtype=heatmap.dtype
        ) + 0.5
    ) / W

    ys = (
        torch.arange(
            H,
            device=heatmap.device,
            dtype=heatmap.dtype
        ) + 0.5
    ) / H

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.reshape(
        1,
        -1
    )

    yy = yy.reshape(
        1,
        -1
    )

    p = heatmap.reshape(
        B,
        -1
    )

    cx = (
        p * xx
    ).sum(
        dim=1
    )

    cy = (
        p * yy
    ).sum(
        dim=1
    )

    var_x = (
        p
        *
        (
            xx
            -
            cx.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )

    var_y = (
        p
        *
        (
            yy
            -
            cy.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )

    std_x = torch.sqrt(
        var_x + 1e-8
    )

    std_y = torch.sqrt(
        var_y + 1e-8
    )

    return (
        cx,
        cy,
        std_x,
        std_y
    )


# =============================================================================
# 6. EXACT C3 HELPER: heatmap_to_box
# =============================================================================

def heatmap_to_box(
    heatmap,
):

    cx, cy, std_x, std_y = heatmap_moments(
        heatmap
    )

    width = (
        std_x * math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )

    height = (
        std_y * math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )

    x1 = (
        cx - width / 2.0
    ).clamp(
        0.0,
        1.0
    )

    y1 = (
        cy - height / 2.0
    ).clamp(
        0.0,
        1.0
    )

    x2 = (
        cx + width / 2.0
    ).clamp(
        0.0,
        1.0
    )

    y2 = (
        cy + height / 2.0
    ).clamp(
        0.0,
        1.0
    )

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1
    )


# =============================================================================
# 7. PATCH ACTUAL C3 FORWARD GLOBALS
# =============================================================================

actual_globals["heatmap_moments"] = heatmap_moments
actual_globals["heatmap_to_box"] = heatmap_to_box
actual_globals["math"] = math


print()
print("-" * 80)
print("PATCHING ACTUAL C3 FORWARD GLOBALS")
print("-" * 80)

print(
    "heatmap_moments present:",
    "heatmap_moments" in actual_globals
)

print(
    "heatmap_to_box present:",
    "heatmap_to_box" in actual_globals
)

print(
    "math present:",
    "math" in actual_globals
)


if actual_globals.get("heatmap_moments") is not heatmap_moments:
    raise RuntimeError(
        "heatmap_moments global patch failed."
    )

if actual_globals.get("heatmap_to_box") is not heatmap_to_box:
    raise RuntimeError(
        "heatmap_to_box global patch failed."
    )

if actual_globals.get("math") is not math:
    raise RuntimeError(
        "math global patch failed."
    )

print()
print("C3 helper dependency injection: PASS")


# =============================================================================
# 8. TEST C3 HELPERS
# =============================================================================

print()
print("-" * 80)
print("TESTING C3 HELPERS")
print("-" * 80)

test_heatmap = torch.ones(
    1,
    56,
    56,
    dtype=torch.float32
)

test_heatmap = (
    test_heatmap
    /
    test_heatmap.sum(
        dim=(1, 2),
        keepdim=True
    )
)


cx, cy, std_x, std_y = heatmap_moments(
    test_heatmap
)

test_box = heatmap_to_box(
    test_heatmap
)

print("cx:", cx)
print("cy:", cy)
print("std_x:", std_x)
print("std_y:", std_y)
print("test_box:", test_box)

if not torch.allclose(
    cx,
    torch.tensor([0.5]),
    atol=1e-6
):
    raise RuntimeError(
        "heatmap_moments centroid X test failed."
    )

if not torch.allclose(
    cy,
    torch.tensor([0.5]),
    atol=1e-6
):
    raise RuntimeError(
        "heatmap_moments centroid Y test failed."
    )

if tuple(test_box.shape) != (1, 4):
    raise RuntimeError(
        "heatmap_to_box shape test failed."
    )

if not torch.isfinite(test_box).all():
    raise RuntimeError(
        "heatmap_to_box produced non-finite values."
    )

if not torch.all(
    (test_box >= 0.0)
    &
    (test_box <= 1.0)
):
    raise RuntimeError(
        "heatmap_to_box produced values outside [0,1]."
    )

print()
print("heatmap_moments: PASS")
print("heatmap_to_box: PASS")


# =============================================================================
# 9. DIRECT C3 FORWARD TEST
# =============================================================================

print()
print("-" * 80)
print("DIRECT C3 FORWARD TEST")
print("-" * 80)

device = next(
    c3_model.parameters()
).device

s1_test = torch.randn(
    1,
    196,
    768,
    dtype=torch.float32,
    device=device
)

s2_test = torch.randn(
    1,
    196,
    768,
    dtype=torch.float32,
    device=device
)

text_test = torch.randn(
    1,
    768,
    dtype=torch.float32,
    device=device
)

c3_model.eval()

with torch.no_grad():

    c3_output = c3_model(
        s1_test,
        s2_test,
        text_test
    )


print("Direct C3 forward: PASS")


# =============================================================================
# 10. VERIFY C3 OUTPUTS
# =============================================================================

expected_c3_outputs = {
    "semantic_logits": (1, 14, 14),
    "semantic_attention": (1, 14, 14),
    "grounding_logits": (1, 56, 56),
    "grounding_heatmap": (1, 56, 56),
    "cx": (1,),
    "cy": (1,),
    "std_x": (1,),
    "std_y": (1,),
    "predicted_box": (1, 4),
}


print()
print("-" * 80)
print("C3 OUTPUT VERIFICATION")
print("-" * 80)

for key, expected_shape in expected_c3_outputs.items():

    if key not in c3_output:
        raise RuntimeError(
            f"C3 output missing: {key}"
        )

    actual_shape = tuple(
        c3_output[key].shape
    )

    if actual_shape != expected_shape:
        raise RuntimeError(
            f"{key}: expected {expected_shape}, "
            f"got {actual_shape}"
        )

    if not torch.isfinite(
        c3_output[key]
    ).all():

        raise RuntimeError(
            f"{key} contains non-finite values."
        )

    print(
        f"{key}: {actual_shape} → PASS"
    )


semantic_sum = c3_output[
    "semantic_attention"
].sum(
    dim=(1, 2)
)

grounding_sum = c3_output[
    "grounding_heatmap"
].sum(
    dim=(1, 2)
)

if not torch.allclose(
    semantic_sum,
    torch.ones_like(semantic_sum),
    atol=1e-4
):
    raise RuntimeError(
        "Semantic attention normalization failed."
    )

if not torch.allclose(
    grounding_sum,
    torch.ones_like(grounding_sum),
    atol=1e-4
):
    raise RuntimeError(
        "Grounding heatmap normalization failed."
    )


print()
print("Semantic attention normalization: PASS")
print("Grounding heatmap normalization: PASS")


# =============================================================================
# 11. PREPARE INPUT FUNCTIONS
# =============================================================================

def _prepare_s1_s2(
    x,
    name
):
    """
    [196,768] -> [1,196,768]
    [B,196,768] -> unchanged
    """

    if not torch.is_tensor(x):
        raise TypeError(
            f"{name} must be a Tensor."
        )

    if x.ndim == 2:

        if tuple(x.shape) != (196, 768):
            raise ValueError(
                f"{name} expected [196,768], "
                f"got {tuple(x.shape)}"
            )

        return x.unsqueeze(0)

    if x.ndim == 3:

        if x.shape[-1] != 768:
            raise ValueError(
                f"{name} final dimension must be 768, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-2] != 196:
            raise ValueError(
                f"{name} expected 196 spatial tokens, "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        f"{name} must have shape [196,768] "
        f"or [B,196,768]. Got {tuple(x.shape)}"
    )


def _prepare_text(
    x
):
    """
    [768] -> [1,768]
    [B,768] -> unchanged
    """

    if not torch.is_tensor(x):
        raise TypeError(
            "text_embedding must be a Tensor."
        )

    if x.ndim == 1:

        if x.shape[0] != 768:
            raise ValueError(
                f"Expected text embedding of size 768, "
                f"got {tuple(x.shape)}"
            )

        return x.unsqueeze(0)

    if x.ndim == 2:

        if x.shape[-1] != 768:
            raise ValueError(
                f"Text embedding final dimension must be 768, "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        "text_embedding must have shape [768] "
        "or [B,768]."
    )


def _prepare_point(
    x
):
    """
    [2] -> [1,2]
    [B,2] -> unchanged

    IMPORTANT:
        A [B,2] point is never unsqueezed.
    """

    if not torch.is_tensor(x):
        raise TypeError(
            "point must be a Tensor."
        )

    if x.ndim == 1:

        if x.numel() != 2:
            raise ValueError(
                f"Point must contain exactly 2 values, "
                f"got {tuple(x.shape)}"
            )

        return x.reshape(
            1,
            2
        )

    if x.ndim == 2:

        if x.shape[-1] != 2:
            raise ValueError(
                f"Point expected [B,2], "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        f"Point must have shape [2] "
        f"or [B,2]. Got {tuple(x.shape)}"
    )


# =============================================================================
# 12. OUTPUT UTILITIES
# =============================================================================

def _find_named_tensor(
    output,
    names
):

    if not isinstance(output, dict):
        return None

    for name in names:

        value = output.get(name)

        if torch.is_tensor(value):
            return value

    return None


def _normalize_heatmap(
    x
):

    if not torch.is_tensor(x):
        raise TypeError(
            "Heatmap must be a Tensor."
        )

    x = x.float()

    if x.ndim == 2:

        x = x.unsqueeze(0)

    elif x.ndim == 4:

        # Expected one spatial channel for a generic heatmap.
        x = x[:, 0]

    if x.ndim != 3:

        raise RuntimeError(
            f"Unexpected heatmap shape: "
            f"{tuple(x.shape)}"
        )

    sums = x.sum(
        dim=(-2, -1),
        keepdim=True
    )

    looks_like_probability = (
        torch.all(
            x >= 0
        ).item()
        and
        torch.all(
            x <= 1.0 + 1e-5
        ).item()
        and
        torch.all(
            sums > 0
        ).item()
    )

    if looks_like_probability:

        return (
            x
            /
            sums.clamp_min(1e-8)
        )

    return torch.softmax(
        x.flatten(1),
        dim=1
    ).reshape_as(x)


def _heatmap_moment_box_generic(
    prob
):

    B, H, W = prob.shape

    ys = torch.linspace(
        0.0,
        1.0,
        H,
        device=prob.device,
        dtype=prob.dtype
    )

    xs = torch.linspace(
        0.0,
        1.0,
        W,
        device=prob.device,
        dtype=prob.dtype
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    cx = (
        prob * xx
    ).sum(
        dim=(-2, -1)
    )

    cy = (
        prob * yy
    ).sum(
        dim=(-2, -1)
    )

    var_x = (
        prob
        *
        (
            xx
            -
            cx[:, None, None]
        ).pow(2)
    ).sum(
        dim=(-2, -1)
    )

    var_y = (
        prob
        *
        (
            yy
            -
            cy[:, None, None]
        ).pow(2)
    ).sum(
        dim=(-2, -1)
    )

    std_x = torch.sqrt(
        var_x.clamp_min(1e-8)
    )

    std_y = torch.sqrt(
        var_y.clamp_min(1e-8)
    )

    return torch.stack(
        [
            (
                cx - 2.0 * std_x
            ).clamp(0.0, 1.0),

            (
                cy - 2.0 * std_y
            ).clamp(0.0, 1.0),

            (
                cx + 2.0 * std_x
            ).clamp(0.0, 1.0),

            (
                cy + 2.0 * std_y
            ).clamp(0.0, 1.0),
        ],
        dim=-1
    )


def _sanitize_bbox(
    bbox
):

    bbox = bbox.float().clamp(
        0.0,
        1.0
    )

    x1 = torch.minimum(
        bbox[:, 0],
        bbox[:, 2]
    )

    y1 = torch.minimum(
        bbox[:, 1],
        bbox[:, 3]
    )

    x2 = torch.maximum(
        bbox[:, 0],
        bbox[:, 2]
    )

    y2 = torch.maximum(
        bbox[:, 1],
        bbox[:, 3]
    )

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=-1
    )


def _extract_evidence(
    output
):

    if not isinstance(output, dict):
        raise RuntimeError(
            "Specialist output must be a dictionary."
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Point specialist:
    #     heatmap
    #     pred_box
    #
    # C3 reference specialist:
    #     grounding_heatmap
    #     predicted_box
    # -------------------------------------------------------------------------

    heatmap = _find_named_tensor(
        output,
        [
            "heatmap",
            "grounding_heatmap",
            "heatmap_probs",
            "heatmap_logits",
            "grounding_logits",
            "match_heatmap",
        ]
    )

    if heatmap is None:
        raise RuntimeError(
            "No grounding heatmap found. "
            f"Available output keys: {list(output.keys())}"
        )

    probability_heatmap = _normalize_heatmap(
        heatmap
    )

    bbox = _find_named_tensor(
        output,
        [
            "pred_box",
            "predicted_box",
            "bbox",
            "box",
        ]
    )

    if bbox is not None:

        bbox = bbox.float()

        if bbox.ndim == 1:
            bbox = bbox.unsqueeze(0)

        if bbox.ndim != 2 or bbox.shape[-1] != 4:
            raise RuntimeError(
                f"Unexpected bbox shape: "
                f"{tuple(bbox.shape)}"
            )

        bbox = _sanitize_bbox(
            bbox
        )

        bbox_source = (
            "specialist_predicted_box"
        )

    else:

        bbox = _heatmap_moment_box_generic(
            probability_heatmap
        )

        bbox = _sanitize_bbox(
            bbox
        )

        bbox_source = (
            "heatmap_moments"
        )

    flat = probability_heatmap.flatten(
        1
    )

    peak_probability = flat.max(
        dim=1
    ).values

    spatial_entropy = -(
        probability_heatmap.clamp_min(
            1e-8
        )
        *
        probability_heatmap.clamp_min(
            1e-8
        ).log()
    ).sum(
        dim=(-2, -1)
    )

    return {
        "heatmap": probability_heatmap,
        "bbox": bbox,
        "bbox_source": bbox_source,
        "peak_probability": peak_probability,
        "spatial_entropy": spatial_entropy,
    }


# =============================================================================
# 13. UNIFIED INFERENCE INTERFACE
# =============================================================================

class UnifiedGroundingInference:

    def __init__(
        self,
        unified_model
    ):

        self.model = unified_model
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
        query=None,
        reference=None,
        patch_id=None,
    ):

        mode = str(
            mode
        ).strip().lower()

        if mode not in (
            "point",
            "reference"
        ):
            raise ValueError(
                f"Unsupported grounding mode: {mode}"
            )

        # ---------------------------------------------------------------------
        # Prepare inputs
        # ---------------------------------------------------------------------

        s1 = _prepare_s1_s2(
            s1,
            "s1"
        )

        s2 = _prepare_s1_s2(
            s2,
            "s2"
        )

        text_embedding = _prepare_text(
            text_embedding
        )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires point."
                )

            point = _prepare_point(
                point
            )

        # ---------------------------------------------------------------------
        # Move to model device
        # ---------------------------------------------------------------------

        device = next(
            self.model.parameters()
        ).device

        s1 = s1.to(device)
        s2 = s2.to(device)
        text_embedding = text_embedding.to(
            device
        )

        if point is not None:
            point = point.to(device)

        # ---------------------------------------------------------------------
        # Shape validation
        # ---------------------------------------------------------------------

        if s1.ndim != 3:
            raise RuntimeError(
                f"s1 must be [B,196,768], got {tuple(s1.shape)}"
            )

        if s2.ndim != 3:
            raise RuntimeError(
                f"s2 must be [B,196,768], got {tuple(s2.shape)}"
            )

        if text_embedding.ndim != 2:
            raise RuntimeError(
                f"text_embedding must be [B,768], "
                f"got {tuple(text_embedding.shape)}"
            )

        if s1.shape[0] != s2.shape[0]:
            raise RuntimeError(
                "S1 and S2 batch sizes do not match."
            )

        if text_embedding.shape[0] != s1.shape[0]:
            raise RuntimeError(
                "Text and visual batch sizes do not match."
            )

        if mode == "point":

            if point.ndim != 2:
                raise RuntimeError(
                    f"point must be [B,2], got {tuple(point.shape)}"
                )

            if point.shape[0] != s1.shape[0]:
                raise RuntimeError(
                    "Point and visual batch sizes do not match."
                )

        # ---------------------------------------------------------------------
        # Specialist dispatch
        # ---------------------------------------------------------------------

        if mode == "point":

            native_output = self.model(
                mode="point",
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
                point=point,
            )

        else:

            native_output = self.model(
                mode="reference",
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
            )

        # ---------------------------------------------------------------------
        # Standardize evidence
        # ---------------------------------------------------------------------

        evidence = _extract_evidence(
            native_output
        )

        heatmap = evidence[
            "heatmap"
        ]

        bbox = evidence[
            "bbox"
        ]

        peak = evidence[
            "peak_probability"
        ]

        # ---------------------------------------------------------------------
        # Evidence concentration
        # ---------------------------------------------------------------------

        B, H, W = heatmap.shape

        x1 = torch.floor(
            bbox[:, 0] * (W - 1)
        ).long().clamp(
            0,
            W - 1
        )

        x2 = torch.ceil(
            bbox[:, 2] * (W - 1)
        ).long().clamp(
            0,
            W - 1
        )

        y1 = torch.floor(
            bbox[:, 1] * (H - 1)
        ).long().clamp(
            0,
            H - 1
        )

        y2 = torch.ceil(
            bbox[:, 3] * (H - 1)
        ).long().clamp(
            0,
            H - 1
        )

        concentration = []

        for i in range(B):

            region = heatmap[
                i,
                y1[i]:y2[i] + 1,
                x1[i]:x2[i] + 1
            ]

            concentration.append(
                region.sum()
            )

        concentration = torch.stack(
            concentration
        )

        # ---------------------------------------------------------------------
        # Diagnostic evidence score
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        # This is NOT a calibrated probability of correctness.
        # ---------------------------------------------------------------------

        evidence_confidence = (
            0.5 * peak
            +
            0.5 * concentration
        ).clamp(
            0.0,
            1.0
        )

        # ---------------------------------------------------------------------
        # Standard result
        # ---------------------------------------------------------------------

        return {
            "mode": mode,

            "bbox":
                bbox.detach()
                .cpu()
                .numpy(),

            "heatmap":
                heatmap.detach()
                .cpu()
                .numpy(),

            "evidence_confidence":
                evidence_confidence.detach()
                .cpu()
                .numpy(),

            "peak_probability":
                peak.detach()
                .cpu()
                .numpy(),

            "spatial_entropy":
                evidence[
                    "spatial_entropy"
                ]
                .detach()
                .cpu()
                .numpy(),

            "bbox_source":
                evidence[
                    "bbox_source"
                ],

            "query": query,

            "reference": reference,

            "patch_id": patch_id,

            "point": (
                point.detach()
                .cpu()
                .numpy()
                if point is not None
                else None
            ),

            "native_output":
                native_output,
        }


# =============================================================================
# 14. AGENT-FACING ROUTER
# =============================================================================

class SatQueryAIGroundingRouter:

    SUPPORTED_MODES = {
        "point",
        "reference"
    }

    def __init__(
        self,
        grounding_inference
    ):

        self.grounding_inference = (
            grounding_inference
        )

        self.model = (
            grounding_inference.model
        )

        self.model.eval()

    def _validate_request(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
    ):

        mode = str(
            mode
        ).strip().lower()

        if mode not in self.SUPPORTED_MODES:

            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Supported modes: "
                f"{sorted(self.SUPPORTED_MODES)}"
            )

        if not torch.is_tensor(s1):
            raise TypeError(
                "s1 must be a Tensor."
            )

        if not torch.is_tensor(s2):
            raise TypeError(
                "s2 must be a Tensor."
            )

        if not torch.is_tensor(text_embedding):
            raise TypeError(
                "text_embedding must be a Tensor."
            )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires a point."
                )

            point_flat = point.detach().float().flatten()

            if point_flat.numel() != 2:
                raise ValueError(
                    "Point must contain exactly 2 coordinates."
                )

            if not torch.all(
                (point_flat >= 0.0)
                &
                (point_flat <= 1.0)
            ):
                raise ValueError(
                    "Point coordinates must lie in [0,1]."
                )

        return mode

    def _specialist_name(
        self,
        mode
    ):

        if mode == "point":

            return type(
                self.model.point_model
            ).__name__

        return type(
            self.model.reference_model
        ).__name__

    def dispatch(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
        query=None,
        reference=None,
        patch_id=None,
    ):

        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Request validation
        # ---------------------------------------------------------------------

        mode = self._validate_request(
            mode=mode,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
        )

        validation_end = time.perf_counter()

        specialist_name = (
            self._specialist_name(
                mode
            )
        )

        # ---------------------------------------------------------------------
        # Execute inference
        # ---------------------------------------------------------------------

        result = self.grounding_inference.predict(
            mode=mode,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
            query=query,
            reference=reference,
            patch_id=patch_id,
        )

        inference_end = time.perf_counter()

        # ---------------------------------------------------------------------
        # Observable execution trace
        # ---------------------------------------------------------------------

        result["execution_trace"] = {

            "task":
                "grounding",

            "requested_mode":
                mode,

            "selected_specialist":
                specialist_name,

            "point_used":
                (
                    point is not None
                    if mode == "point"
                    else False
                ),

            "query_available":
                query is not None,

            "reference_available":
                reference is not None,

            "patch_id_available":
                patch_id is not None,

            "input_shapes": {

                "s1":
                    tuple(s1.shape),

                "s2":
                    tuple(s2.shape),

                "text_embedding":
                    tuple(
                        text_embedding.shape
                    ),

                "point":
                    (
                        tuple(point.shape)
                        if point is not None
                        else None
                    ),
            },

            "validation_time_ms":
                (
                    validation_end
                    -
                    start_time
                ) * 1000.0,

            "inference_time_ms":
                (
                    inference_end
                    -
                    validation_end
                ) * 1000.0,

            "total_time_ms":
                (
                    inference_end
                    -
                    start_time
                ) * 1000.0,

            "status":
                "success",
        }

        return result


# =============================================================================
# 15. REBUILD INFERENCE INTERFACE
# =============================================================================

grounding_inference = UnifiedGroundingInference(
    unified_model
)


# =============================================================================
# 16. REBUILD ROUTER
# =============================================================================

grounding_router = SatQueryAIGroundingRouter(
    grounding_inference
)


print()
print("-" * 80)
print("UNIFIED INFERENCE + ROUTER")
print("-" * 80)

print(
    "Inference:",
    type(
        grounding_inference
    ).__name__
)

print(
    "Router:",
    type(
        grounding_router
    ).__name__
)

print("Point mode → PointGroundingModel")
print("Reference mode → C3QueryConditionedGrounding")


# =============================================================================
# 17. END-TO-END POINT TEST
# =============================================================================

print()
print("-" * 80)
print("POINT MODE END-TO-END TEST")
print("-" * 80)

point_start = time.perf_counter()

point_result = grounding_router.dispatch(
    mode="point",

    s1=s1_test,

    s2=s2_test,

    text_embedding=text_test,

    point=torch.tensor(
        [[0.50, 0.50]],
        dtype=torch.float32,
        device=device
    ),

    query="locate the target at the supplied point",

    patch_id="unified_smoke_point",
)

point_total_ms = (
    time.perf_counter()
    -
    point_start
) * 1000.0


print(
    "Point dispatch: PASS"
)

print(
    "Point bbox shape:",
    point_result["bbox"].shape
)

print(
    "Point heatmap shape:",
    point_result["heatmap"].shape
)

print(
    "Point confidence:",
    point_result[
        "evidence_confidence"
    ]
)

print(
    "Point dispatch time:",
    round(
        point_total_ms,
        3
    ),
    "ms"
)


# =============================================================================
# 18. POINT OUTPUT CONTRACT
# =============================================================================

required_fields = [
    "mode",
    "bbox",
    "heatmap",
    "evidence_confidence",
    "peak_probability",
    "spatial_entropy",
    "bbox_source",
    "point",
    "native_output",
    "execution_trace",
]

for key in required_fields:

    if key not in point_result:
        raise RuntimeError(
            f"Point result missing field: {key}"
        )

print()
print("Point evidence contract: PASS")

if point_result["mode"] != "point":
    raise RuntimeError(
        "Point result mode is incorrect."
    )

if point_result["point"] is None:
    raise RuntimeError(
        "Point information was not preserved."
    )

point_trace = point_result[
    "execution_trace"
]

if point_trace[
    "selected_specialist"
] != "PointGroundingModel":

    raise RuntimeError(
        "Point route selected incorrect specialist."
    )

if point_trace[
    "status"
] != "success":

    raise RuntimeError(
        "Point execution status is not success."
    )

print(
    "Point specialist selection: PASS"
)

print(
    "Point execution trace: PASS"
)


# =============================================================================
# 19. END-TO-END REFERENCE TEST
# =============================================================================

print()
print("-" * 80)
print("REFERENCE MODE END-TO-END TEST")
print("-" * 80)

reference_start = time.perf_counter()

reference_result = grounding_router.dispatch(
    mode="reference",

    s1=s1_test,

    s2=s2_test,

    text_embedding=text_test,

    query="locate the largest urban area",

    reference="largest urban area",

    patch_id="unified_smoke_reference",
)

reference_total_ms = (
    time.perf_counter()
    -
    reference_start
) * 1000.0


print(
    "Reference dispatch: PASS"
)

print(
    "Reference bbox shape:",
    reference_result["bbox"].shape
)

print(
    "Reference heatmap shape:",
    reference_result["heatmap"].shape
)

print(
    "Reference confidence:",
    reference_result[
        "evidence_confidence"
    ]
)

print(
    "Reference dispatch time:",
    round(
        reference_total_ms,
        3
    ),
    "ms"
)


# =============================================================================
# 20. REFERENCE OUTPUT CONTRACT
# =============================================================================

for key in required_fields:

    if key not in reference_result:
        raise RuntimeError(
            f"Reference result missing field: {key}"
        )

print()
print(
    "Reference evidence contract: PASS"
)

if reference_result["mode"] != "reference":
    raise RuntimeError(
        "Reference result mode is incorrect."
    )

reference_trace = reference_result[
    "execution_trace"
]

if reference_trace[
    "selected_specialist"
] != (
    "C3QueryConditionedGrounding"
):

    raise RuntimeError(
        "Reference route selected incorrect specialist."
    )

if reference_trace[
    "status"
] != "success":

    raise RuntimeError(
        "Reference execution status is not success."
    )

print(
    "Reference specialist selection: PASS"
)

print(
    "Reference execution trace: PASS"
)


# =============================================================================
# 21. VERIFY FINAL RESULT SHAPES
# =============================================================================

print()
print("-" * 80)
print("FINAL STANDARDIZED OUTPUT VERIFICATION")
print("-" * 80)

if point_result[
    "bbox"
].shape != (1, 4):

    raise RuntimeError(
        "Point bbox final shape incorrect."
    )

if reference_result[
    "bbox"
].shape != (1, 4):

    raise RuntimeError(
        "Reference bbox final shape incorrect."
    )

if point_result[
    "heatmap"
].ndim != 3:

    raise RuntimeError(
        "Point heatmap final dimensions incorrect."
    )

if reference_result[
    "heatmap"
].ndim != 3:

    raise RuntimeError(
        "Reference heatmap final dimensions incorrect."
    )

if not torch.isfinite(
    torch.tensor(
        point_result[
            "bbox"
        ]
    )
).all():

    raise RuntimeError(
        "Point standardized bbox is non-finite."
    )

if not torch.isfinite(
    torch.tensor(
        reference_result[
            "bbox"
        ]
    )
).all():

    raise RuntimeError(
        "Reference standardized bbox is non-finite."
    )


print(
    "Point bbox contract: PASS"
)

print(
    "Reference bbox contract: PASS"
)

print(
    "Point heatmap contract: PASS"
)

print(
    "Reference heatmap contract: PASS"
)

print(
    "Point standardized evidence: PASS"
)

print(
    "Reference standardized evidence: PASS"
)


# =============================================================================
# 22. FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("SATQUERYAI UNIFIED GROUNDING END-TO-END TEST COMPLETE")
print("=" * 80)

print()
print("C3 helper dependencies:       PASS")
print("Direct C3 forward:            PASS")
print("C3 output contract:           PASS")
print("C3 probability normalization: PASS")

print()
print("Point route:")
print(
    "  Router"
    " → PointGroundingModel"
    " → Unified Evidence: PASS"
)

print()
print("Reference route:")
print(
    "  Router"
    " → C3QueryConditionedGrounding"
    " → Unified Evidence: PASS"
)

print()
print("Specialist routing:            PASS")
print("Standardized evidence:         PASS")
print("Execution tracing:             PASS")

print()
print("Unified parameters:", unified_params)

print()
print("No training performed.")
print("No checkpoint modified.")
print("No specialist modified.")
print("No model weights modified.")

print()
print("=" * 80)
print("GROUNDING SUBSYSTEM INTEGRATION VERIFIED")
print("=" * 80)