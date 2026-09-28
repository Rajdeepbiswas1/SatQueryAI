"""
SatQueryAI Unified Grounding Integration

CLEAN SOURCE RECOVERY

This file contains ONLY:

- required imports
- verified specialist imports
- integration helper functions
- UnifiedGroundingModel
- UnifiedGroundingInference
- SatQueryAIGroundingRouter

Excluded intentionally:

- notebook runtime assignments
- datasets
- DataLoaders
- training code
- checkpoint loading
- evaluation state
- temporary notebook variables
- model instances
- Kaggle runtime state
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_grounding_recovered import PointGroundingModel
from .c3_reference_grounding_recovered import (
    C3QueryConditionedGrounding,
)


# ============================================================
# VERIFIED UNIFIED GROUNDING MODEL
# ============================================================

class UnifiedGroundingModel(nn.Module):
    """
    Unified grounding interface.

    mode = "point"
        Uses the exact trained point-grounding specialist.

    mode = "reference"
        Uses the exact trained C3 reference-grounding specialist.

    This first integration deliberately does NOT retrain or
    modify either specialist.
    """

    def __init__(
        self,
        point_model,
        reference_model,
    ):
        super().__init__()

        self.point_model = point_model
        self.reference_model = reference_model

    def forward(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
    ):

        if mode == "point":

            if point is None:

                raise ValueError(
                    "Point mode requires point coordinates."
                )

            return self.point_model(
                s1,
                s2,
                text_embedding,
                point,
            )

        if mode == "reference":

            return self.reference_model(
                s1,
                s2,
                text_embedding,
            )

        raise ValueError(
            f"Unsupported grounding mode: {mode}"
        )


# ============================================================
# VERIFIED INTEGRATION HELPERS
# ============================================================

# Helper: _to_batched
def _to_batched(x, name):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch Tensor.")

    if x.ndim == 1:
        return x.unsqueeze(0)

    if x.ndim == 2:
        return x.unsqueeze(0)

    return x

# Helper: _move_to_model_device
def _move_to_model_device(x):
    device = next(unified_model.parameters()).device
    return x.to(device)

# Helper: _prepare_s1_s2
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

# Helper: _prepare_text
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

# Helper: _prepare_point
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

# Helper: _find_named_tensor
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

# Helper: _normalize_heatmap
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

# Helper: _heatmap_moment_box_generic
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

# Helper: _sanitize_bbox
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

# Helper: _extract_evidence
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


# ============================================================
# VERIFIED UNIFIED GROUNDING INFERENCE
# ============================================================

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


# ============================================================
# VERIFIED SATQUERYAI GROUNDING ROUTER
# ============================================================

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

            if not torch.is_tensor(point):
                raise TypeError(
                    "point must be a Tensor."
                )

            # Supported point formats:
            #
            #   [2]    -> single point
            #   [B,2]  -> batch of points
            #
            # The Point specialist itself expects [B,2].
            if point.ndim == 1:

                if point.numel() != 2:
                    raise ValueError(
                        "Single point must contain "
                        "exactly 2 coordinates."
                    )

                point_for_validation = (
                    point.reshape(1, 2)
                )

            elif point.ndim == 2:

                if point.shape[-1] != 2:
                    raise ValueError(
                        "Point batch must have shape [B,2]. "
                        f"Got {tuple(point.shape)}"
                    )

                if point.shape[0] != s1.shape[0]:
                    raise ValueError(
                        "Point and visual batch sizes do "
                        "not match. "
                        f"point={tuple(point.shape)}, "
                        f"s1={tuple(s1.shape)}"
                    )

                point_for_validation = point

            else:

                raise ValueError(
                    "Point must have shape [2] "
                    "or [B,2]. "
                    f"Got {tuple(point.shape)}"
                )

            point_float = (
                point_for_validation
                .detach()
                .float()
            )

            if not torch.all(
                (point_float >= 0.0)
                &
                (point_float <= 1.0)
            ):
                raise ValueError(
                    "Point coordinates must lie in [0,1]."
                )

        return mode

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

            "patch_available":
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
