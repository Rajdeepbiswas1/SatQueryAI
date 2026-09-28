# ============================================================
# UNIFIED POINT + REFERENCE GROUNDING INTEGRATION
# RUN ON KAGGLE — GPU OFF
#
# PURPOSE:
#   1. Recover exact PointGroundingModel
#   2. Load exact point checkpoint
#   3. Reuse exact recovered C3 reference model/checkpoint
#   4. Build one unified routing interface
#   5. Verify both specialists in one model
# ============================================================

import os
from pathlib import Path
import torch
import torch.nn as nn

print("=" * 80)
print("UNIFIED POINT + REFERENCE GROUNDING INTEGRATION")
print("=" * 80)

# ------------------------------------------------------------
# 1. LOCATE POINT CHECKPOINT
# ------------------------------------------------------------

point_candidates = [
    Path("/kaggle/working/point_grounding/best.pt"),
    Path("/kaggle/working/point_grounding/checkpoints/best.pt"),
]

for path in Path("/kaggle/input").rglob("best.pt"):

    name = str(path).lower()

    if "point" in name and path not in point_candidates:
        point_candidates.append(path)

point_checkpoint_path = None

for path in point_candidates:

    if path.exists():
        point_checkpoint_path = path
        break

if point_checkpoint_path is None:

    raise FileNotFoundError(
        "Point grounding best.pt was not found.\n"
        "Expected either:\n"
        "/kaggle/working/point_grounding/best.pt\n"
        "or a point-grounding checkpoint under /kaggle/input."
    )

print("\nPoint checkpoint:")
print(" ", point_checkpoint_path)

# ------------------------------------------------------------
# 2. LOAD POINT CHECKPOINT
# ------------------------------------------------------------

point_checkpoint = torch.load(
    point_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

print("\nPoint checkpoint keys:")

for key in point_checkpoint.keys():
    print("  ", key)

print(
    "\nPoint checkpoint epoch:",
    point_checkpoint.get("epoch")
)

print(
    "Point stored validation IoU:",
    point_checkpoint.get("val_mean_iou")
)

point_state = point_checkpoint.get(
    "model_state_dict"
)

if point_state is None:

    raise RuntimeError(
        "Point checkpoint does not contain model_state_dict."
    )

print(
    "Point state-dict entries:",
    len(point_state)
)

# ------------------------------------------------------------
# 3. FIND EXACT POINT MODEL CLASS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("LOCATING EXACT POINT MODEL CLASS")
print("-" * 80)

PointGroundingModel = globals().get(
    "PointGroundingModel",
    None
)

if PointGroundingModel is None:

    raise RuntimeError(
        "PointGroundingModel is not present in the current kernel."
    )

print(
    "Found:",
    PointGroundingModel
)

# ------------------------------------------------------------
# 4. INSTANTIATE + VERIFY POINT MODEL
# ------------------------------------------------------------

point_model = PointGroundingModel()

point_parameter_count = sum(
    p.numel()
    for p in point_model.parameters()
)

EXPECTED_POINT_PARAMS = 1_531_909

print(
    "Point parameter count:",
    f"{point_parameter_count:,}"
)

if point_parameter_count != EXPECTED_POINT_PARAMS:

    raise RuntimeError(
        "Point parameter count mismatch.\n"
        f"Expected: {EXPECTED_POINT_PARAMS:,}\n"
        f"Actual:   {point_parameter_count:,}"
    )

point_model_state = point_model.state_dict()

missing = sorted(
    set(point_model_state.keys())
    - set(point_state.keys())
)

unexpected = sorted(
    set(point_state.keys())
    - set(point_model_state.keys())
)

if missing or unexpected:

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    raise RuntimeError(
        "Point checkpoint structure does not match "
        "the exact PointGroundingModel."
    )

shape_mismatches = []

for key in point_model_state:

    if tuple(point_model_state[key].shape) != tuple(
        point_state[key].shape
    ):

        shape_mismatches.append(
            (
                key,
                tuple(point_model_state[key].shape),
                tuple(point_state[key].shape),
            )
        )

if shape_mismatches:

    for item in shape_mismatches:
        print(item)

    raise RuntimeError(
        "Point checkpoint tensor-shape mismatch."
    )

point_model.load_state_dict(
    point_state,
    strict=True,
)

print(
    "Point strict checkpoint load: PASS"
)

# ------------------------------------------------------------
# 5. VERIFY C3 MODEL STILL EXISTS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("VERIFYING EXACT C3 MODEL")
print("-" * 80)

if "c3_model" not in globals():

    raise RuntimeError(
        "c3_model is not present. "
        "Run the exact C3 checkpoint verification cell first."
    )

reference_model = c3_model

reference_model.eval()

reference_parameter_count = sum(
    p.numel()
    for p in reference_model.parameters()
)

EXPECTED_REFERENCE_PARAMS = 2_186_753

print(
    "Reference model:",
    reference_model.__class__
)

print(
    "Reference parameter count:",
    f"{reference_parameter_count:,}"
)

if reference_parameter_count != EXPECTED_REFERENCE_PARAMS:

    raise RuntimeError(
        "Reference C3 parameter count mismatch."
    )

# ------------------------------------------------------------
# 6. DEFINE UNIFIED INTERFACE
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. CREATE UNIFIED MODEL
# ------------------------------------------------------------

unified_grounding = UnifiedGroundingModel(
    point_model=point_model,
    reference_model=reference_model,
)

unified_grounding.eval()

print(
    "\nUnified model created:",
    unified_grounding.__class__.__name__
)

# ------------------------------------------------------------
# 8. COUNT PARAMS
# ------------------------------------------------------------

unified_parameter_count = sum(
    p.numel()
    for p in unified_grounding.parameters()
)

print(
    "Unified registered parameters:",
    f"{unified_parameter_count:,}"
)

print(
    "Expected sum:",
    f"{EXPECTED_POINT_PARAMS + EXPECTED_REFERENCE_PARAMS:,}"
)

if unified_parameter_count != (
    EXPECTED_POINT_PARAMS
    +
    EXPECTED_REFERENCE_PARAMS
):

    raise RuntimeError(
        "Unified model parameter registration mismatch."
    )

print(
    "Unified parameter registration: PASS"
)

# ------------------------------------------------------------
# 9. VERIFY MODE DISPATCH
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("UNIFIED MODE DISPATCH")
print("-" * 80)

# The model is intentionally tested only structurally here.
# Real-data forward verification is performed below if suitable
# batches already exist in the current kernel.

print(
    'Supported mode: "point"'
)

print(
    'Supported mode: "reference"'
)

print(
    "Point specialist:",
    unified_grounding.point_model.__class__.__name__
)

print(
    "Reference specialist:",
    unified_grounding.reference_model.__class__.__name__
)

# ------------------------------------------------------------
# 10. TRY EXISTING REAL VALIDATION BATCHES
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("SEARCHING FOR EXISTING REAL VALIDATION BATCHES")
print("-" * 80)

possible_point_loaders = [
    "point_val_loader",
    "point_validation_loader",
    "point_val_dataloader",
]

possible_reference_loaders = [
    "c3_val_loader",
    "reference_val_loader",
    "reference_validation_loader",
]

point_loader = None
reference_loader = None

for name in possible_point_loaders:

    obj = globals().get(name)

    if obj is not None:

        point_loader = obj
        print(
            "Point loader found:",
            name
        )
        break

for name in possible_reference_loaders:

    obj = globals().get(name)

    if obj is not None:

        reference_loader = obj
        print(
            "Reference loader found:",
            name
        )
        break

# ------------------------------------------------------------
# POINT REAL-BATCH TEST
# ------------------------------------------------------------

if point_loader is not None:

    print("\n" + "=" * 80)
    print("POINT REAL-BATCH UNIFIED TEST")
    print("=" * 80)

    point_batch = next(iter(point_loader))

    print(
        "Point batch type:",
        type(point_batch)
    )

    if isinstance(point_batch, dict):

        print(
            "Point batch keys:",
            list(point_batch.keys())
        )

    elif isinstance(point_batch, (list, tuple)):

        print(
            "Point batch length:",
            len(point_batch)
        )

    # Attempt common dictionary structure first.
    if isinstance(point_batch, dict):

        def get_first(d, names):

            for name in names:

                if name in d:
                    return d[name]

            return None

        s1 = get_first(
            point_batch,
            ["s1", "S1"]
        )

        s2 = get_first(
            point_batch,
            ["s2", "S2"]
        )

        text_embedding = get_first(
            point_batch,
            [
                "text",
                "text_embedding",
                "Text",
            ]
        )

        point = get_first(
            point_batch,
            [
                "point",
                "Point",
            ]
        )

        if (
            s1 is not None
            and
            s2 is not None
            and
            text_embedding is not None
            and
            point is not None
        ):

            with torch.no_grad():

                output = unified_grounding(
                    mode="point",
                    s1=s1,
                    s2=s2,
                    text_embedding=text_embedding,
                    point=point,
                )

            print(
                "Point unified forward: PASS"
            )

            if isinstance(output, dict):

                print(
                    "Point output keys:",
                    list(output.keys())
                )

        else:

            print(
                "Point loader found, but its batch schema "
                "does not match the expected names."
            )

else:

    print(
        "No existing point validation loader found."
    )


# ------------------------------------------------------------
# REFERENCE REAL-BATCH TEST
# ------------------------------------------------------------

if reference_loader is not None:

    print("\n" + "=" * 80)
    print("REFERENCE REAL-BATCH UNIFIED TEST")
    print("=" * 80)

    reference_batch = next(
        iter(reference_loader)
    )

    print(
        "Reference batch type:",
        type(reference_batch)
    )

    if isinstance(reference_batch, dict):

        print(
            "Reference batch keys:",
            list(reference_batch.keys())
        )

        def get_reference(d, names):

            for name in names:

                if name in d:
                    return d[name]

            return None

        s1 = get_reference(
            reference_batch,
            ["s1", "S1"]
        )

        s2 = get_reference(
            reference_batch,
            ["s2", "S2"]
        )

        text_embedding = get_reference(
            reference_batch,
            [
                "text",
                "text_embedding",
                "Text",
            ]
        )

        if (
            s1 is not None
            and
            s2 is not None
            and
            text_embedding is not None
        ):

            with torch.no_grad():

                output = unified_grounding(
                    mode="reference",
                    s1=s1,
                    s2=s2,
                    text_embedding=text_embedding,
                )

            print(
                "Reference unified forward: PASS"
            )

            if isinstance(output, dict):

                print(
                    "Reference output keys:",
                    list(output.keys())
                )

        else:

            print(
                "Reference loader found, but its batch schema "
                "does not match the expected names."
            )

else:

    print(
        "No existing reference validation loader found."
    )

# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("UNIFIED GROUNDING INTEGRATION STATUS")
print("=" * 80)

print(
    "Point specialist:     READY"
)

print(
    "Reference specialist: READY"
)

print(
    "Unified dispatcher:   READY"
)

print(
    "Registered params:",
    f"{unified_parameter_count:,}"
)

print(
    "\nNo training performed."
)

print(
    "No specialist checkpoint modified."
)

print(
    "No C3 architecture modified."
)

print(
    "\nUNIFIED POINT + REFERENCE GROUNDING INTEGRATION COMPLETE"
)