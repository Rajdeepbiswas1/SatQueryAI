import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path(r"D:\SatQueryAI")

GROUNDING_DIR = (
    PROJECT
    / "src"
    / "models"
    / "grounding_recovered"
)

POINT_VAL = (
    PROJECT
    / "data"
    / "processed"
    / "point_grounding_split"
    / "point_val.parquet"
)

REFERENCE_VAL = (
    PROJECT
    / "kaggle"
    / "reference_grounding_end_to_end"
    / "reference_val.parquet"
)

DOFA_DIR = (
    PROJECT
    / "kaggle"
    / "reference_grounding"
    / "dofa_features"
)

TEXT_DIR = (
    PROJECT
    / "kaggle"
    / "reference_grounding"
    / "grounding_text_embeddings"
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

sys.path.insert(
    0,
    str(GROUNDING_DIR),
)

from point_grounding_recovered import (
    PointGroundingModel,
)

from c3_reference_grounding_recovered import (
    C3QueryConditionedGrounding,
)

from unified_grounding_recovered import (
    UnifiedGroundingModel,
    UnifiedGroundingInference,
    SatQueryAIGroundingRouter,
)


# ============================================================
# CONFIGURATION
# ============================================================

POINT_SAMPLES = 4
REFERENCE_SAMPLES = 4


# ============================================================
# HELPERS
# ============================================================

def parse_four_values(value):
    """
    Convert a box-like value into float32 [4].

    Handles:
      numpy arrays
      Python lists/tuples
      strings such as:
        "[0.58 0.12, 1.0 0.81]"
        "[0.58, 0.12, 1.0, 0.81]"
    """

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        array = np.asarray(
            array,
            dtype=np.float32,
        ).reshape(-1)

    elif isinstance(value, np.ndarray):
        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

    elif isinstance(value, (list, tuple)):
        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

    else:
        numbers = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)"
            r"(?:[eE][-+]?\d+)?",
            str(value),
        )

        array = np.asarray(
            [float(x) for x in numbers],
            dtype=np.float32,
        )

    if array.size != 4:
        raise ValueError(
            f"Expected 4 numeric values, got "
            f"{array.size} from {value!r}"
        )

    return array


def parse_point(value):
    if torch.is_tensor(value):
        array = (
            value.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

    elif isinstance(value, np.ndarray):
        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

    elif isinstance(value, (list, tuple)):
        array = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

    else:
        numbers = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)"
            r"(?:[eE][-+]?\d+)?",
            str(value),
        )

        array = np.asarray(
            [float(x) for x in numbers],
            dtype=np.float32,
        )

    if array.size != 2:
        raise ValueError(
            f"Expected 2 numeric values for point, "
            f"got {array.size} from {value!r}"
        )

    return array.astype(
        np.float32
    )

def extract_reference_phrase(text):
    match = re.search(
        r"<ref>(.*?)</ref>",
        str(text),
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return str(text).strip()


def iou_xyxy(pred, target):
    pred = np.asarray(
        pred,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    ix1 = max(
        float(pred[0]),
        float(target[0]),
    )

    iy1 = max(
        float(pred[1]),
        float(target[1]),
    )

    ix2 = min(
        float(pred[2]),
        float(target[2]),
    )

    iy2 = min(
        float(pred[3]),
        float(target[3]),
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    pred_area = max(
        0.0,
        float(pred[2] - pred[0]),
    ) * max(
        0.0,
        float(pred[3] - pred[1]),
    )

    target_area = max(
        0.0,
        float(target[2] - target[0]),
    ) * max(
        0.0,
        float(target[3] - target[1]),
    )

    union = (
        pred_area
        +
        target_area
        -
        intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def point_inside_box(point, box):
    x, y = (
        float(point[0]),
        float(point[1]),
    )

    x1, y1, x2, y2 = (
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def finite_numpy(value):
    array = np.asarray(value)

    return bool(
        np.isfinite(array).all()
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print(
    "SATQUERYAI — PHASE B REAL GROUNDING TEST"
)
print("=" * 72)
print(
    "Using persisted real validation data "
    "and cached DOFA/text features."
)
print(
    "No retraining."
)
print()


# ============================================================
# 1. LOAD VALIDATION DATA
# ============================================================

print(
    "[1/8] Loading real validation data..."
)

point_df = pd.read_parquet(
    POINT_VAL,
    engine="pyarrow",
)

reference_df = pd.read_parquet(
    REFERENCE_VAL,
    engine="pyarrow",
)

point_df = point_df.head(
    POINT_SAMPLES
).copy()

reference_df = reference_df.head(
    REFERENCE_SAMPLES
).copy()

print(
    f"Point samples: {len(point_df)}"
)

print(
    f"Reference samples: {len(reference_df)}"
)

print()


# ============================================================
# 2. BUILD TEXT CACHE INDEX
# ============================================================

print(
    "[2/8] Resolving text-cache mappings..."
)

required_source_rows = set(
    int(x)
    for x in point_df[
        "annotation_index"
    ].tolist()
)

required_source_rows.update(
    int(x)
    for x in reference_df[
        "original_row_index"
    ].tolist()
)

text_rows = {}

text_shards = sorted(
    TEXT_DIR.glob(
        "grounding_text_shard_*.pt"
    )
)

if not text_shards:
    raise RuntimeError(
        f"No text shards found in:\n{TEXT_DIR}"
    )

for shard_path in text_shards:

    shard = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    source_indices = shard[
        "source_row_index"
    ]

    for local_index, source_index in enumerate(
        source_indices.tolist()
    ):

        source_index = int(
            source_index
        )

        if source_index not in required_source_rows:
            continue

        text_rows[source_index] = {
            "shard": shard_path,
            "local_index": local_index,
            "patch_id": shard["patch_ids"][
                local_index
            ],
            "text_embedding": shard[
                "text_embeddings"
            ][local_index].clone(),
            "point": shard[
                "points"
            ][local_index].clone(),
            "target_box": shard[
                "target_boxes"
            ][local_index].clone(),
            "mode": int(
                shard[
                    "modes"
                ][local_index].item()
            ),
            "category": shard[
                "categories"
            ][local_index],
        }

    if (
        required_source_rows
        and
        required_source_rows.issubset(
            set(text_rows.keys())
        )
    ):
        break


missing_source_rows = (
    required_source_rows
    -
    set(text_rows.keys())
)

if missing_source_rows:
    raise RuntimeError(
        "Could not resolve required "
        f"text-cache source rows: "
        f"{sorted(missing_source_rows)}"
    )

print(
    f"Resolved text rows: "
    f"{len(text_rows)}"
)

print()


# ============================================================
# 3. VERIFY POINT / REFERENCE MAPPING
# ============================================================

print(
    "[3/8] Verifying source-row mappings..."
)

point_records = []
reference_records = []

for _, row in point_df.iterrows():

    source_index = int(
        row["annotation_index"]
    )

    cache = text_rows[
        source_index
    ]

    parquet_patch = str(
        row["patch_id"]
    )

    cache_patch = str(
        cache["patch_id"]
    )

    if parquet_patch != cache_patch:
        raise RuntimeError(
            "Point patch mismatch:\n"
            f"annotation_index={source_index}\n"
            f"parquet={parquet_patch}\n"
            f"cache={cache_patch}"
        )

    parquet_point = parse_point(
        row["point_xy"]
    )

    parquet_box = parse_four_values(
        row["target_box"]
    )

    cache_point = (
        cache["point"]
        .numpy()
        .astype(np.float32)
    )

    cache_box = (
        cache["target_box"]
        .numpy()
        .astype(np.float32)
    )

    if not np.allclose(
        parquet_point,
        cache_point,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Point mismatch between "
            "Parquet and text cache:\n"
            f"source_row={source_index}\n"
            f"parquet={parquet_point}\n"
            f"cache={cache_point}"
        )

    if not np.allclose(
        parquet_box,
        cache_box,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Point target-box mismatch "
            "between Parquet and text cache:\n"
            f"source_row={source_index}\n"
            f"parquet={parquet_box}\n"
            f"cache={cache_box}"
        )

    point_records.append(
        {
            "source_row_index": source_index,
            "patch_id": parquet_patch,
            "text_embedding": cache[
                "text_embedding"
            ],
            "point": parquet_point,
            "target_box": parquet_box,
        }
    )


for _, row in reference_df.iterrows():

    source_index = int(
        row["original_row_index"]
    )

    cache = text_rows[
        source_index
    ]

    parquet_patch = str(
        row["patch_id"]
    )

    cache_patch = str(
        cache["patch_id"]
    )

    if parquet_patch != cache_patch:
        raise RuntimeError(
            "Reference patch mismatch:\n"
            f"source_row={source_index}\n"
            f"parquet={parquet_patch}\n"
            f"cache={cache_patch}"
        )

    parquet_box = parse_four_values(
        row["output"]
    )

    cache_box = (
        cache["target_box"]
        .numpy()
        .astype(np.float32)
    )

    if not np.allclose(
        parquet_box,
        cache_box,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Reference target-box mismatch "
            "between Parquet and text cache:\n"
            f"source_row={source_index}\n"
            f"parquet={parquet_box}\n"
            f"cache={cache_box}"
        )

    reference_records.append(
        {
            "source_row_index": source_index,
            "patch_id": parquet_patch,
            "text_embedding": cache[
                "text_embedding"
            ],
            "reference": extract_reference_phrase(
                row["input"]
            ),
            "target_box": parquet_box,
            "input": str(
                row["input"]
            ),
        }
    )


print(
    "Point Parquet ↔ text-cache mapping: PASS"
)

print(
    "Reference Parquet ↔ text-cache mapping: PASS"
)

print()


# ============================================================
# 4. LOAD REQUIRED DOFA FEATURES
# ============================================================

print(
    "[4/8] Resolving DOFA features by patch_id..."
)

required_patches = {
    record["patch_id"]
    for record in (
        point_records
        +
        reference_records
    )
}

dofa_records = {}

dofa_shards = sorted(
    DOFA_DIR.glob(
        "shard_*.pt"
    )
)

if not dofa_shards:
    raise RuntimeError(
        f"No DOFA shards found in:\n{DOFA_DIR}"
    )

for shard_path in dofa_shards:

    if required_patches.issubset(
        set(dofa_records.keys())
    ):
        break

    shard = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    patch_ids = shard[
        "patch_ids"
    ]

    for local_index, patch_id in enumerate(
        patch_ids
    ):

        patch_id = str(
            patch_id
        )

        if patch_id not in required_patches:
            continue

        dofa_records[patch_id] = {
            "shard": shard_path,
            "local_index": local_index,
            "s1": shard[
                "s1"
            ][local_index].clone(),
            "s2": shard[
                "s2"
            ][local_index].clone(),
        }


missing_patches = (
    required_patches
    -
    set(dofa_records.keys())
)

if missing_patches:
    raise RuntimeError(
        "Could not resolve required "
        f"DOFA patches:\n"
        +
        "\n".join(
            sorted(missing_patches)
        )
    )

print(
    f"Resolved DOFA patches: "
    f"{len(dofa_records)}"
)

for patch_id, record in dofa_records.items():

    if tuple(record["s1"].shape) != (
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected S1 shape for "
            f"{patch_id}: "
            f"{tuple(record['s1'].shape)}"
        )

    if tuple(record["s2"].shape) != (
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected S2 shape for "
            f"{patch_id}: "
            f"{tuple(record['s2'].shape)}"
        )

print(
    "DOFA feature shapes: PASS"
)

print()


# ============================================================
# 5. BUILD REAL BATCHES
# ============================================================

print(
    "[5/8] Building real inference batches..."
)

point_s1 = torch.stack(
    [
        dofa_records[
            record["patch_id"]
        ]["s1"].float()
        for record in point_records
    ]
)

point_s2 = torch.stack(
    [
        dofa_records[
            record["patch_id"]
        ]["s2"].float()
        for record in point_records
    ]
)

point_text = torch.stack(
    [
        record["text_embedding"].float()
        for record in point_records
    ]
)

point_xy = torch.tensor(
    np.stack(
        [
            record["point"]
            for record in point_records
        ]
    ),
    dtype=torch.float32,
)

point_gt = torch.tensor(
    np.stack(
        [
            record["target_box"]
            for record in point_records
        ]
    ),
    dtype=torch.float32,
)


reference_s1 = torch.stack(
    [
        dofa_records[
            record["patch_id"]
        ]["s1"].float()
        for record in reference_records
    ]
)

reference_s2 = torch.stack(
    [
        dofa_records[
            record["patch_id"]
        ]["s2"].float()
        for record in reference_records
    ]
)

reference_text = torch.stack(
    [
        record["text_embedding"].float()
        for record in reference_records
    ]
)

reference_gt = torch.tensor(
    np.stack(
        [
            record["target_box"]
            for record in reference_records
        ]
    ),
    dtype=torch.float32,
)

print(
    f"Point S1: {tuple(point_s1.shape)}"
)

print(
    f"Point S2: {tuple(point_s2.shape)}"
)

print(
    f"Point text: {tuple(point_text.shape)}"
)

print(
    f"Point coordinates: "
    f"{tuple(point_xy.shape)}"
)

print(
    f"Point target boxes: "
    f"{tuple(point_gt.shape)}"
)

print(
    f"Reference S1: "
    f"{tuple(reference_s1.shape)}"
)

print(
    f"Reference S2: "
    f"{tuple(reference_s2.shape)}"
)

print(
    f"Reference text: "
    f"{tuple(reference_text.shape)}"
)

print(
    f"Reference target boxes: "
    f"{tuple(reference_gt.shape)}"
)

expected_visual_shape = (
    len(point_records),
    196,
    768,
)

if tuple(point_s1.shape) != expected_visual_shape:
    raise RuntimeError(
        "Unexpected real Point S1 batch shape."
    )

print(
    "Real batch construction: PASS"
)

print()


# ============================================================
# 6. LOAD VERIFIED MODELS
# ============================================================

print(
    "[6/8] Loading verified models..."
)

point_model = PointGroundingModel()
reference_model = (
    C3QueryConditionedGrounding()
)

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
    point_ckpt[
        "model_state_dict"
    ],
    strict=True,
)

reference_model.load_state_dict(
    reference_ckpt[
        "model_state_dict"
    ],
    strict=True,
)

point_model.eval()
reference_model.eval()

point_params = sum(
    p.numel()
    for p in point_model.parameters()
)

reference_params = sum(
    p.numel()
    for p in reference_model.parameters()
)

if point_params != 1_531_909:
    raise RuntimeError(
        f"Unexpected Point parameter count: "
        f"{point_params:,}"
    )

if reference_params != 2_186_753:
    raise RuntimeError(
        f"Unexpected C3 parameter count: "
        f"{reference_params:,}"
    )

unified_model = UnifiedGroundingModel(
    point_model,
    reference_model,
)

unified_params = sum(
    p.numel()
    for p in unified_model.parameters()
)

if unified_params != 3_718_662:
    raise RuntimeError(
        f"Unexpected Unified parameter count: "
        f"{unified_params:,}"
    )

unified_model.eval()

inference = UnifiedGroundingInference(
    unified_model
)

router = SatQueryAIGroundingRouter(
    inference
)

print(
    "Point checkpoint: PASS"
)

print(
    "C3 checkpoint: PASS"
)

print(
    "Unified model: PASS "
    f"({unified_params:,} parameters)"
)

print(
    "Router: PASS"
)

print()


# ============================================================
# 7. RUN REAL POINT / REFERENCE ROUTES
# ============================================================

print(
    "[7/8] Running real Point and Reference routes..."
)

# ------------------------------------------------------------
# Point
# ------------------------------------------------------------

point_start = time.perf_counter()

point_result = router.dispatch(
    mode="point",
    s1=point_s1,
    s2=point_s2,
    text_embedding=point_text,
    point=point_xy,
    query=(
        "Generate a bounding box around "
        "the land cover class instance at "
        "the supplied point."
    ),
    reference=None,
    patch_id=[
        record["patch_id"]
        for record in point_records
    ],
)

point_elapsed = (
    time.perf_counter()
    -
    point_start
)

# ------------------------------------------------------------
# Reference
# ------------------------------------------------------------

reference_start = time.perf_counter()

reference_result = router.dispatch(
    mode="reference",
    s1=reference_s1,
    s2=reference_s2,
    text_embedding=reference_text,
    point=None,
    query=[
        record["input"]
        for record in reference_records
    ],
    reference=[
        record["reference"]
        for record in reference_records
    ],
    patch_id=[
        record["patch_id"]
        for record in reference_records
    ],
)

reference_elapsed = (
    time.perf_counter()
    -
    reference_start
)

print(
    f"Point real route time: "
    f"{point_elapsed * 1000:.2f} ms"
)

print(
    f"Reference real route time: "
    f"{reference_elapsed * 1000:.2f} ms"
)

print()


# ============================================================
# 8. VALIDATE REAL RESULTS
# ============================================================

print(
    "[8/8] Validating real grounding results..."
)

# ------------------------------------------------------------
# Point evidence
# ------------------------------------------------------------

point_bbox = np.asarray(
    point_result["bbox"],
    dtype=np.float32,
)

point_heatmap = np.asarray(
    point_result["heatmap"],
    dtype=np.float32,
)

point_confidence = np.asarray(
    point_result[
        "evidence_confidence"
    ],
    dtype=np.float32,
)

point_peak = np.asarray(
    point_result[
        "peak_probability"
    ],
    dtype=np.float32,
)

point_entropy = np.asarray(
    point_result[
        "spatial_entropy"
    ],
    dtype=np.float32,
)

if point_bbox.shape != (
    POINT_SAMPLES,
    4,
):
    raise RuntimeError(
        f"Unexpected Point bbox shape: "
        f"{point_bbox.shape}"
    )

if point_heatmap.shape[0] != (
    POINT_SAMPLES
):
    raise RuntimeError(
        "Unexpected Point heatmap batch."
    )

if not finite_numpy(
    point_bbox
):
    raise RuntimeError(
        "Point bbox contains "
        "non-finite values."
    )

if not finite_numpy(
    point_heatmap
):
    raise RuntimeError(
        "Point heatmap contains "
        "non-finite values."
    )

point_ious = []

point_containment = 0

for index in range(
    POINT_SAMPLES
):

    pred_box = point_bbox[
        index
    ]

    gt_box = point_records[
        index
    ]["target_box"]

    point = point_records[
        index
    ]["point"]

    current_iou = iou_xyxy(
        pred_box,
        gt_box,
    )

    point_ious.append(
        current_iou
    )

    if point_inside_box(
        point,
        pred_box,
    ):
        point_containment += 1

# ------------------------------------------------------------
# Reference evidence
# ------------------------------------------------------------

reference_bbox = np.asarray(
    reference_result["bbox"],
    dtype=np.float32,
)

reference_heatmap = np.asarray(
    reference_result["heatmap"],
    dtype=np.float32,
)

reference_confidence = np.asarray(
    reference_result[
        "evidence_confidence"
    ],
    dtype=np.float32,
)

reference_peak = np.asarray(
    reference_result[
        "peak_probability"
    ],
    dtype=np.float32,
)

reference_entropy = np.asarray(
    reference_result[
        "spatial_entropy"
    ],
    dtype=np.float32,
)

if reference_bbox.shape != (
    REFERENCE_SAMPLES,
    4,
):
    raise RuntimeError(
        "Unexpected Reference bbox shape: "
        f"{reference_bbox.shape}"
    )

if reference_heatmap.shape[0] != (
    REFERENCE_SAMPLES
):
    raise RuntimeError(
        "Unexpected Reference heatmap batch."
    )

if not finite_numpy(
    reference_bbox
):
    raise RuntimeError(
        "Reference bbox contains "
        "non-finite values."
    )

if not finite_numpy(
    reference_heatmap
):
    raise RuntimeError(
        "Reference heatmap contains "
        "non-finite values."
    )

reference_ious = []

for index in range(
    REFERENCE_SAMPLES
):

    pred_box = reference_bbox[
        index
    ]

    gt_box = reference_records[
        index
    ]["target_box"]

    current_iou = iou_xyxy(
        pred_box,
        gt_box,
    )

    reference_ious.append(
        current_iou
    )


# ============================================================
# REPORT
# ============================================================

print(
    "REAL POINT ROUTE: PASS"
)

print(
    "  Specialist:",
    point_result[
        "execution_trace"
    ][
        "selected_specialist"
    ],
)

print(
    "  IoUs:",
    [
        round(x, 4)
        for x in point_ious
    ],
)

print(
    "  Mean IoU:",
    round(
        float(
            np.mean(
                point_ious
            )
        ),
        6,
    ),
)

print(
    "  Point containment:",
    f"{point_containment}/{POINT_SAMPLES}",
)

print(
    "  Evidence confidence:",
    [
        round(float(x), 4)
        for x in point_confidence
    ],
)

print(
    "  Peak probability:",
    [
        round(float(x), 4)
        for x in point_peak
    ],
)

print(
    "  Spatial entropy:",
    [
        round(float(x), 4)
        for x in point_entropy
    ],
)

print(
    "  Bboxes:",
    point_bbox.tolist(),
)

print()

print(
    "REAL REFERENCE ROUTE: PASS"
)

print(
    "  Specialist:",
    reference_result[
        "execution_trace"
    ][
        "selected_specialist"
    ],
)

print(
    "  IoUs:",
    [
        round(x, 4)
        for x in reference_ious
    ],
)

print(
    "  Mean IoU:",
    round(
        float(
            np.mean(
                reference_ious
            )
        ),
        6,
    ),
)

print(
    "  Evidence confidence:",
    [
        round(float(x), 4)
        for x in reference_confidence
    ],
)

print(
    "  Peak probability:",
    [
        round(float(x), 4)
        for x in reference_peak
    ],
)

print(
    "  Spatial entropy:",
    [
        round(float(x), 4)
        for x in reference_entropy
    ],
)

print(
    "  Bboxes:",
    reference_bbox.tolist(),
)

print()


# ============================================================
# EXECUTION TRACE VALIDATION
# ============================================================

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

for name, result in (
    ("POINT", point_result),
    ("REFERENCE", reference_result),
):

    trace = result.get(
        "execution_trace"
    )

    if not isinstance(
        trace,
        dict,
    ):
        raise RuntimeError(
            f"{name} execution trace missing."
        )

    missing = (
        required_trace
        -
        set(trace.keys())
    )

    if missing:
        raise RuntimeError(
            f"{name} trace missing: "
            f"{sorted(missing)}"
        )

    if trace["status"] != "success":
        raise RuntimeError(
            f"{name} trace status: "
            f"{trace['status']}"
        )

print(
    "Execution traces: PASS"
)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 72)
print(
    "PHASE B REAL LOCAL GROUNDING TEST: COMPLETE"
)
print("=" * 72)

print()
print(
    "Real Point data → DOFA → "
    "PointGroundingModel → "
    "Unified Evidence: PASS"
)

print(
    "Real Reference data → DOFA → "
    "C3QueryConditionedGrounding → "
    "Unified Evidence: PASS"
)

print()
print(
    "Parquet/cache mapping: PASS"
)

print(
    "DOFA feature retrieval: PASS"
)

print(
    "Real Point inference: PASS"
)

print(
    "Real Reference inference: PASS"
)

print(
    "Evidence validation: PASS"
)

print(
    "Execution trace validation: PASS"
)

print("=" * 72)