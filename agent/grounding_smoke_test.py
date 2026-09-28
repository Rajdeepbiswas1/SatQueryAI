# =============================================================================
# SATQUERYAI — REAL-DATA AGENT GROUNDING SMOKE TEST
# =============================================================================
#
# Purpose:
#   Verify the complete runtime path:
#
#   Natural-language query
#       ->
#   Agent query analysis
#       ->
#   Capability registry
#       ->
#   Grounding tool adapter
#       ->
#   Verified grounding router
#       ->
#   Point / Reference specialist
#       ->
#   Standardized evidence
#       ->
#   Agent execution trace
#
# No training.
# No checkpoint modification.
# No model modification.
#
# Real local data only.
# =============================================================================

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from agent.bootstrap import create_runtime_agent


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

POINT_VAL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "point_grounding_split"
    / "point_val.parquet"
)

REFERENCE_VAL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

DOFA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grounding_text_embeddings"
    / "grounding_text_embedding_shards"
)


# =============================================================================
# HELPERS
# =============================================================================

def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def to_float_tensor(value, shape=None) -> torch.Tensor:
    """
    Convert a dataframe object/list/string/numpy value into a float tensor.
    """
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(
            value,
            dtype=torch.float32,
        )
    else:
        numbers = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)",
            str(value),
        )

        if not numbers:
            raise ValueError(
                f"Could not parse numeric value:\n{value}"
            )

        tensor = torch.tensor(
            [float(x) for x in numbers],
            dtype=torch.float32,
        )

    if shape is not None:
        tensor = tensor.reshape(shape)

    return tensor


def extract_reference(query: str):
    """
    Extract <ref>...</ref> when present.
    """
    match = re.search(
        r"<ref>(.*?)</ref>",
        str(query),
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return None


def find_dofa_features(patch_id: str):
    """
    Locate one patch in the verified DOFA shards without loading the complete
    5+ GB cache into RAM.
    """
    patch_id = str(patch_id)

    files = sorted(
        DOFA_DIR.glob("*.pt")
    )

    if not files:
        raise FileNotFoundError(
            f"No DOFA shards found in:\n{DOFA_DIR}"
        )

    print(
        f"Searching DOFA cache for patch:\n  {patch_id}"
    )

    for shard_index, path in enumerate(files, start=1):

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        patch_ids = [
            str(x.item() if torch.is_tensor(x) else x)
            for x in data["patch_ids"]
        ]

        for row_index, candidate in enumerate(patch_ids):

            if candidate == patch_id:

                s1 = data["s1"][row_index].float()
                s2 = data["s2"][row_index].float()

                print(
                    f"DOFA match: shard {shard_index}/{len(files)}"
                )

                return (
                    s1,
                    s2,
                )

    raise KeyError(
        f"Patch not found in DOFA cache:\n{patch_id}"
    )


def find_text_embedding(source_row_index: int):
    """
    Locate the exact grounding text embedding by source_row_index.

    IMPORTANT:
        We intentionally use source_row_index rather than the Point
        dataframe's grounding_row_index.
    """
    source_row_index = int(
        source_row_index
    )

    files = sorted(
        TEXT_DIR.glob(
            "grounding_text_shard_*.pt"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No grounding text shards found in:\n{TEXT_DIR}"
        )

    print(
        "Searching text cache for source_row_index:",
        source_row_index,
    )

    for shard_index, path in enumerate(files, start=1):

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        source_indices = data["source_row_index"]

        for local_index, row_index in enumerate(
            source_indices.tolist()
        ):

            if int(row_index) == source_row_index:

                embedding = (
                    data["text_embeddings"][local_index]
                    .float()
                )

                print(
                    f"Text match: shard "
                    f"{shard_index}/{len(files)}"
                )

                return embedding

    raise KeyError(
        "Text embedding not found for "
        f"source_row_index={source_row_index}"
    )


def box_iou(box_a, box_b) -> float:
    """
    Compute IoU for normalized [x1,y1,x2,y2] boxes.
    """
    a = np.asarray(box_a, dtype=np.float32).reshape(4)
    b = np.asarray(box_b, dtype=np.float32).reshape(4)

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    inter_area = (
        inter_w * inter_h
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - inter_area
    )

    if union <= 0.0:
        return 0.0

    return float(
        inter_area / union
    )


def get_bbox_from_evidence(evidence):
    """
    Read the standardized grounding bbox.
    """
    if not isinstance(evidence, dict):
        raise TypeError(
            "Grounding evidence must be a dict."
        )

    if "bbox" not in evidence:
        raise KeyError(
            "Standardized evidence is missing 'bbox'."
        )

    bbox = np.asarray(
        evidence["bbox"],
        dtype=np.float32,
    )

    # Expected for batch size 1:
    #   [1,4]
    # or:
    #   [4]
    bbox = bbox.reshape(-1)

    if bbox.size != 4:
        raise ValueError(
            f"Unexpected bbox shape: {bbox.shape}"
        )

    return bbox


def print_trace(trace):
    print("\nEXECUTION TRACE")

    required = [
        "query",
        "normalized_query",
        "detected_task",
        "grounding_mode",
        "selected_tool",
        "capability_available",
        "status",
        "routing_confidence",
        "matched_signals",
        "validation_ms",
        "execution_ms",
        "total_ms",
    ]

    for key in required:

        if key not in trace:
            raise KeyError(
                f"Execution trace missing field: {key}"
            )

        print(
            f"  {key}: {trace[key]}"
        )


# =============================================================================
# REAL POINT TEST
# =============================================================================

def run_point_test(agent):
    print()
    print("=" * 80)
    print("REAL POINT GROUNDING — AGENT EXECUTION")
    print("=" * 80)

    df = pd.read_parquet(
        POINT_VAL
    )

    if df.empty:
        raise RuntimeError(
            "Point validation dataframe is empty."
        )

    row = df.iloc[0]

    patch_id = str(
        row["patch_id"]
    )

    query = str(
        row["input"]
    )

    point = to_float_tensor(
        row["point_xy"],
        shape=(1, 2),
    )

    target_box = to_float_tensor(
        row["target_box"],
        shape=(4,),
    )

    # -------------------------------------------------------------------------
    # IMPORTANT MAPPING
    #
    # The Point parquet contains both:
    #   grounding_row_index
    #   annotation_index
    #
    # The verified text-cache mapping is based on:
    #   annotation_index <-> source_row_index
    #
    # Therefore we intentionally use annotation_index here.
    # -------------------------------------------------------------------------

    annotation_index = int(
        row["annotation_index"]
    )

    print("Dataset row:")
    print("  annotation_index:", annotation_index)
    print("  grounding_row_index:", int(row["grounding_row_index"]))
    print("  patch_id:", patch_id)
    print("  point:", point.tolist())
    print("  target_box:", target_box.tolist())
    print("  query:", query)

    s1, s2 = find_dofa_features(
        patch_id
    )

    text_embedding = find_text_embedding(
        annotation_index
    )

    s1 = s1.unsqueeze(0)
    s2 = s2.unsqueeze(0)
    text_embedding = text_embedding.unsqueeze(0)

    # -------------------------------------------------------------------------
    # Shape checks
    # -------------------------------------------------------------------------

    print("\nINPUT SHAPES")

    print("  S1:", tuple(s1.shape))
    print("  S2:", tuple(s2.shape))
    print("  Text:", tuple(text_embedding.shape))
    print("  Point:", tuple(point.shape))

    if tuple(s1.shape) != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S1 shape: {tuple(s1.shape)}"
        )

    if tuple(s2.shape) != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(s2.shape)}"
        )

    if tuple(text_embedding.shape) != (1, 768):
        raise RuntimeError(
            f"Unexpected text shape: {tuple(text_embedding.shape)}"
        )

    if tuple(point.shape) != (1, 2):
        raise RuntimeError(
            f"Unexpected point shape: {tuple(point.shape)}"
        )

    # -------------------------------------------------------------------------
    # Finite checks
    # -------------------------------------------------------------------------

    if not torch.isfinite(s1).all():
        raise RuntimeError("S1 contains non-finite values.")

    if not torch.isfinite(s2).all():
        raise RuntimeError("S2 contains non-finite values.")

    if not torch.isfinite(text_embedding).all():
        raise RuntimeError(
            "Text embedding contains non-finite values."
        )

    if not torch.isfinite(point).all():
        raise RuntimeError(
            "Point contains non-finite values."
        )

    # -------------------------------------------------------------------------
    # Point containment in ground truth
    # -------------------------------------------------------------------------

    px, py = (
        point[0].tolist()
    )

    x1, y1, x2, y2 = (
        target_box.tolist()
    )

    if not (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    ):
        raise RuntimeError(
            "Dataset point is not inside its target box."
        )

    print("\nGROUND-TRUTH POINT CONTAINMENT: PASS")

    # -------------------------------------------------------------------------
    # REAL AGENT EXECUTION
    # -------------------------------------------------------------------------

    start = time.perf_counter()

    result = agent.execute(
        query=query,
        s1=s1,
        s2=s2,
        text_embedding=text_embedding,
        point=point,
        reference=None,
        patch_id=patch_id,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    # -------------------------------------------------------------------------
    # Result checks
    # -------------------------------------------------------------------------

    if result.get("answer_status") != "success":
        raise RuntimeError(
            "Point agent execution failed:\n"
            f"{result}"
        )

    analysis = result["analysis"]
    evidence = result["evidence"]
    trace = result["execution_trace"]

    print("\nAGENT ANALYSIS")
    print("  task:", analysis["task"])
    print("  grounding_mode:", analysis["grounding_mode"])
    print("  routing_confidence:", analysis["confidence"])

    if analysis["task"] != "grounding":
        raise RuntimeError(
            f"Expected grounding task, got {analysis['task']}"
        )

    if analysis["grounding_mode"] != "point":
        raise RuntimeError(
            "Expected point grounding mode."
        )

    bbox = get_bbox_from_evidence(
        evidence
    )

    predicted_heatmap = evidence.get(
        "heatmap"
    )

    if predicted_heatmap is None:
        raise RuntimeError(
            "Standardized evidence missing heatmap."
        )

    print("\nSTANDARDIZED EVIDENCE")
    print("  bbox:", bbox.tolist())
    print(
        "  heatmap shape:",
        np.asarray(predicted_heatmap).shape,
    )
    print(
        "  evidence_confidence:",
        evidence.get("evidence_confidence"),
    )
    print(
        "  peak_probability:",
        evidence.get("peak_probability"),
    )
    print(
        "  spatial_entropy:",
        evidence.get("spatial_entropy"),
    )

    # -------------------------------------------------------------------------
    # Point containment in predicted box
    # -------------------------------------------------------------------------

    pred_x1, pred_y1, pred_x2, pred_y2 = (
        bbox.tolist()
    )

    point_contained = (
        pred_x1 <= px <= pred_x2
        and
        pred_y1 <= py <= pred_y2
    )

    if not point_contained:
        raise RuntimeError(
            "Predicted grounding box does not contain the supplied point."
        )

    print(
        "\nPREDICTED BOX CONTAINS POINT: PASS"
    )

    print_trace(
        trace
    )

    if trace["selected_tool"] != "GroundingModalityDispatcher":
        raise RuntimeError(
            "Unexpected outer grounding tool."
        )

    inner_trace = evidence.get("execution_trace", {})

    if inner_trace.get("selected_specialist") != "PointGroundingModel":
        raise RuntimeError(
            "Unexpected Point grounding specialist."
        )

    print(
        "\nGROUNDING DISPATCH TRACE: PASS"
    )

    print(
        "  outer tool:",
        trace["selected_tool"]
    )

    print(
        "  inner specialist:",
        inner_trace.get("selected_specialist")
    )

    if trace["status"] != "success":
        raise RuntimeError(
            "Point execution trace does not report success."
        )

    print(
        f"\nExternal wall-clock execution time: "
        f"{elapsed:.2f} ms"
    )

    print("\nREAL POINT AGENT EXECUTION: PASS")


# =============================================================================
# REAL REFERENCE TEST
# =============================================================================

def run_reference_test(agent):
    print()
    print("=" * 80)
    print("REAL REFERENCE GROUNDING — AGENT EXECUTION")
    print("=" * 80)

    df = pd.read_parquet(
        REFERENCE_VAL
    )

    if df.empty:
        raise RuntimeError(
            "Reference validation dataframe is empty."
        )

    row = df.iloc[0]

    patch_id = str(
        row["patch_id"]
    )

    query = str(
        row["input"]
    )

    # Reference dataset uses original_row_index for its text mapping.
    source_row_index = int(
        row["original_row_index"]
    )

    target_box = to_float_tensor(
        row["output"],
        shape=(4,),
    )

    reference = extract_reference(
        query
    )

    if not reference:
        raise RuntimeError(
            "Could not extract a reference phrase from the validation query."
        )

    print("Dataset row:")
    print("  original_row_index:", source_row_index)
    print("  patch_id:", patch_id)
    print("  reference:", reference)
    print("  target_box:", target_box.tolist())
    print("  query:", query)

    s1, s2 = find_dofa_features(
        patch_id
    )

    text_embedding = find_text_embedding(
        source_row_index
    )

    s1 = s1.unsqueeze(0)
    s2 = s2.unsqueeze(0)
    text_embedding = text_embedding.unsqueeze(0)

    print("\nINPUT SHAPES")
    print("  S1:", tuple(s1.shape))
    print("  S2:", tuple(s2.shape))
    print("  Text:", tuple(text_embedding.shape))

    if tuple(s1.shape) != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S1 shape: {tuple(s1.shape)}"
        )

    if tuple(s2.shape) != (1, 196, 768):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(s2.shape)}"
        )

    if tuple(text_embedding.shape) != (1, 768):
        raise RuntimeError(
            f"Unexpected text shape: {tuple(text_embedding.shape)}"
        )

    if not torch.isfinite(s1).all():
        raise RuntimeError("S1 contains non-finite values.")

    if not torch.isfinite(s2).all():
        raise RuntimeError("S2 contains non-finite values.")

    if not torch.isfinite(text_embedding).all():
        raise RuntimeError(
            "Text embedding contains non-finite values."
        )

    # -------------------------------------------------------------------------
    # REAL AGENT EXECUTION
    # -------------------------------------------------------------------------

    start = time.perf_counter()

    result = agent.execute(
        query=query,
        s1=s1,
        s2=s2,
        text_embedding=text_embedding,
        point=None,
        reference=reference,
        patch_id=patch_id,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    if result.get("answer_status") != "success":
        raise RuntimeError(
            "Reference agent execution failed:\n"
            f"{result}"
        )

    analysis = result["analysis"]
    evidence = result["evidence"]
    trace = result["execution_trace"]

    print("\nAGENT ANALYSIS")
    print("  task:", analysis["task"])
    print("  grounding_mode:", analysis["grounding_mode"])
    print("  routing_confidence:", analysis["confidence"])

    if analysis["task"] != "grounding":
        raise RuntimeError(
            f"Expected grounding task, got {analysis['task']}"
        )

    if analysis["grounding_mode"] != "reference":
        raise RuntimeError(
            "Expected reference grounding mode."
        )

    bbox = get_bbox_from_evidence(
        evidence
    )

    predicted_heatmap = evidence.get(
        "heatmap"
    )

    if predicted_heatmap is None:
        raise RuntimeError(
            "Standardized evidence missing heatmap."
        )

    print("\nSTANDARDIZED EVIDENCE")
    print("  bbox:", bbox.tolist())
    print(
        "  heatmap shape:",
        np.asarray(predicted_heatmap).shape,
    )
    print(
        "  evidence_confidence:",
        evidence.get("evidence_confidence"),
    )
    print(
        "  peak_probability:",
        evidence.get("peak_probability"),
    )
    print(
        "  spatial_entropy:",
        evidence.get("spatial_entropy"),
    )

    # -------------------------------------------------------------------------
    # Real prediction vs ground truth
    # -------------------------------------------------------------------------

    iou = box_iou(
        bbox,
        target_box.numpy(),
    )

    print(
        f"\nREAL REFERENCE IoU: {iou:.6f}"
    )

    print_trace(
        trace
    )

    if trace["selected_tool"] != "GroundingModalityDispatcher":
        raise RuntimeError(
            "Unexpected outer grounding tool."
        )

    inner_trace = evidence.get("execution_trace", {})

    if inner_trace.get("selected_specialist") != "C3QueryConditionedGrounding":
        raise RuntimeError(
            "Unexpected Reference grounding specialist."
        )

    print(
        "\nGROUNDING DISPATCH TRACE: PASS"
    )

    print(
        "  outer tool:",
        trace["selected_tool"]
    )

    print(
        "  inner specialist:",
        inner_trace.get("selected_specialist")
    )

    if trace["status"] != "success":
        raise RuntimeError(
            "Reference execution trace does not report success."
        )

    print(
        f"\nExternal wall-clock execution time: "
        f"{elapsed:.2f} ms"
    )

    print("\nREAL REFERENCE AGENT EXECUTION: PASS")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SATQUERYAI — REAL-DATA AGENT GROUNDING SMOKE TEST")
    print("=" * 80)

    print("\nDEVICE: CPU")
    print("TRAINING: NOT STARTED")
    print("CHECKPOINTS: FROZEN")

    # -------------------------------------------------------------------------
    # Verify required paths
    # -------------------------------------------------------------------------

    print("\nPATH CHECK")

    require_path(
        POINT_VAL,
        "Point validation parquet",
    )

    require_path(
        REFERENCE_VAL,
        "Reference validation parquet",
    )

    require_path(
        DOFA_DIR,
        "DOFA cache directory",
    )

    require_path(
        TEXT_DIR,
        "Text embedding cache directory",
    )

    print("  Point validation: PASS")
    print("  Reference validation: PASS")
    print("  DOFA cache: PASS")
    print("  Text cache: PASS")

    # -------------------------------------------------------------------------
    # Build the actual runtime agent
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING RUNTIME AGENT")
    print("=" * 80)

    agent, grounding_tool, _ = (
        create_runtime_agent()
    )

    print(
        "Registered capabilities:",
        agent.registry.capabilities(),
    )

    if "grounding" not in agent.registry.capabilities():
        raise RuntimeError(
            "Grounding capability is not registered."
        )

    print(
        "Grounding router:",
        type(
            grounding_tool.router
        ).__name__,
    )

    print(
        "Unified parameters:",
        grounding_tool.router.parameter_count,
    )

    if (
        grounding_tool.router.parameter_count
        != 3_718_662
    ):
        raise RuntimeError(
            "Unexpected unified parameter count."
        )

    # -------------------------------------------------------------------------
    # Execute both real routes through the agent
    # -------------------------------------------------------------------------

    run_point_test(
        agent
    )

    run_reference_test(
        agent
    )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SATQUERYAI REAL-DATA AGENT GROUNDING SMOKE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()