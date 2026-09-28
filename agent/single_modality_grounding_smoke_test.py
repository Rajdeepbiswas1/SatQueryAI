# =============================================================================
# SATQUERYAI — REAL-DATA SINGLE-MODALITY GROUNDING SMOKE TEST
# =============================================================================
#
# Purpose:
#   Verify the real one-modality grounding paths:
#
#       S1 + text
#           ->
#       SingleModalityS1GroundingV1
#
#       S2 + text
#           ->
#       SingleModalityS2GroundingV1
#
# No training.
# No checkpoint modification.
# No model modification.
#
# Real cached validation data only.
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


EXPECTED_PARAMS = 1_161_986


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
    Convert dataframe / numpy / list / serialized numeric input
    into a float32 tensor.
    """
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()

    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(
            np.array(
                value,
                copy=True,
            )
        ).float()

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
    Locate one patch from the verified paired DOFA cache.
    Only the requested patch is returned.
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

    for shard_index, path in enumerate(
        files,
        start=1,
    ):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        patch_ids = [
            str(
                x.item()
                if torch.is_tensor(x)
                else x
            )
            for x in data["patch_ids"]
        ]

        for row_index, candidate in enumerate(
            patch_ids
        ):
            if candidate == patch_id:
                print(
                    f"DOFA match: shard "
                    f"{shard_index}/{len(files)}"
                )

                s1 = (
                    data["s1"][row_index]
                    .float()
                    .cpu()
                )

                s2 = (
                    data["s2"][row_index]
                    .float()
                    .cpu()
                )

                return s1, s2

    raise KeyError(
        f"Patch not found in DOFA cache:\n{patch_id}"
    )


def find_text_embedding(source_row_index: int):
    """
    Retrieve the exact grounding text embedding using the verified
    source_row_index mapping.
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

    for shard_index, path in enumerate(
        files,
        start=1,
    ):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        source_indices = data[
            "source_row_index"
        ]

        for local_index, row_index in enumerate(
            source_indices.tolist()
        ):
            if int(row_index) == source_row_index:

                print(
                    f"Text match: shard "
                    f"{shard_index}/{len(files)}"
                )

                return (
                    data["text_embeddings"][
                        local_index
                    ]
                    .float()
                    .cpu()
                )

    raise KeyError(
        "Text embedding not found for "
        f"source_row_index={source_row_index}"
    )


def box_iou(box_a, box_b) -> float:
    a = np.asarray(
        box_a,
        dtype=np.float32,
    ).reshape(4)

    b = np.asarray(
        box_b,
        dtype=np.float32,
    ).reshape(4)

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(
        ax1,
        bx1,
    )

    inter_y1 = max(
        ay1,
        by1,
    )

    inter_x2 = min(
        ax2,
        bx2,
    )

    inter_y2 = min(
        ay2,
        by2,
    )

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

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
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


def get_bbox(evidence):
    if not isinstance(
        evidence,
        dict,
    ):
        raise TypeError(
            "Grounding evidence must be a dict."
        )

    if "bbox" not in evidence:
        raise KeyError(
            "Grounding evidence has no 'bbox'."
        )

    bbox = np.asarray(
        evidence["bbox"],
        dtype=np.float32,
    ).reshape(-1)

    if bbox.size != 4:
        raise ValueError(
            f"Unexpected bbox shape: {bbox.shape}"
        )

    return bbox


def verify_trace(
    result,
    expected_specialist,
    expected_modality,
):
    if result.get(
        "answer_status"
    ) != "success":
        raise RuntimeError(
            "Agent execution failed:\n"
            f"{result}"
        )

    analysis = result[
        "analysis"
    ]

    evidence = result[
        "evidence"
    ]

    trace = result[
        "execution_trace"
    ]

    if analysis[
        "task"
    ] != "grounding":
        raise RuntimeError(
            f"Expected grounding task, got "
            f"{analysis['task']}"
        )

    if analysis[
        "grounding_mode"
    ] != "reference":
        raise RuntimeError(
            "Expected reference grounding mode."
        )

    if trace[
        "status"
    ] != "success":
        raise RuntimeError(
            "Outer agent trace failed."
        )

    actual_specialist = evidence.get(
        "specialist"
    )

    if actual_specialist != expected_specialist:
        raise RuntimeError(
            "Unexpected specialist:\n"
            f"expected={expected_specialist}\n"
            f"actual={actual_specialist}"
        )

    if evidence.get(
        "modality"
    ) != expected_modality:
        raise RuntimeError(
            "Unexpected modality:\n"
            f"expected={expected_modality}\n"
            f"actual={evidence.get('modality')}"
        )

    return (
        analysis,
        evidence,
        trace,
    )


def run_single_modality_test(
    agent,
    modality,
    visual,
    text_embedding,
    query,
    patch_id,
    target_box,
):
    print()
    print("=" * 80)
    print(
        f"REAL SINGLE-MODALITY {modality.upper()} "
        "GROUNDING — AGENT EXECUTION"
    )
    print("=" * 80)

    expected_specialist = (
        "SingleModality"
        f"{modality.upper()}"
        "GroundingV1"
    )

    print(
        "Expected specialist:",
        expected_specialist,
    )

    visual = visual.unsqueeze(
        0
    )

    text = text_embedding.unsqueeze(
        0
    )

    print("\nINPUT SHAPES")

    print(
        f"  {modality.upper()}:",
        tuple(visual.shape),
    )

    print(
        "  Text:",
        tuple(text.shape),
    )

    if tuple(
        visual.shape
    ) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            f"Unexpected {modality.upper()} "
            f"shape: {tuple(visual.shape)}"
        )

    if tuple(
        text.shape
    ) != (
        1,
        768,
    ):
        raise RuntimeError(
            f"Unexpected text shape: "
            f"{tuple(text.shape)}"
        )

    if not torch.isfinite(
        visual
    ).all():
        raise RuntimeError(
            f"{modality.upper()} contains "
            "non-finite values."
        )

    if not torch.isfinite(
        text
    ).all():
        raise RuntimeError(
            "Text embedding contains "
            "non-finite values."
        )

    start = time.perf_counter()

    if modality == "s1":
        result = agent.execute(
            query=query,
            input_modalities=["s1"],
            s1=visual,
            text_embedding=text,
            point=None,
            reference=extract_reference(query),
            patch_id=patch_id,
        )
    else:
        result = agent.execute(
            query=query,
            input_modalities=["s2"],
            s2=visual,
            text_embedding=text,
            point=None,
            reference=extract_reference(query),
            patch_id=patch_id,
        )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000.0

    (
        analysis,
        evidence,
        trace,
    ) = verify_trace(
        result,
        expected_specialist,
        modality,
    )

    bbox = get_bbox(
        evidence
    )

    heatmap = evidence.get(
        "heatmap"
    )

    if heatmap is None:
        raise RuntimeError(
            "Single-modality grounding "
            "evidence has no heatmap."
        )

    heatmap_shape = np.asarray(
        heatmap
    ).shape

    if heatmap_shape not in (
        (1, 56, 56),
        (56, 56),
    ):
        raise RuntimeError(
            f"Unexpected heatmap shape: "
            f"{heatmap_shape}"
        )

    parameter_count = evidence.get(
        "parameter_count"
    )

    if parameter_count != EXPECTED_PARAMS:
        raise RuntimeError(
            "Parameter count mismatch:\n"
            f"expected={EXPECTED_PARAMS}\n"
            f"actual={parameter_count}"
        )

    iou = box_iou(
        bbox,
        target_box,
    )

    print("\nAGENT ANALYSIS")
    print(
        "  task:",
        analysis["task"],
    )
    print(
        "  grounding_mode:",
        analysis["grounding_mode"],
    )
    print(
        "  routing_confidence:",
        analysis["confidence"],
    )

    print("\nSTANDARDIZED EVIDENCE")
    print(
        "  specialist:",
        evidence.get("specialist"),
    )
    print(
        "  modality:",
        evidence.get("modality"),
    )
    print(
        "  checkpoint_epoch:",
        evidence.get("checkpoint_epoch"),
    )
    print(
        "  parameter_count:",
        parameter_count,
    )
    print(
        "  bbox:",
        bbox.tolist(),
    )
    print(
        "  heatmap shape:",
        heatmap_shape,
    )
    print(
        "  IoU vs ground truth:",
        f"{iou:.6f}",
    )
    print(
        "  evidence_confidence:",
        evidence.get(
            "evidence_confidence"
        ),
    )
    print(
        "  peak_probability:",
        evidence.get(
            "peak_probability"
        ),
    )

    print("\nEXECUTION TRACE")
    print(
        "  selected_tool:",
        trace.get(
            "selected_tool"
        ),
    )
    print(
        "  status:",
        trace.get(
            "status"
        ),
    )
    print(
        "  validation_ms:",
        trace.get(
            "validation_ms"
        ),
    )
    print(
        "  execution_ms:",
        trace.get(
            "execution_ms"
        ),
    )
    print(
        "  total_ms:",
        trace.get(
            "total_ms"
        ),
    )

    print(
        f"\nExternal wall-clock execution time: "
        f"{elapsed:.2f} ms"
    )

    print(
        f"\nREAL SINGLE-MODALITY "
        f"{modality.upper()} GROUNDING: PASS"
    )

    return {
        "modality": modality,
        "specialist": evidence.get(
            "specialist"
        ),
        "bbox": bbox,
        "iou": iou,
        "heatmap_shape": heatmap_shape,
        "parameter_count": parameter_count,
        "checkpoint_epoch": evidence.get(
            "checkpoint_epoch"
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print(
        "SATQUERYAI — REAL-DATA SINGLE-MODALITY "
        "GROUNDING SMOKE TEST"
    )
    print("=" * 80)

    print("\nDEVICE: CPU")
    print("TRAINING: NOT STARTED")
    print("CHECKPOINTS: FROZEN")

    # -------------------------------------------------------------------------
    # PATH CHECK
    # -------------------------------------------------------------------------

    print("\nPATH CHECK")

    require_path(
        REFERENCE_VAL,
        "Reference validation",
    )

    require_path(
        DOFA_DIR,
        "DOFA cache",
    )

    require_path(
        TEXT_DIR,
        "Text cache",
    )

    print(
        "  Reference validation: PASS"
    )

    print(
        "  DOFA cache: PASS"
    )

    print(
        "  Text cache: PASS"
    )

    # -------------------------------------------------------------------------
    # BUILD AGENT
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BUILDING RUNTIME AGENT")
    print("=" * 80)

    agent, _, _ = create_runtime_agent()

    print(
        "Registered capabilities:",
        agent.registry.capabilities(),
    )

    # -------------------------------------------------------------------------
    # LOAD ONE REAL REFERENCE SAMPLE
    # -------------------------------------------------------------------------

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

    source_row_index = int(
        row["original_row_index"]
    )

    target_box = to_float_tensor(
        row["output"],
        shape=(4,),
    ).numpy()

    reference = extract_reference(
        query
    )

    if not reference:
        raise RuntimeError(
            "Could not extract reference phrase "
            "from validation query."
        )

    print("\nDATASET SAMPLE")

    print(
        "  original_row_index:",
        source_row_index,
    )

    print(
        "  patch_id:",
        patch_id,
    )

    print(
        "  reference:",
        reference,
    )

    print(
        "  target_box:",
        target_box.tolist(),
    )

    print(
        "  query:",
        query,
    )

    # -------------------------------------------------------------------------
    # LOAD FEATURES
    # -------------------------------------------------------------------------

    s1, s2 = find_dofa_features(
        patch_id
    )

    text_embedding = find_text_embedding(
        source_row_index
    )

    # -------------------------------------------------------------------------
    # TEST S1
    # -------------------------------------------------------------------------

    s1_result = run_single_modality_test(
        agent=agent,
        modality="s1",
        visual=s1,
        text_embedding=text_embedding,
        query=query,
        patch_id=patch_id,
        target_box=target_box,
    )

    # -------------------------------------------------------------------------
    # TEST S2
    # -------------------------------------------------------------------------

    s2_result = run_single_modality_test(
        agent=agent,
        modality="s2",
        visual=s2,
        text_embedding=text_embedding,
        query=query,
        patch_id=patch_id,
        target_box=target_box,
    )

    # -------------------------------------------------------------------------
    # FINAL CHECK
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "SINGLE-MODALITY GROUNDING RESULTS"
    )

    print("=" * 80)

    print(
        "  S1 specialist:",
        s1_result["specialist"],
    )

    print(
        "  S1 IoU:",
        f"{s1_result['iou']:.6f}",
    )

    print(
        "  S2 specialist:",
        s2_result["specialist"],
    )

    print(
        "  S2 IoU:",
        f"{s2_result['iou']:.6f}",
    )

    if (
        s1_result["specialist"]
        != "SingleModalityS1GroundingV1"
    ):
        raise RuntimeError(
            "Final S1 specialist verification failed."
        )

    if (
        s2_result["specialist"]
        != "SingleModalityS2GroundingV1"
    ):
        raise RuntimeError(
            "Final S2 specialist verification failed."
        )

    if (
        s1_result["parameter_count"]
        != EXPECTED_PARAMS
        or
        s2_result["parameter_count"]
        != EXPECTED_PARAMS
    ):
        raise RuntimeError(
            "Final parameter-count verification failed."
        )

    print(
        "\nSATQUERYAI REAL-DATA SINGLE-MODALITY "
        "GROUNDING SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()