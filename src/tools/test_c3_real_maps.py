from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

# The 10 maps were extracted here.
MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_real_map_tests"
    / "Reference_Maps"
)

VAL_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "c3_real_map_tests"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUT_DIR / "real_map_validation.json"


# ============================================================
# EXACT AREA-CONSERVING AGGREGATION
# ============================================================

def aggregate_area_conserving(
    src: np.ndarray,
    out_h: int = 56,
    out_w: int = 56,
) -> np.ndarray:
    """
    Convert a source mask to an output area-fraction grid using
    exact geometric overlap.

    Source:
        H x W

    Output:
        out_h x out_w

    Each output value is the fraction of that output cell occupied
    by the source mask.

    For our C3 pipeline:
        120 x 120 -> 56 x 56
    """

    src = np.asarray(src, dtype=np.float64)

    if src.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got shape {src.shape}"
        )

    h, w = src.shape

    # Normalized boundaries of source pixels.
    src_y = np.linspace(0.0, 1.0, h + 1)
    src_x = np.linspace(0.0, 1.0, w + 1)

    # Normalized boundaries of destination cells.
    dst_y = np.linspace(0.0, 1.0, out_h + 1)
    dst_x = np.linspace(0.0, 1.0, out_w + 1)

    # Vertical overlap weights.
    wy = np.zeros(
        (out_h, h),
        dtype=np.float64,
    )

    for oy in range(out_h):

        dst_y0 = dst_y[oy]
        dst_y1 = dst_y[oy + 1]
        dst_height = dst_y1 - dst_y0

        for sy in range(h):

            src_y0 = src_y[sy]
            src_y1 = src_y[sy + 1]

            overlap = max(
                0.0,
                min(dst_y1, src_y1)
                - max(dst_y0, src_y0),
            )

            if overlap > 0.0:
                wy[oy, sy] = overlap / dst_height

    # Horizontal overlap weights.
    wx = np.zeros(
        (out_w, w),
        dtype=np.float64,
    )

    for ox in range(out_w):

        dst_x0 = dst_x[ox]
        dst_x1 = dst_x[ox + 1]
        dst_width = dst_x1 - dst_x0

        for sx in range(w):

            src_x0 = src_x[sx]
            src_x1 = src_x[sx + 1]

            overlap = max(
                0.0,
                min(dst_x1, src_x1)
                - max(dst_x0, src_x0),
            )

            if overlap > 0.0:
                wx[ox, sx] = overlap / dst_width

    # Exact separable area aggregation.
    result = wy @ src @ wx.T

    # Remove microscopic floating-point excursions.
    result = np.clip(
        result,
        0.0,
        1.0,
    )

    return result


# ============================================================
# FIND REFERENCE MAP
# ============================================================

def find_map(patch_id: str) -> Path | None:
    """
    Locate:
        Reference_Maps/<tile>/<patch>/<patch>_reference_map.tif
    """

    # Exact expected structure.
    direct_matches = list(
        MAP_ROOT.glob(
            f"*/{patch_id}/{patch_id}_reference_map.tif"
        )
    )

    if direct_matches:
        return direct_matches[0]

    # Fallback recursive search.
    recursive_matches = list(
        MAP_ROOT.rglob(
            f"{patch_id}_reference_map.tif"
        )
    )

    if recursive_matches:
        return recursive_matches[0]

    return None


# ============================================================
# MAIN VALIDATION
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 REAL REFERENCE-MAP VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic path checks
    # --------------------------------------------------------

    if not VAL_PARQUET.exists():
        raise FileNotFoundError(
            f"Validation parquet not found:\n{VAL_PARQUET}"
        )

    if not MAP_ROOT.exists():
        raise FileNotFoundError(
            f"Reference-map directory not found:\n{MAP_ROOT}"
        )

    # --------------------------------------------------------
    # Load validation split
    # --------------------------------------------------------

    df = pd.read_parquet(VAL_PARQUET)

    if "patch_id" not in df.columns:
        raise ValueError(
            "'patch_id' column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    patch_ids = (
        df["patch_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Validation patches available: {len(patch_ids)}"
    )

    if len(patch_ids) == 0:
        raise RuntimeError(
            "No validation patch IDs found."
        )

    # --------------------------------------------------------
    # Deterministic sample of 10 validation patches
    # --------------------------------------------------------

    sample_count = min(
        10,
        len(patch_ids),
    )

    indices = np.linspace(
        0,
        len(patch_ids) - 1,
        sample_count,
        dtype=int,
    )

    selected_patch_ids = [
        patch_ids[i]
        for i in indices
    ]

    print(
        f"Maps selected for testing: "
        f"{len(selected_patch_ids)}"
    )

    # --------------------------------------------------------
    # Validate maps
    # --------------------------------------------------------

    results = []

    for number, patch_id in enumerate(
        selected_patch_ids,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(
            f"MAP {number}/{len(selected_patch_ids)}"
        )
        print(
            f"PATCH: {patch_id}"
        )

        map_path = find_map(patch_id)

        # ----------------------------------------------------
        # Map not found
        # ----------------------------------------------------

        if map_path is None:

            print(
                "STATUS: MAP NOT FOUND IN EXTRACTED TEST SET"
            )

            results.append(
                {
                    "patch_id": patch_id,
                    "found": False,
                }
            )

            continue

        # ----------------------------------------------------
        # Read map
        # ----------------------------------------------------

        print(
            f"PATH: {map_path}"
        )

        with rasterio.open(map_path) as src:

            arr = src.read(1)

            shape = arr.shape
            dtype = str(arr.dtype)

            unique_values, counts = np.unique(
                arr,
                return_counts=True,
            )

        print(
            f"SHAPE: {shape}"
        )

        print(
            f"DTYPE: {dtype}"
        )

        print(
            "UNIQUE CLC VALUES:",
            unique_values.tolist(),
        )

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if shape != (120, 120):

            raise AssertionError(
                f"Unexpected reference-map shape "
                f"for {patch_id}: {shape}"
            )

        # ----------------------------------------------------
        # Per-CLC-class validation
        # ----------------------------------------------------

        class_results = []

        for clc_value, source_count in zip(
            unique_values,
            counts,
        ):

            clc_value = int(clc_value)
            source_count = int(source_count)

            # Binary mask for this semantic class.
            mask = (
                arr == clc_value
            ).astype(np.float64)

            # Exact 120 -> 56 aggregation.
            target = aggregate_area_conserving(
                mask,
                out_h=56,
                out_w=56,
            )

            # ------------------------------------------------
            # Area conservation
            # ------------------------------------------------

            source_area = float(
                mask.sum()
            )

            output_equivalent_area = (
                float(target.sum())
                * (120.0 / 56.0)
                * (120.0 / 56.0)
            )

            absolute_area_error = abs(
                source_area
                - output_equivalent_area
            )

            # ------------------------------------------------
            # Output range
            # ------------------------------------------------

            target_min = float(
                target.min()
            )

            target_max = float(
                target.max()
            )

            # ------------------------------------------------
            # Report
            # ------------------------------------------------

            print(
                f"  CLC {clc_value} | "
                f"pixels={source_count} | "
                f"area_error={absolute_area_error:.12g} | "
                f"range="
                f"[{target_min:.6f}, "
                f"{target_max:.6f}]"
            )

            # ------------------------------------------------
            # Assertions
            # ------------------------------------------------

            if absolute_area_error >= 1e-10:

                raise AssertionError(
                    "Area conservation failed:\n"
                    f"Patch: {patch_id}\n"
                    f"CLC: {clc_value}\n"
                    f"Source area: {source_area}\n"
                    f"Output equivalent area: "
                    f"{output_equivalent_area}\n"
                    f"Error: {absolute_area_error}"
                )

            if target_min < -1e-12:

                raise AssertionError(
                    "Negative target value detected:\n"
                    f"Patch: {patch_id}\n"
                    f"CLC: {clc_value}\n"
                    f"Minimum: {target_min}"
                )

            if target_max > 1.0 + 1e-12:

                raise AssertionError(
                    "Target value > 1 detected:\n"
                    f"Patch: {patch_id}\n"
                    f"CLC: {clc_value}\n"
                    f"Maximum: {target_max}"
                )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            class_results.append(
                {
                    "clc_value": clc_value,
                    "source_pixels": source_count,
                    "output_equivalent_area": (
                        output_equivalent_area
                    ),
                    "absolute_area_error": (
                        absolute_area_error
                    ),
                    "target_min": target_min,
                    "target_max": target_max,
                }
            )

        # ----------------------------------------------------
        # Map passed
        # ----------------------------------------------------

        results.append(
            {
                "patch_id": patch_id,
                "found": True,
                "map_path": str(map_path),
                "shape": list(shape),
                "dtype": dtype,
                "unique_clc_values": [
                    int(v)
                    for v in unique_values
                ],
                "class_results": class_results,
            }
        )

        print(
            "STATUS: PASS"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    found_count = sum(
        1
        for result in results
        if result.get("found", False)
    )

    missing_count = (
        len(results)
        - found_count
    )

    print("\n" + "=" * 70)
    print("REAL MAP VALIDATION COMPLETE")
    print("=" * 70)
    print(
        f"Maps tested: {len(results)}"
    )
    print(
        f"Maps found: {found_count}"
    )
    print(
        f"Maps missing: {missing_count}"
    )
    print(
        f"Results: {RESULT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()