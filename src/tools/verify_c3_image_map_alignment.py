from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

REFERENCE_MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_real_map_tests"
    / "Reference_Maps"
)

PREPROCESSED_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_real_map_tests"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "image_map_alignment.json"
)


# ============================================================
# FIND REFERENCE MAPS
# ============================================================

def find_reference_maps() -> list[Path]:
    """Find all reference-map TIFF files."""

    if not REFERENCE_MAP_ROOT.exists():
        raise FileNotFoundError(
            "Reference-map directory does not exist:\n"
            f"{REFERENCE_MAP_ROOT}"
        )

    maps = sorted(
        REFERENCE_MAP_ROOT.rglob(
            "*_reference_map.tif"
        )
    )

    return maps


# ============================================================
# EXTRACT PATCH ID
# ============================================================

def extract_patch_id(
    map_path: Path,
) -> str:
    """
    Convert:

        PATCH_ID_reference_map.tif

    into:

        PATCH_ID
    """

    suffix = "_reference_map"

    if not map_path.stem.endswith(suffix):
        raise ValueError(
            "Unexpected reference-map filename:\n"
            f"{map_path.name}"
        )

    return map_path.stem[:-len(suffix)]


# ============================================================
# VERIFY PREPROCESSED IMAGE
# ============================================================

def verify_preprocessed_image(
    image_path: Path,
    expected_patch_id: str,
) -> dict:
    """Verify one preprocessed image tensor."""

    result = {
        "path": str(image_path),
        "exists": image_path.exists(),
        "expected_patch_id": expected_patch_id,
    }

    if not image_path.exists():
        result["status"] = "MISSING"
        return result

    try:
        data = torch.load(
            image_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        result["status"] = "LOAD_ERROR"
        result["error"] = str(exc)
        return result

    result["loaded_type"] = type(data).__name__

    if not isinstance(data, dict):
        result["status"] = "INVALID_FORMAT"
        result["error"] = (
            "Expected a dictionary."
        )
        return result

    result["keys"] = list(data.keys())

    # --------------------------------------------------------
    # PATCH ID
    # --------------------------------------------------------

    patch_id = data.get("patch_id")

    if patch_id is None:
        result["status"] = "MISSING_PATCH_ID"
        return result

    patch_id = str(patch_id)

    result["patch_id_in_tensor"] = patch_id
    result["patch_id_matches"] = (
        patch_id == expected_patch_id
    )

    if patch_id != expected_patch_id:
        result["status"] = "PATCH_ID_MISMATCH"
        return result

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = data.get("image")

    if image is None:
        result["status"] = "MISSING_IMAGE"
        return result

    if not isinstance(image, torch.Tensor):
        result["status"] = "INVALID_IMAGE_TYPE"
        result["image_type"] = type(image).__name__
        return result

    result["image_type"] = "torch.Tensor"
    result["image_shape"] = list(image.shape)
    result["image_dtype"] = str(image.dtype)

    expected_shape = (
        14,
        120,
        120,
    )

    if tuple(image.shape) != expected_shape:
        result["status"] = "IMAGE_SHAPE_MISMATCH"
        result["expected_shape"] = list(
            expected_shape
        )
        return result

    # --------------------------------------------------------
    # FINITE VALUES
    # --------------------------------------------------------

    image_float = image.float()

    finite_mask = torch.isfinite(
        image_float
    )

    finite_all = bool(
        finite_mask.all().item()
    )

    result["finite_all"] = finite_all

    if not finite_all:
        result["status"] = "NONFINITE_VALUES"
        return result

    # --------------------------------------------------------
    # CHANNEL NAMES
    # --------------------------------------------------------

    channel_names = data.get(
        "channel_names"
    )

    if channel_names is None:
        result["channel_names"] = None
        result["channel_count"] = None
        result["channel_count_matches"] = None
    else:
        channel_names = [
            str(name)
            for name in channel_names
        ]

        result["channel_names"] = channel_names
        result["channel_count"] = len(
            channel_names
        )
        result["channel_count_matches"] = (
            len(channel_names) == 14
        )

    # --------------------------------------------------------
    # IMAGE STATISTICS
    # --------------------------------------------------------

    result["image_min"] = float(
        image_float.min().item()
    )

    result["image_max"] = float(
        image_float.max().item()
    )

    result["image_mean"] = float(
        image_float.mean().item()
    )

    result["image_std"] = float(
        image_float.std().item()
    )

    result["status"] = "PASS"

    return result


# ============================================================
# VERIFY REFERENCE MAP
# ============================================================

def verify_reference_map(
    map_path: Path,
) -> dict:
    """Verify the reference map dimensions and metadata."""

    result = {
        "path": str(map_path),
        "exists": map_path.exists(),
    }

    if not map_path.exists():
        result["status"] = "MISSING"
        return result

    try:
        with rasterio.open(map_path) as src:

            result["shape"] = [
                src.height,
                src.width,
            ]

            result["dtype"] = str(
                src.dtypes[0]
            )

            result["count"] = int(
                src.count
            )

            result["crs"] = (
                str(src.crs)
                if src.crs is not None
                else None
            )

            result["transform"] = list(
                src.transform
            )

    except Exception as exc:
        result["status"] = "READ_ERROR"
        result["error"] = str(exc)
        return result

    expected_shape = (
        120,
        120,
    )

    actual_shape = (
        result["shape"][0],
        result["shape"][1],
    )

    if actual_shape != expected_shape:
        result["status"] = (
            "REFERENCE_MAP_SHAPE_MISMATCH"
        )
        return result

    result["status"] = "PASS"

    return result


# ============================================================
# VERIFY ONE PATCH
# ============================================================

def verify_patch(
    map_path: Path,
) -> dict:
    """Verify reference map and matching preprocessed image."""

    patch_id = extract_patch_id(
        map_path
    )

    print("\n" + "-" * 70)
    print(
        f"PATCH: {patch_id}"
    )

    # --------------------------------------------------------
    # REFERENCE MAP
    # --------------------------------------------------------

    print(
        f"REFERENCE MAP: {map_path}"
    )

    map_result = verify_reference_map(
        map_path
    )

    if map_result["status"] == "PASS":

        print(
            "MAP SHAPE: "
            f"{tuple(map_result['shape'])}"
        )

        print(
            "MAP DTYPE: "
            f"{map_result['dtype']}"
        )

        print(
            "MAP SHAPE CHECK: PASS"
        )

    else:

        print(
            "MAP CHECK: FAIL — "
            f"{map_result['status']}"
        )

    # --------------------------------------------------------
    # PREPROCESSED IMAGE
    # --------------------------------------------------------

    image_path = (
        PREPROCESSED_ROOT
        / f"{patch_id}.pt"
    )

    print(
        f"PREPROCESSED IMAGE: {image_path}"
    )

    image_result = verify_preprocessed_image(
        image_path,
        patch_id,
    )

    if image_result["status"] == "PASS":

        print(
            "IMAGE PATCH ID: "
            f"{image_result['patch_id_in_tensor']}"
        )

        print(
            "PATCH ID MATCH: PASS"
        )

        print(
            "IMAGE SHAPE: "
            f"{tuple(image_result['image_shape'])}"
        )

        print(
            "IMAGE SHAPE CHECK: PASS"
        )

        print(
            "FINITE VALUES: PASS"
        )

        if (
            image_result["channel_names"]
            is not None
        ):
            print(
                "CHANNEL COUNT: "
                f"{image_result['channel_count']}"
            )

            if image_result[
                "channel_count_matches"
            ]:
                print(
                    "CHANNEL COUNT CHECK: PASS"
                )
            else:
                print(
                    "CHANNEL COUNT CHECK: FAIL"
                )
        else:
            print(
                "CHANNEL COUNT: NOT AVAILABLE"
            )

    else:

        print(
            "IMAGE CHECK: FAIL — "
            f"{image_result['status']}"
        )

    # --------------------------------------------------------
    # FINAL PATCH STATUS
    # --------------------------------------------------------

    patch_passed = (
        map_result["status"] == "PASS"
        and image_result["status"] == "PASS"
    )

    result = {
        "patch_id": patch_id,
        "reference_map": map_result,
        "preprocessed_image": image_result,
        "status": (
            "PASS"
            if patch_passed
            else "FAIL"
        ),
    }

    print(
        "PATCH STATUS: "
        f"{result['status']}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 IMAGE <-> REFERENCE-MAP ALIGNMENT")
    print("=" * 70)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find maps
    # --------------------------------------------------------

    map_paths = find_reference_maps()

    print(
        f"Reference maps found: {len(map_paths)}"
    )

    if not map_paths:
        raise RuntimeError(
            "No reference maps were found."
        )

    # --------------------------------------------------------
    # Verify every extracted test map
    # --------------------------------------------------------

    results = []

    for map_number, map_path in enumerate(
        map_paths,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"MAP {map_number}/{len(map_paths)}"
        )

        result = verify_patch(
            map_path
        )

        results.append(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    failed = (
        len(results)
        - passed
    )

    summary = {
        "reference_maps_tested": len(results),
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "results": results,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "ALIGNMENT VALIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Reference maps tested: {len(results)}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Results: {RESULT_PATH}"
    )

    if failed == 0:

        print(
            "ALL IMAGE <-> MAP ALIGNMENT "
            "TESTS PASSED"
        )

    else:

        print(
            "ALIGNMENT VALIDATION FAILED"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()