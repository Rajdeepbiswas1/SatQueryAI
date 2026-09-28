from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

COMPONENT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
    / "component_selection_results.csv"
)

ROW_AUDIT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
    / "reference_row_clc_audit.csv"
)

MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
    / "component_consistency_report.csv"
)

SUMMARY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
    / "component_consistency_summary.json"
)

IMAGE_SIZE = 120

STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8,
)

TOLERANCE = 1e-10


# ============================================================
# LOAD BBOX WEIGHTS
# ============================================================

def bbox_pixel_weights(
    bbox: tuple[float, float, float, float],
    height: int,
    width: int,
) -> tuple[np.ndarray, float]:

    x1, y1, x2, y2 = bbox

    pixel_x0 = np.linspace(
        0.0,
        1.0,
        width + 1,
    )

    pixel_x1 = pixel_x0[1:]

    pixel_y0 = np.linspace(
        0.0,
        1.0,
        height + 1,
    )

    pixel_y1 = pixel_y0[1:]

    x_overlap = np.maximum(
        0.0,
        np.minimum(
            pixel_x1,
            x2,
        )
        - np.maximum(
            pixel_x0[:-1],
            x1,
        ),
    )

    y_overlap = np.maximum(
        0.0,
        np.minimum(
            pixel_y1,
            y2,
        )
        - np.maximum(
            pixel_y0[:-1],
            y1,
        ),
    )

    weights = (
        y_overlap[:, None]
        * x_overlap[None, :]
    )

    bbox_area = (
        max(
            0.0,
            x2 - x1,
        )
        * max(
            0.0,
            y2 - y1,
        )
    )

    return weights, bbox_area


# ============================================================
# COMPONENT METRICS
# ============================================================

def component_metrics(
    component_mask: np.ndarray,
    bbox_weights: np.ndarray,
    bbox_area: float,
) -> dict:

    pixel_count = int(
        component_mask.sum()
    )

    if pixel_count == 0:

        return {
            "pixel_count": 0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    component_area = (
        pixel_count
        / (
            IMAGE_SIZE
            * IMAGE_SIZE
        )
    )

    intersection = float(
        bbox_weights[
            component_mask
        ].sum()
    )

    precision = (
        intersection
        / component_area
    )

    recall = (
        intersection
        / bbox_area
        if bbox_area > 0.0
        else 0.0
    )

    union = (
        component_area
        + bbox_area
        - intersection
    )

    iou = (
        intersection
        / union
        if union > 0.0
        else 0.0
    )

    return {
        "pixel_count": pixel_count,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# ANALYZE CANDIDATE MASK
# ============================================================

def analyze_candidate_mask(
    candidate_mask: np.ndarray,
    bbox_weights: np.ndarray,
    bbox_area: float,
) -> dict:

    candidate_pixels = int(
        candidate_mask.sum()
    )

    candidate_area = (
        candidate_pixels
        / (
            IMAGE_SIZE
            * IMAGE_SIZE
        )
    )

    candidate_intersection = float(
        bbox_weights[
            candidate_mask
        ].sum()
    )

    candidate_precision = (
        candidate_intersection
        / candidate_area
        if candidate_area > 0.0
        else 0.0
    )

    candidate_recall = (
        candidate_intersection
        / bbox_area
        if bbox_area > 0.0
        else 0.0
    )

    labels, component_count = (
        ndimage.label(
            candidate_mask,
            structure=STRUCTURE,
        )
    )

    component_sizes = np.bincount(
        labels.ravel()
    )

    component_sizes = component_sizes[
        1:
    ]

    if component_count == 0:

        empty = {
            "pixel_count": 0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        return {
            "candidate_pixels": 0,
            "candidate_area": 0.0,
            "candidate_precision": 0.0,
            "candidate_recall": 0.0,
            "component_count": 0,
            "largest": empty,
            "smallest": empty,
            "oracle_best_iou": 0.0,
            "oracle_best_precision": 0.0,
            "oracle_best_recall": 0.0,
        }

    # --------------------------------------------------------
    # Largest component
    # --------------------------------------------------------

    largest_number = (
        int(
            np.argmax(
                component_sizes
            )
        )
        + 1
    )

    largest_mask = (
        labels
        == largest_number
    )

    largest = component_metrics(
        largest_mask,
        bbox_weights,
        bbox_area,
    )

    # --------------------------------------------------------
    # Smallest component
    # --------------------------------------------------------

    smallest_number = (
        int(
            np.argmin(
                component_sizes
            )
        )
        + 1
    )

    smallest_mask = (
        labels
        == smallest_number
    )

    smallest = component_metrics(
        smallest_mask,
        bbox_weights,
        bbox_area,
    )

    # --------------------------------------------------------
    # Oracle best component
    # --------------------------------------------------------

    best_iou = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for component_number in range(
        1,
        component_count + 1,
    ):

        mask = (
            labels
            == component_number
        )

        metrics = component_metrics(
            mask,
            bbox_weights,
            bbox_area,
        )

        if metrics["iou"] > best_iou:

            best_iou = metrics["iou"]
            best_precision = (
                metrics["precision"]
            )
            best_recall = (
                metrics["recall"]
            )

    return {
        "candidate_pixels": candidate_pixels,
        "candidate_area": candidate_area,
        "candidate_precision": float(
            candidate_precision
        ),
        "candidate_recall": float(
            candidate_recall
        ),
        "component_count": int(
            component_count
        ),
        "largest": largest,
        "smallest": smallest,
        "oracle_best_iou": float(
            best_iou
        ),
        "oracle_best_precision": float(
            best_precision
        ),
        "oracle_best_recall": float(
            best_recall
        ),
    }


# ============================================================
# FIND MAP
# ============================================================

def find_map(
    patch_id: str,
) -> Path:

    matches = list(
        MAP_ROOT.rglob(
            f"{patch_id}_reference_map.tif"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one map for "
            f"{patch_id}, found {len(matches)}"
        )

    return matches[0]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 COMPONENT CONSISTENCY VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not COMPONENT_FILE.exists():
        raise FileNotFoundError(
            COMPONENT_FILE
        )

    if not ROW_AUDIT_FILE.exists():
        raise FileNotFoundError(
            ROW_AUDIT_FILE
        )

    component_df = pd.read_csv(
        COMPONENT_FILE
    )

    row_df = pd.read_csv(
        ROW_AUDIT_FILE
    )

    print(
        f"Component rows: "
        f"{len(component_df)}"
    )

    print(
        f"Row-audit rows: "
        f"{len(row_df)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    component_required = {
        "original_row_index",
        "patch_id",
        "k",
        "candidate_clc_codes",
        "candidate_component_count",
        "candidate_pixels",
        "candidate_area",
        "candidate_coverage_recall",
        "candidate_precision",
        "largest_pixels",
        "largest_iou",
        "largest_precision",
        "largest_recall",
        "smallest_pixels",
        "smallest_iou",
        "smallest_precision",
        "smallest_recall",
        "oracle_best_iou",
        "oracle_best_precision",
        "oracle_best_recall",
        "selected_iou",
        "selected_precision",
        "selected_recall",
        "comparator",
    }

    row_required = {
        "original_row_index",
        "patch_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }

    missing_component = (
        component_required
        - set(component_df.columns)
    )

    missing_row = (
        row_required
        - set(row_df.columns)
    )

    if missing_component:

        raise ValueError(
            "Missing component columns:\n"
            f"{sorted(missing_component)}"
        )

    if missing_row:

        raise ValueError(
            "Missing row-audit columns:\n"
            f"{sorted(missing_row)}"
        )

    # --------------------------------------------------------
    # Row lookup
    # --------------------------------------------------------

    if row_df[
        "original_row_index"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate original_row_index in "
            "row audit."
        )

    row_lookup = (
        row_df
        .set_index(
            "original_row_index"
        )
        .to_dict(
            orient="index"
        )
    )

    # --------------------------------------------------------
    # Cache maps
    # --------------------------------------------------------

    map_cache: dict[
        str,
        np.ndarray,
    ] = {}

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    verification_rows = []

    failures = []

    processed = 0

    unique_patches = (
        component_df[
            "patch_id"
        ]
        .astype(str)
        .nunique()
    )

    print(
        f"Unique patches: "
        f"{unique_patches}"
    )

    # ========================================================
    # PROCESS EACH COMPONENT-ANALYSIS ROW
    # ========================================================

    for index, stored in component_df.iterrows():

        row_id = int(
            stored[
                "original_row_index"
            ]
        )

        patch_id = str(
            stored[
                "patch_id"
            ]
        )

        # ----------------------------------------------------
        # Retrieve matching bbox.
        # ----------------------------------------------------

        if row_id not in row_lookup:

            raise RuntimeError(
                "No matching row-audit entry:\n"
                f"{row_id}"
            )

        source = row_lookup[
            row_id
        ]

        # Verify patch IDs agree.
        source_patch_id = str(
            source[
                "patch_id"
            ]
        )

        if source_patch_id != patch_id:

            raise RuntimeError(
                "Patch ID mismatch:\n"
                f"component={patch_id}\n"
                f"audit={source_patch_id}\n"
                f"row={row_id}"
            )

        bbox = (
            float(
                source["bbox_x1"]
            ),
            float(
                source["bbox_y1"]
            ),
            float(
                source["bbox_x2"]
            ),
            float(
                source["bbox_y2"]
            ),
        )

        # ----------------------------------------------------
        # Load reference map once per patch.
        # ----------------------------------------------------

        if patch_id not in map_cache:

            map_path = find_map(
                patch_id
            )

            with rasterio.open(
                map_path
            ) as src:

                clc_map = src.read(
                    1
                )

            if clc_map.shape != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ):

                raise RuntimeError(
                    "Unexpected map shape:\n"
                    f"patch={patch_id}\n"
                    f"shape={clc_map.shape}"
                )

            map_cache[
                patch_id
            ] = clc_map

        else:

            clc_map = map_cache[
                patch_id
            ]

        # ----------------------------------------------------
        # Exact bbox weights.
        # ----------------------------------------------------

        bbox_weights, bbox_area = (
            bbox_pixel_weights(
                bbox,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        # ----------------------------------------------------
        # Decode CLC candidate codes.
        # ----------------------------------------------------

        try:

            candidate_codes = json.loads(
                str(
                    stored[
                        "candidate_clc_codes"
                    ]
                )
            )

            candidate_codes = [
                int(x)
                for x in candidate_codes
            ]

        except Exception as exc:

            raise RuntimeError(
                "Could not parse candidate "
                "CLC codes:\n"
                f"{stored['candidate_clc_codes']}"
            ) from exc

        # ----------------------------------------------------
        # Candidate mask.
        # ----------------------------------------------------

        candidate_mask = np.isin(
            clc_map,
            candidate_codes,
        )

        # ----------------------------------------------------
        # Independently recompute.
        # ----------------------------------------------------

        recomputed = (
            analyze_candidate_mask(
                candidate_mask,
                bbox_weights,
                bbox_area,
            )
        )

        largest = recomputed[
            "largest"
        ]

        smallest = recomputed[
            "smallest"
        ]

        # ----------------------------------------------------
        # Comparator-specific selection.
        # ----------------------------------------------------

        comparator = str(
            stored[
                "comparator"
            ]
        ).lower()

        if comparator == "largest":

            selected = largest

        elif comparator == "smallest":

            selected = smallest

        elif comparator == "ordinary":

            # The original experiment uses the oracle
            # component for ordinary queries.
            selected = {
                "iou": (
                    recomputed[
                        "oracle_best_iou"
                    ]
                ),
                "precision": (
                    recomputed[
                        "oracle_best_precision"
                    ]
                ),
                "recall": (
                    recomputed[
                        "oracle_best_recall"
                    ]
                ),
            }

        else:

            raise RuntimeError(
                f"Unexpected comparator: "
                f"{comparator}"
            )

        # ----------------------------------------------------
        # Differences.
        # ----------------------------------------------------

        diffs = {
            "candidate_component_count": abs(
                int(
                    stored[
                        "candidate_component_count"
                    ]
                )
                - int(
                    recomputed[
                        "component_count"
                    ]
                )
            ),

            "candidate_pixels": abs(
                int(
                    stored[
                        "candidate_pixels"
                    ]
                )
                - int(
                    recomputed[
                        "candidate_pixels"
                    ]
                )
            ),

            "candidate_area": abs(
                float(
                    stored[
                        "candidate_area"
                    ]
                )
                - float(
                    recomputed[
                        "candidate_area"
                    ]
                )
            ),

            "candidate_coverage_recall": abs(
                float(
                    stored[
                        "candidate_coverage_recall"
                    ]
                )
                - float(
                    recomputed[
                        "candidate_recall"
                    ]
                )
            ),

            "candidate_precision": abs(
                float(
                    stored[
                        "candidate_precision"
                    ]
                )
                - float(
                    recomputed[
                        "candidate_precision"
                    ]
                )
            ),

            "largest_pixels": abs(
                int(
                    stored[
                        "largest_pixels"
                    ]
                )
                - int(
                    largest[
                        "pixel_count"
                    ]
                )
            ),

            "largest_iou": abs(
                float(
                    stored[
                        "largest_iou"
                    ]
                )
                - float(
                    largest[
                        "iou"
                    ]
                )
            ),

            "largest_precision": abs(
                float(
                    stored[
                        "largest_precision"
                    ]
                )
                - float(
                    largest[
                        "precision"
                    ]
                )
            ),

            "largest_recall": abs(
                float(
                    stored[
                        "largest_recall"
                    ]
                )
                - float(
                    largest[
                        "recall"
                    ]
                )
            ),

            "smallest_pixels": abs(
                int(
                    stored[
                        "smallest_pixels"
                    ]
                )
                - int(
                    smallest[
                        "pixel_count"
                    ]
                )
            ),

            "smallest_iou": abs(
                float(
                    stored[
                        "smallest_iou"
                    ]
                )
                - float(
                    smallest[
                        "iou"
                    ]
                )
            ),

            "smallest_precision": abs(
                float(
                    stored[
                        "smallest_precision"
                    ]
                )
                - float(
                    smallest[
                        "precision"
                    ]
                )
            ),

            "smallest_recall": abs(
                float(
                    stored[
                        "smallest_recall"
                    ]
                )
                - float(
                    smallest[
                        "recall"
                    ]
                )
            ),

            "oracle_best_iou": abs(
                float(
                    stored[
                        "oracle_best_iou"
                    ]
                )
                - float(
                    recomputed[
                        "oracle_best_iou"
                    ]
                )
            ),

            "oracle_best_precision": abs(
                float(
                    stored[
                        "oracle_best_precision"
                    ]
                )
                - float(
                    recomputed[
                        "oracle_best_precision"
                    ]
                )
            ),

            "oracle_best_recall": abs(
                float(
                    stored[
                        "oracle_best_recall"
                    ]
                )
                - float(
                    recomputed[
                        "oracle_best_recall"
                    ]
                )
            ),

            "selected_iou": abs(
                float(
                    stored[
                        "selected_iou"
                    ]
                )
                - float(
                    selected[
                        "iou"
                    ]
                )
            ),

            "selected_precision": abs(
                float(
                    stored[
                        "selected_precision"
                    ]
                )
                - float(
                    selected[
                        "precision"
                    ]
                )
            ),

            "selected_recall": abs(
                float(
                    stored[
                        "selected_recall"
                    ]
                )
                - float(
                    selected[
                        "recall"
                    ]
                )
            ),
        }

        # ----------------------------------------------------
        # Determine pass/fail.
        # ----------------------------------------------------

        integer_checks = [
            diffs[
                "candidate_component_count"
            ],
            diffs[
                "candidate_pixels"
            ],
            diffs[
                "largest_pixels"
            ],
            diffs[
                "smallest_pixels"
            ],
        ]

        numeric_checks = [
            value
            for key, value in diffs.items()
            if key
            not in {
                "candidate_component_count",
                "candidate_pixels",
                "largest_pixels",
                "smallest_pixels",
            }
        ]

        passed = (
            all(
                value == 0
                for value in integer_checks
            )
            and all(
                value <= TOLERANCE
                for value in numeric_checks
            )
        )

        verification_rows.append(
            {
                "original_row_index": row_id,
                "patch_id": patch_id,
                "k": int(
                    stored["k"]
                ),
                "comparator": comparator,
                "passed": passed,
                **diffs,
            }
        )

        if not passed:

            failures.append(
                {
                    "original_row_index": row_id,
                    "patch_id": patch_id,
                    "k": int(
                        stored["k"]
                    ),
                    "comparator": comparator,
                    "diffs": diffs,
                }
            )

        processed += 1

        if (
            processed <= 10
            or processed % 2500 == 0
            or processed == len(
                component_df
            )
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{len(component_df)}"
            )

    # ========================================================
    # REPORT
    # ========================================================

    verification_df = pd.DataFrame(
        verification_rows
    )

    verification_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Maximum numeric differences.
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in verification_df.columns
        if column not in {
            "original_row_index",
            "patch_id",
            "k",
            "comparator",
            "passed",
            "candidate_component_count",
            "candidate_pixels",
            "largest_pixels",
            "smallest_pixels",
        }
    ]

    max_differences = {}

    for column in numeric_columns:

        max_differences[column] = float(
            verification_df[
                column
            ].max()
        )

    passed_count = int(
        verification_df[
            "passed"
        ].sum()
    )

    failed_count = (
        len(verification_df)
        - passed_count
    )

    summary = {
        "rows_checked": int(
            len(verification_df)
        ),
        "passed": passed_count,
        "failed": failed_count,
        "all_passed": (
            failed_count == 0
        ),
        "tolerance": TOLERANCE,
        "max_differences": max_differences,
        "output_file": str(
            OUTPUT_FILE
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("C3 COMPONENT CONSISTENCY RESULT")
    print("=" * 70)

    print(
        f"Rows checked: {len(verification_df)}"
    )

    print(
        f"Passed: {passed_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        f"Tolerance: {TOLERANCE}"
    )

    print("\nMaximum differences:")

    for key, value in max_differences.items():

        print(
            f"  {key}: {value:.12g}"
        )

    print(
        f"\nReport:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary:\n{SUMMARY_FILE}"
    )

    if failed_count == 0:

        print(
            "\nALL COMPONENT RESULTS "
            "REPRODUCED EXACTLY"
        )

    else:

        print(
            f"\nCONSISTENCY CHECK FAILED "
            f"FOR {failed_count} ROWS"
        )

        print(
            "\nFirst failures:"
        )

        for failure in failures[:10]:

            print(
                failure
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()