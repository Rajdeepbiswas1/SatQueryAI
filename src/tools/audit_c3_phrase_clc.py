from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

TRAIN_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

REFERENCE_MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

ROW_OUTPUT = (
    OUTPUT_ROOT
    / "reference_row_clc_audit.csv"
)

PHRASE_OUTPUT = (
    OUTPUT_ROOT
    / "phrase_clc_summary.csv"
)

CLC_OUTPUT = (
    OUTPUT_ROOT
    / "phrase_clc_matrix.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_ROOT
    / "audit_summary.json"
)


# ============================================================
# REGULAR EXPRESSIONS
# ============================================================

REF_RE = re.compile(
    r"<ref>(.*?)</ref>",
    flags=re.IGNORECASE | re.DOTALL,
)

BBOX_RE = re.compile(
    r"\[\s*"
    r"([0-9]*\.?[0-9]+)"
    r"\s+"
    r"([0-9]*\.?[0-9]+)"
    r"\s*,\s*"
    r"([0-9]*\.?[0-9]+)"
    r"\s+"
    r"([0-9]*\.?[0-9]+)"
    r"\s*\]"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_reference_rows() -> pd.DataFrame:

    if not TRAIN_PARQUET.exists():
        raise FileNotFoundError(
            f"Train parquet not found:\n{TRAIN_PARQUET}"
        )

    if not VAL_PARQUET.exists():
        raise FileNotFoundError(
            f"Validation parquet not found:\n{VAL_PARQUET}"
        )

    train = pd.read_parquet(
        TRAIN_PARQUET
    )

    val = pd.read_parquet(
        VAL_PARQUET
    )

    train["audit_split"] = "train"
    val["audit_split"] = "val"

    df = pd.concat(
        [train, val],
        ignore_index=True,
    )

    return df


# ============================================================
# PARSE REFERENCE PHRASE
# ============================================================

def extract_reference_phrase(
    text: str,
) -> str | None:

    match = REF_RE.search(
        str(text)
    )

    if match is None:
        return None

    phrase = match.group(1).strip()

    return phrase


# ============================================================
# PARSE NORMALIZED BBOX
# ============================================================

def parse_bbox(
    text: str,
) -> tuple[float, float, float, float] | None:

    match = BBOX_RE.search(
        str(text)
    )

    if match is None:
        return None

    x1, y1, x2, y2 = (
        float(match.group(i))
        for i in range(1, 5)
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# FIND REFERENCE MAP
# ============================================================

def build_map_index() -> dict[str, Path]:

    if not REFERENCE_MAP_ROOT.exists():
        raise FileNotFoundError(
            "Reference-map directory not found:\n"
            f"{REFERENCE_MAP_ROOT}"
        )

    print("Indexing reference maps...")

    index: dict[str, Path] = {}

    for path in REFERENCE_MAP_ROOT.rglob(
        "*_reference_map.tif"
    ):

        filename = path.name

        suffix = "_reference_map.tif"

        if not filename.endswith(suffix):
            continue

        patch_id = filename[
            :-len(suffix)
        ]

        if patch_id in index:
            raise RuntimeError(
                "Duplicate reference map for "
                f"patch_id:\n{patch_id}"
            )

        index[patch_id] = path

    return index


# ============================================================
# EXACT BBOX ↔ PIXEL AREA WEIGHTS
# ============================================================

def bbox_pixel_weights(
    height: int,
    width: int,
    bbox: tuple[float, float, float, float],
) -> tuple[
    np.ndarray,
    float,
    tuple[int, int, int, int],
]:

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Validate normalized coordinates
    # --------------------------------------------------------

    if not (
        0.0 <= x1 <= 1.0
        and 0.0 <= y1 <= 1.0
        and 0.0 <= x2 <= 1.0
        and 0.0 <= y2 <= 1.0
    ):
        raise ValueError(
            f"Bounding box outside [0,1]: {bbox}"
        )

    if x2 < x1 or y2 < y1:
        raise ValueError(
            f"Invalid bounding box: {bbox}"
        )

    # --------------------------------------------------------
    # Pixel boundaries in normalized coordinates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Exact overlap lengths
    # --------------------------------------------------------

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

    # Each pixel gets the exact fraction of its area
    # covered by the bbox.
    weights = (
        y_overlap[:, None]
        * x_overlap[None, :]
    )

    bbox_area = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    # --------------------------------------------------------
    # Pixel index envelope
    # --------------------------------------------------------

    px0 = int(
        np.floor(
            x1 * width
        )
    )

    px1 = int(
        np.ceil(
            x2 * width
        )
    )

    py0 = int(
        np.floor(
            y1 * height
        )
    )

    py1 = int(
        np.ceil(
            y2 * height
        )
    )

    px0 = max(
        0,
        min(width, px0),
    )

    px1 = max(
        0,
        min(width, px1),
    )

    py0 = max(
        0,
        min(height, py0),
    )

    py1 = max(
        0,
        min(height, py1),
    )

    envelope = (
        px0,
        py0,
        px1,
        py1,
    )

    return (
        weights,
        bbox_area,
        envelope,
    )


# ============================================================
# CLC AREA DISTRIBUTION INSIDE BBOX
# ============================================================

def calculate_clc_distribution(
    clc_map: np.ndarray,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> dict:

    height, width = clc_map.shape

    (
        weights,
        bbox_area,
        envelope,
    ) = bbox_pixel_weights(
        height,
        width,
        bbox,
    )

    total_weight = float(
        weights.sum()
    )

    if bbox_area <= 0.0:
        raise ValueError(
            f"Zero-area bbox: {bbox}"
        )

    # The pixel overlap calculation should equal
    # the normalized bbox area.
    area_error = abs(
        total_weight
        - bbox_area
    )

    if area_error > 1e-12:
        raise RuntimeError(
            "BBox area-weight calculation failed:\n"
            f"bbox={bbox}\n"
            f"computed={total_weight}\n"
            f"expected={bbox_area}\n"
            f"error={area_error}"
        )

    # --------------------------------------------------------
    # Weighted CLC areas
    # --------------------------------------------------------

    weighted_areas: dict[int, float] = {}

    unique_values = np.unique(
        clc_map
    )

    for value in unique_values:

        value = int(value)

        mask = (
            clc_map == value
        )

        area = float(
            weights[mask].sum()
        )

        if area > 0.0:

            weighted_areas[value] = area

    # --------------------------------------------------------
    # Convert to fractions
    # --------------------------------------------------------

    fractions: dict[int, float] = {}

    for value, area in weighted_areas.items():

        fractions[value] = (
            area / bbox_area
        )

    # Sort descending.
    sorted_classes = sorted(
        fractions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # Top classes
    # --------------------------------------------------------

    if sorted_classes:

        top_clc = sorted_classes[0][0]
        top_fraction = sorted_classes[0][1]

    else:

        top_clc = None
        top_fraction = 0.0

    second_clc = (
        sorted_classes[1][0]
        if len(sorted_classes) >= 2
        else None
    )

    second_fraction = (
        sorted_classes[1][1]
        if len(sorted_classes) >= 2
        else 0.0
    )

    # --------------------------------------------------------
    # Number of semantic classes inside bbox
    # --------------------------------------------------------

    class_count = len(
        sorted_classes
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = 0.0

    for fraction in fractions.values():

        if fraction > 0.0:

            entropy -= (
                fraction
                * np.log(
                    fraction
                )
            )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "bbox_area_normalized": bbox_area,
        "bbox_weight_area": total_weight,
        "bbox_area_error": area_error,
        "envelope_x0": envelope[0],
        "envelope_y0": envelope[1],
        "envelope_x1": envelope[2],
        "envelope_y1": envelope[3],
        "top_clc": top_clc,
        "top_fraction": top_fraction,
        "second_clc": second_clc,
        "second_fraction": second_fraction,
        "clc_class_count": class_count,
        "clc_entropy": entropy,
        "clc_fractions": {
            str(k): float(v)
            for k, v in sorted_classes
        },
    }


# ============================================================
# PHRASE HELPERS
# ============================================================

def normalize_phrase(
    phrase: str,
) -> str:

    phrase = str(
        phrase
    ).strip().lower()

    phrase = re.sub(
        r"\s+",
        " ",
        phrase,
    )

    return phrase


def detect_comparator(
    phrase: str,
) -> str:

    p = normalize_phrase(
        phrase
    )

    if re.search(
        r"\blargest\b",
        p,
    ):
        return "largest"

    if re.search(
        r"\bsmallest\b",
        p,
    ):
        return "smallest"

    return "ordinary"


def detect_continuity(
    phrase: str,
) -> str:

    p = normalize_phrase(
        phrase
    )

    if (
        "continuous" in p
        or "contiguous" in p
    ):
        return "continuous"

    return "not_explicit"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 PHRASE -> CLC REFERENCE-MAP AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load rows
    # --------------------------------------------------------

    df = load_reference_rows()

    print(
        f"Rows loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Basic structure checks
    # --------------------------------------------------------

    required_columns = {
        "patch_id",
        "input",
        "output",
        "type",
        "category",
        "split",
        "original_row_index",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Ensure reference grounding rows only
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["category"]
        .astype(str)
        .str.lower()
        .eq("reference")
    ].copy()

    print(
        f"Reference-category rows: {len(df)} "
        f"(from {before})"
    )

    # --------------------------------------------------------
    # Parse reference phrase and bbox
    # --------------------------------------------------------

    df["reference_phrase"] = (
        df["input"]
        .astype(str)
        .map(
            extract_reference_phrase
        )
    )

    df["parsed_bbox"] = (
        df["output"]
        .astype(str)
        .map(parse_bbox)
    )

    missing_phrase = int(
        df["reference_phrase"]
        .isna()
        .sum()
    )

    missing_bbox = int(
        df["parsed_bbox"]
        .isna()
        .sum()
    )

    print(
        f"Rows with missing reference phrase: "
        f"{missing_phrase}"
    )

    print(
        f"Rows with missing bbox: "
        f"{missing_bbox}"
    )

    if missing_phrase:
        raise RuntimeError(
            "Reference phrase parsing failed."
        )

    if missing_bbox:
        raise RuntimeError(
            "Bounding-box parsing failed."
        )

    # --------------------------------------------------------
    # Phrase features
    # --------------------------------------------------------

    df["phrase_normalized"] = (
        df["reference_phrase"]
        .map(normalize_phrase)
    )

    df["comparator"] = (
        df["reference_phrase"]
        .map(detect_comparator)
    )

    df["continuity"] = (
        df["reference_phrase"]
        .map(detect_continuity)
    )

    # --------------------------------------------------------
    # Map index
    # --------------------------------------------------------

    map_index = build_map_index()

    print(
        f"Reference maps indexed: "
        f"{len(map_index)}"
    )

    # --------------------------------------------------------
    # Patch coverage
    # --------------------------------------------------------

    patch_ids = (
        df["patch_id"]
        .astype(str)
        .unique()
    )

    print(
        f"Unique reference patches in rows: "
        f"{len(patch_ids)}"
    )

    missing_maps = [
        patch_id
        for patch_id in patch_ids
        if patch_id not in map_index
    ]

    print(
        f"Missing reference maps: "
        f"{len(missing_maps)}"
    )

    if missing_maps:
        print(
            "\nFirst missing patch IDs:"
        )

        for patch_id in missing_maps[:20]:
            print(
                patch_id
            )

        raise RuntimeError(
            "Some reference patches have no reference map."
        )

    # --------------------------------------------------------
    # Prepare output columns
    # --------------------------------------------------------

    row_results = []

    # --------------------------------------------------------
    # Process by patch
    # --------------------------------------------------------

    grouped = df.groupby(
        "patch_id",
        sort=True,
    )

    total_patches = len(
        grouped
    )

    print(
        f"\nProcessing {total_patches} "
        f"patches..."
    )

    for patch_number, (
        patch_id,
        patch_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        map_path = map_index[
            str(patch_id)
        ]

        with rasterio.open(
            map_path
        ) as src:

            clc_map = src.read(1)

        if clc_map.shape != (
            120,
            120,
        ):
            raise RuntimeError(
                "Unexpected reference-map shape:\n"
                f"patch={patch_id}\n"
                f"shape={clc_map.shape}"
            )

        # ----------------------------------------------------
        # Process each reference row
        # ----------------------------------------------------

        for _, row in patch_df.iterrows():

            bbox = row[
                "parsed_bbox"
            ]

            distribution = (
                calculate_clc_distribution(
                    clc_map,
                    bbox,
                )
            )

            top_fraction = (
                distribution[
                    "top_fraction"
                ]
            )

            second_fraction = (
                distribution[
                    "second_fraction"
                ]
            )

            # ------------------------------------------------
            # Row result
            # ------------------------------------------------

            result = {
                "original_row_index": int(
                    row[
                        "original_row_index"
                    ]
                ),
                "ID": int(
                    row["ID"]
                ),
                "audit_split": str(
                    row["audit_split"]
                ),
                "patch_id": str(
                    patch_id
                ),
                "reference_phrase": str(
                    row[
                        "reference_phrase"
                    ]
                ),
                "phrase_normalized": str(
                    row[
                        "phrase_normalized"
                    ]
                ),
                "comparator": str(
                    row["comparator"]
                ),
                "continuity": str(
                    row["continuity"]
                ),
                "bbox_x1": float(
                    bbox[0]
                ),
                "bbox_y1": float(
                    bbox[1]
                ),
                "bbox_x2": float(
                    bbox[2]
                ),
                "bbox_y2": float(
                    bbox[3]
                ),
                "top_clc": (
                    distribution[
                        "top_clc"
                    ]
                ),
                "top_fraction": (
                    top_fraction
                ),
                "second_clc": (
                    distribution[
                        "second_clc"
                    ]
                ),
                "second_fraction": (
                    second_fraction
                ),
                "clc_class_count": (
                    distribution[
                        "clc_class_count"
                    ]
                ),
                "clc_entropy": (
                    distribution[
                        "clc_entropy"
                    ]
                ),
                "bbox_area_normalized": (
                    distribution[
                        "bbox_area_normalized"
                    ]
                ),
                "bbox_area_error": (
                    distribution[
                        "bbox_area_error"
                    ]
                ),
                "clc_fractions_json": json.dumps(
                    distribution[
                        "clc_fractions"
                    ],
                    sort_keys=True,
                ),
            }

            row_results.append(
                result
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            patch_number <= 10
            or patch_number % 500 == 0
            or patch_number == total_patches
        ):

            print(
                f"Processed "
                f"{patch_number}/"
                f"{total_patches} patches"
            )

    # ========================================================
    # ROW-LEVEL DATAFRAME
    # ========================================================

    audit_df = pd.DataFrame(
        row_results
    )

    if audit_df.empty:
        raise RuntimeError(
            "No row-level audit results produced."
        )

    # --------------------------------------------------------
    # Global validation
    # --------------------------------------------------------

    print("\nValidating results...")

    if len(audit_df) != len(df):
        raise RuntimeError(
            "Row count changed during audit:\n"
            f"Expected: {len(df)}\n"
            f"Got: {len(audit_df)}"
        )

    if (
        audit_df[
            "bbox_area_error"
        ].abs()
        > 1e-12
    ).any():

        raise RuntimeError(
            "At least one bbox area calculation "
            "failed exact conservation."
        )

    if (
        audit_df[
            "top_fraction"
        ] < 0
    ).any():

        raise RuntimeError(
            "Negative CLC fraction detected."
        )

    if (
        audit_df[
            "top_fraction"
        ] > 1 + 1e-12
    ).any():

        raise RuntimeError(
            "CLC fraction > 1 detected."
        )

    # ========================================================
    # SAVE ROW-LEVEL AUDIT
    # ========================================================

    audit_df.to_csv(
        ROW_OUTPUT,
        index=False,
    )

    # ========================================================
    # PARSE CLC FRACTIONS FOR LONG-FORM MATRIX
    # ========================================================

    matrix_rows = []

    for _, row in audit_df.iterrows():

        fractions = json.loads(
            row[
                "clc_fractions_json"
            ]
        )

        for clc_value, fraction in fractions.items():

            matrix_rows.append(
                {
                    "original_row_index": row[
                        "original_row_index"
                    ],
                    "patch_id": row[
                        "patch_id"
                    ],
                    "reference_phrase": row[
                        "reference_phrase"
                    ],
                    "phrase_normalized": row[
                        "phrase_normalized"
                    ],
                    "comparator": row[
                        "comparator"
                    ],
                    "clc_value": int(
                        clc_value
                    ),
                    "bbox_fraction": float(
                        fraction
                    ),
                }
            )

    matrix_df = pd.DataFrame(
        matrix_rows
    )

    matrix_df.to_csv(
        CLC_OUTPUT,
        index=False,
    )

    # ========================================================
    # PHRASE SUMMARY
    # ========================================================

    phrase_rows = []

    phrase_grouped = audit_df.groupby(
        [
            "phrase_normalized",
            "comparator",
            "continuity",
        ],
        sort=True,
    )

    for (
        phrase_key,
        phrase_df,
    ) in phrase_grouped:

        phrase, comparator, continuity = (
            phrase_key
        )

        # --------------------------------------------
        # Top CLC counts
        # --------------------------------------------

        top_counts = Counter(
            phrase_df[
                "top_clc"
            ].tolist()
        )

        # --------------------------------------------
        # Top CLC weighted mean fractions
        # --------------------------------------------

        weighted_by_clc = (
            matrix_df[
                matrix_df[
                    "phrase_normalized"
                ] == phrase
            ]
            .groupby(
                "clc_value"
            )[
                "bbox_fraction"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if len(weighted_by_clc):

            dominant_clc = int(
                weighted_by_clc.index[0]
            )

            dominant_mean_fraction = float(
                weighted_by_clc.iloc[0]
            )

        else:

            dominant_clc = None
            dominant_mean_fraction = 0.0

        # --------------------------------------------
        # Row-level dominant CLC
        # --------------------------------------------

        dominant_row_clc = (
            top_counts.most_common(
                1
            )[0][0]
            if top_counts
            else None
        )

        dominant_row_count = (
            top_counts.get(
                dominant_row_clc,
                0,
            )
            if dominant_row_clc is not None
            else 0
        )

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        phrase_rows.append(
            {
                "phrase_normalized": phrase,
                "comparator": comparator,
                "continuity": continuity,
                "rows": len(
                    phrase_df
                ),
                "unique_patches": (
                    phrase_df[
                        "patch_id"
                    ]
                    .nunique()
                ),
                "dominant_row_clc": (
                    dominant_row_clc
                ),
                "dominant_row_count": (
                    dominant_row_count
                ),
                "dominant_row_share": (
                    dominant_row_count
                    / len(phrase_df)
                ),
                "mean_top_fraction": float(
                    phrase_df[
                        "top_fraction"
                    ].mean()
                ),
                "median_top_fraction": float(
                    phrase_df[
                        "top_fraction"
                    ].median()
                ),
                "mean_second_fraction": float(
                    phrase_df[
                        "second_fraction"
                    ].mean()
                ),
                "mean_clc_class_count": float(
                    phrase_df[
                        "clc_class_count"
                    ].mean()
                ),
                "median_clc_class_count": float(
                    phrase_df[
                        "clc_class_count"
                    ].median()
                ),
                "mean_clc_entropy": float(
                    phrase_df[
                        "clc_entropy"
                    ].mean()
                ),
                "dominant_mean_fraction_clc": (
                    dominant_clc
                ),
                "dominant_mean_fraction": (
                    dominant_mean_fraction
                ),
            }
        )

    phrase_summary_df = pd.DataFrame(
        phrase_rows
    )

    phrase_summary_df = (
        phrase_summary_df
        .sort_values(
            [
                "comparator",
                "phrase_normalized",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    phrase_summary_df.to_csv(
        PHRASE_OUTPUT,
        index=False,
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary = {
        "train_rows": int(
            len(
                pd.read_parquet(
                    TRAIN_PARQUET
                )
            )
        ),
        "validation_rows": int(
            len(
                pd.read_parquet(
                    VAL_PARQUET
                )
            )
        ),
        "reference_rows_audited": int(
            len(audit_df)
        ),
        "unique_patches": int(
            audit_df[
                "patch_id"
            ].nunique()
        ),
        "unique_phrases": int(
            audit_df[
                "phrase_normalized"
            ].nunique()
        ),
        "unique_top_clc_values": int(
            audit_df[
                "top_clc"
            ].nunique()
        ),
        "missing_maps": len(
            missing_maps
        ),
        "missing_reference_phrases": (
            missing_phrase
        ),
        "missing_bboxes": (
            missing_bbox
        ),
        "max_bbox_area_error": float(
            audit_df[
                "bbox_area_error"
            ].abs().max()
        ),
        "mean_top_fraction": float(
            audit_df[
                "top_fraction"
            ].mean()
        ),
        "median_top_fraction": float(
            audit_df[
                "top_fraction"
            ].median()
        ),
        "mean_clc_class_count": float(
            audit_df[
                "clc_class_count"
            ].mean()
        ),
        "median_clc_class_count": float(
            audit_df[
                "clc_class_count"
            ].median()
        ),
        "mean_clc_entropy": float(
            audit_df[
                "clc_entropy"
            ].mean()
        ),
        "comparator_counts": {
            str(k): int(v)
            for k, v in audit_df[
                "comparator"
            ].value_counts().items()
        },
        "output_files": {
            "row_audit": str(
                ROW_OUTPUT
            ),
            "phrase_summary": str(
                PHRASE_OUTPUT
            ),
            "clc_matrix": str(
                CLC_OUTPUT
            ),
        },
    }

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("C3 PHRASE -> CLC AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"Reference rows audited: "
        f"{len(audit_df)}"
    )

    print(
        f"Unique patches: "
        f"{audit_df['patch_id'].nunique()}"
    )

    print(
        f"Unique phrases: "
        f"{audit_df['phrase_normalized'].nunique()}"
    )

    print(
        f"Unique dominant CLC values: "
        f"{audit_df['top_clc'].nunique()}"
    )

    print(
        f"Mean top CLC fraction: "
        f"{audit_df['top_fraction'].mean():.6f}"
    )

    print(
        f"Median top CLC fraction: "
        f"{audit_df['top_fraction'].median():.6f}"
    )

    print(
        f"Mean CLC classes per bbox: "
        f"{audit_df['clc_class_count'].mean():.6f}"
    )

    print(
        f"Median CLC classes per bbox: "
        f"{audit_df['clc_class_count'].median():.6f}"
    )

    print(
        f"Maximum bbox area error: "
        f"{audit_df['bbox_area_error'].abs().max():.12g}"
    )

    print(
        f"\nRow audit:\n{ROW_OUTPUT}"
    )

    print(
        f"Phrase summary:\n{PHRASE_OUTPUT}"
    )

    print(
        f"CLC matrix:\n{CLC_OUTPUT}"
    )

    print(
        f"Summary:\n{SUMMARY_OUTPUT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()