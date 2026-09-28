from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError(
        "scipy is required for connected-component analysis. "
        "Install it in the active SatQueryAI virtual environment with:\n"
        "pip install scipy"
    ) from exc


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

AUDIT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "c3_phrase_clc_audit"
)

ROW_FILE = (
    AUDIT_DIR
    / "reference_row_clc_audit.csv"
)

MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_FILE = (
    AUDIT_DIR
    / "component_selection_results.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "component_selection_summary.json"
)

FAMILY_SUMMARY_FILE = (
    AUDIT_DIR
    / "component_selection_by_family.csv"
)

K_SUMMARY_FILE = (
    AUDIT_DIR
    / "component_selection_by_k.csv"
)

TOP_K_VALUES = [1, 2, 3, 4, 5]

IMAGE_SIZE = 120

# 8-connected components:
# horizontal, vertical, and diagonal neighbors are connected.
CONNECTIVITY_STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8,
)


# ============================================================
# SEMANTIC-FAMILY NORMALIZATION
# ============================================================

def normalize_semantic_family(
    phrase: str,
) -> str:

    text = str(
        phrase
    ).strip().lower()

    patterns = [
        r"\blargest\b",
        r"\bsmallest\b",
        r"\bcontinuous\b",
        r"\bcontiguous\b",
        r"\bconnected\b",
        r"\bpatch\b",
        r"\bregion\b",
        r"\barea\b",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    text = re.sub(
        r"^of\s+",
        "",
        text,
    )

    text = re.sub(
        r"^a\s+",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# PARSE CLC FRACTIONS
# ============================================================

def parse_clc_distribution(
    text: str,
) -> dict[int, float]:

    value = json.loads(
        str(text)
    )

    return {
        int(k): float(v)
        for k, v in value.items()
    }


# ============================================================
# LOAD AUDIT DATA
# ============================================================

def load_rows() -> pd.DataFrame:

    if not ROW_FILE.exists():
        raise FileNotFoundError(
            f"Missing audit file:\n{ROW_FILE}"
        )

    df = pd.read_csv(
        ROW_FILE
    )

    required = {
        "original_row_index",
        "patch_id",
        "reference_phrase",
        "audit_split",
        "top_clc",
        "clc_fractions_json",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    # Reconstruct the same 19 semantic families used
    # in the previous analyses.
    df["semantic_family"] = (
        df["reference_phrase"]
        .map(
            normalize_semantic_family
        )
    )

    df["clc_distribution"] = (
        df["clc_fractions_json"]
        .map(
            parse_clc_distribution
        )
    )

    df["audit_split"] = (
        df["audit_split"]
        .astype(str)
        .str.lower()
    )

    return df


# ============================================================
# BUILD MAP INDEX
# ============================================================

def build_map_index() -> dict[str, Path]:

    if not MAP_ROOT.exists():
        raise FileNotFoundError(
            f"Reference-map root does not exist:\n"
            f"{MAP_ROOT}"
        )

    print(
        "Indexing reference maps..."
    )

    index = {}

    for path in MAP_ROOT.rglob(
        "*_reference_map.tif"
    ):

        suffix = "_reference_map.tif"

        if not path.name.endswith(
            suffix
        ):
            continue

        patch_id = path.name[
            :-len(suffix)
        ]

        if patch_id in index:
            raise RuntimeError(
                "Duplicate reference map found:\n"
                f"{patch_id}\n"
                f"{index[patch_id]}\n"
                f"{path}"
            )

        index[patch_id] = path

    return index


# ============================================================
# LEARN TRAIN-ONLY FAMILY -> CLC RANKING
# ============================================================

def learn_family_profiles(
    train_df: pd.DataFrame,
) -> dict[str, list[int]]:

    profiles = {}

    for family, family_df in train_df.groupby(
        "semantic_family",
        sort=True,
    ):

        sums: dict[int, float] = {}
        counts: dict[int, int] = {}

        # IMPORTANT:
        # This deliberately reproduces the same ranking
        # definition used by the preceding train->validation
        # CLC-transfer experiment:
        #
        # mean fraction is calculated over rows in which
        # the particular CLC code is present.

        for distribution in (
            family_df[
                "clc_distribution"
            ]
        ):

            for clc_value, fraction in (
                distribution.items()
            ):

                clc_value = int(
                    clc_value
                )

                sums[clc_value] = (
                    sums.get(
                        clc_value,
                        0.0,
                    )
                    + float(fraction)
                )

                counts[clc_value] = (
                    counts.get(
                        clc_value,
                        0,
                    )
                    + 1
                )

        means = {
            clc_value: (
                sums[clc_value]
                / counts[clc_value]
            )
            for clc_value in sums
        }

        ranked = sorted(
            means.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        profiles[str(family)] = [
            int(clc_value)
            for clc_value, _ in ranked
        ]

    return profiles


# ============================================================
# EXACT BBOX PIXEL WEIGHTS
# ============================================================

def bbox_pixel_weights(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    height: int,
    width: int,
) -> tuple[np.ndarray, float]:

    x1, y1, x2, y2 = bbox

    if not (
        0.0 <= x1 <= 1.0
        and 0.0 <= y1 <= 1.0
        and 0.0 <= x2 <= 1.0
        and 0.0 <= y2 <= 1.0
    ):
        raise ValueError(
            f"Invalid normalized bbox:\n{bbox}"
        )

    if x2 < x1 or y2 < y1:
        raise ValueError(
            f"Invalid bbox ordering:\n{bbox}"
        )

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

    computed_area = float(
        weights.sum()
    )

    error = abs(
        computed_area
        - bbox_area
    )

    if error > 1e-12:
        raise RuntimeError(
            "BBox area calculation failed:\n"
            f"bbox={bbox}\n"
            f"computed={computed_area}\n"
            f"expected={bbox_area}\n"
            f"error={error}"
        )

    return (
        weights,
        bbox_area,
    )


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

    if pixel_count <= 0:

        return {
            "pixel_count": 0,
            "area": 0.0,
            "intersection": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "bbox_x1": None,
            "bbox_y1": None,
            "bbox_x2": None,
            "bbox_y2": None,
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
        if component_area > 0.0
        else 0.0
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

    ys, xs = np.where(
        component_mask
    )

    x1 = float(
        xs.min()
        / IMAGE_SIZE
    )

    y1 = float(
        ys.min()
        / IMAGE_SIZE
    )

    # Pixel-aligned exclusive upper edge.
    x2 = float(
        (
            xs.max()
            + 1
        )
        / IMAGE_SIZE
    )

    y2 = float(
        (
            ys.max()
            + 1
        )
        / IMAGE_SIZE
    )

    return {
        "pixel_count": pixel_count,
        "area": component_area,
        "intersection": intersection,
        "precision": float(
            np.clip(
                precision,
                0.0,
                1.0,
            )
        ),
        "recall": float(
            np.clip(
                recall,
                0.0,
                1.0,
            )
        ),
        "iou": float(
            np.clip(
                iou,
                0.0,
                1.0,
            )
        ),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
    }


# ============================================================
# ANALYZE ONE CANDIDATE MASK
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

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    labels, component_count = (
        ndimage.label(
            candidate_mask,
            structure=CONNECTIVITY_STRUCTURE,
        )
    )

    component_sizes = np.bincount(
        labels.ravel()
    )

    # Ignore label 0 (background).
    component_sizes = component_sizes[
        1:
    ]

    if component_count == 0:

        empty = {
            "pixel_count": 0,
            "area": 0.0,
            "intersection": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "bbox_x1": None,
            "bbox_y1": None,
            "bbox_x2": None,
            "bbox_y2": None,
        }

        return {
            "candidate_pixels": 0,
            "candidate_area": 0.0,
            "candidate_intersection": 0.0,
            "candidate_precision": 0.0,
            "candidate_recall": 0.0,
            "component_count": 0,
            "largest": empty,
            "smallest": empty,
            "best_oracle": empty,
        }

    # --------------------------------------------------------
    # Largest component
    # --------------------------------------------------------

    largest_component_number = (
        int(
            np.argmax(
                component_sizes
            )
        )
        + 1
    )

    largest_mask = (
        labels
        == largest_component_number
    )

    largest_metrics = component_metrics(
        largest_mask,
        bbox_weights,
        bbox_area,
    )

    # --------------------------------------------------------
    # Smallest component
    # --------------------------------------------------------

    smallest_component_number = (
        int(
            np.argmin(
                component_sizes
            )
        )
        + 1
    )

    smallest_mask = (
        labels
        == smallest_component_number
    )

    smallest_metrics = component_metrics(
        smallest_mask,
        bbox_weights,
        bbox_area,
    )

    # --------------------------------------------------------
    # Oracle best-overlap component
    #
    # This is NOT a deployable selector.
    # It answers:
    # "Does a good component exist somewhere in the
    # candidate semantic mask?"
    #
    # This lets us separate semantic-candidate quality
    # from deterministic component-selection difficulty.
    # --------------------------------------------------------

    best_metrics = None
    best_label = None

    for component_number in range(
        1,
        component_count + 1,
    ):

        component_mask = (
            labels
            == component_number
        )

        metrics = component_metrics(
            component_mask,
            bbox_weights,
            bbox_area,
        )

        if (
            best_metrics is None
            or metrics["iou"]
            > best_metrics["iou"]
        ):

            best_metrics = metrics
            best_label = (
                component_number
            )

    if best_metrics is None:
        raise RuntimeError(
            "Connected-component analysis "
            "produced no component metrics."
        )

    return {
        "candidate_pixels": candidate_pixels,
        "candidate_area": candidate_area,
        "candidate_intersection": (
            candidate_intersection
        ),
        "candidate_precision": float(
            np.clip(
                candidate_precision,
                0.0,
                1.0,
            )
        ),
        "candidate_recall": float(
            np.clip(
                candidate_recall,
                0.0,
                1.0,
            )
        ),
        "component_count": int(
            component_count
        ),
        "largest": largest_metrics,
        "smallest": smallest_metrics,
        "best_oracle": best_metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 COMPONENT SELECTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load rows
    # --------------------------------------------------------

    df = load_rows()

    print(
        f"Rows loaded: {len(df)}"
    )

    train = df[
        df["audit_split"]
        == "train"
    ].copy()

    val = df[
        df["audit_split"]
        == "val"
    ].copy()

    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Validation rows: {len(val)}"
    )

    print(
        f"Validation patches: "
        f"{val['patch_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Verify no train/val patch overlap
    # --------------------------------------------------------

    train_patches = set(
        train["patch_id"]
        .astype(str)
        .unique()
    )

    val_patches = set(
        val["patch_id"]
        .astype(str)
        .unique()
    )

    overlap = (
        train_patches
        & val_patches
    )

    print(
        f"Train/validation patch overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train/validation patch overlap detected."
        )

    # --------------------------------------------------------
    # Train-only family profiles
    # --------------------------------------------------------

    print(
        "\nLearning CLC profiles from TRAIN ONLY..."
    )

    family_profiles = (
        learn_family_profiles(
            train
        )
    )

    print(
        f"Families learned: "
        f"{len(family_profiles)}"
    )

    # --------------------------------------------------------
    # Build map index
    # --------------------------------------------------------

    map_index = build_map_index()

    print(
        f"Reference maps indexed: "
        f"{len(map_index)}"
    )

    missing_maps = (
        set(
            val["patch_id"]
            .astype(str)
            .unique()
        )
        - set(
            map_index.keys()
        )
    )

    print(
        f"Missing validation maps: "
        f"{len(missing_maps)}"
    )

    if missing_maps:
        raise RuntimeError(
            "Validation maps missing."
        )

    # ========================================================
    # CACHE MAP ANALYSIS
    # ========================================================

    # Each patch may have multiple reference rows.
    # We therefore read each map once and reuse it.
    map_cache = {}

    # ========================================================
    # ROW-LEVEL RESULTS
    # ========================================================

    results = []

    processed_rows = 0

    # --------------------------------------------------------
    # Process patch-by-patch
    # --------------------------------------------------------

    for patch_number, (
        patch_id,
        patch_df,
    ) in enumerate(
        val.groupby(
            "patch_id",
            sort=True,
        ),
        start=1,
    ):

        patch_id = str(
            patch_id
        )

        map_path = map_index[
            patch_id
        ]

        if patch_id not in map_cache:

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
                    "Unexpected CLC map shape:\n"
                    f"Patch: {patch_id}\n"
                    f"Shape: {clc_map.shape}"
                )

            map_cache[
                patch_id
            ] = clc_map

        else:

            clc_map = map_cache[
                patch_id
            ]

        # ----------------------------------------------------
        # Process rows for this patch
        # ----------------------------------------------------

        for _, row in patch_df.iterrows():

            family = str(
                row[
                    "semantic_family"
                ]
            )

            if family not in family_profiles:
                raise RuntimeError(
                    "Validation family missing "
                    "from training profile:\n"
                    f"{family}"
                )

            ranked_clc = (
                family_profiles[
                    family
                ]
            )

            bbox = (
                float(
                    row["bbox_x1"]
                ),
                float(
                    row["bbox_y1"]
                ),
                float(
                    row["bbox_x2"]
                ),
                float(
                    row["bbox_y2"]
                ),
            )

            bbox_weights, bbox_area = (
                bbox_pixel_weights(
                    bbox,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            )

            comparator = str(
                row[
                    "reference_phrase"
                ]
            )

            # Determine comparator from semantic phrase.
            phrase_lower = comparator.lower()

            if re.search(
                r"\blargest\b",
                phrase_lower,
            ):
                comparator_type = "largest"

            elif re.search(
                r"\bsmallest\b",
                phrase_lower,
            ):
                comparator_type = "smallest"

            else:
                comparator_type = "ordinary"

            # ------------------------------------------------
            # K = 1..5
            # ------------------------------------------------

            for k in TOP_K_VALUES:

                candidate_codes = ranked_clc[
                    :k
                ]

                candidate_set = set(
                    candidate_codes
                )

                candidate_mask = np.isin(
                    clc_map,
                    list(
                        candidate_set
                    ),
                )

                analysis = (
                    analyze_candidate_mask(
                        candidate_mask,
                        bbox_weights,
                        bbox_area,
                    )
                )

                largest = (
                    analysis["largest"]
                )

                smallest = (
                    analysis["smallest"]
                )

                best = (
                    analysis["best_oracle"]
                )

                selected = (
                    largest
                    if comparator_type
                    == "largest"
                    else smallest
                    if comparator_type
                    == "smallest"
                    else best
                )

                results.append(
                    {
                        "original_row_index": int(
                            row[
                                "original_row_index"
                            ]
                        ),
                        "patch_id": patch_id,
                        "reference_phrase": str(
                            row[
                                "reference_phrase"
                            ]
                        ),
                        "semantic_family": family,
                        "comparator": comparator_type,
                        "k": int(k),
                        "candidate_clc_codes": json.dumps(
                            candidate_codes
                        ),
                        "candidate_component_count": int(
                            analysis[
                                "component_count"
                            ]
                        ),
                        "candidate_pixels": int(
                            analysis[
                                "candidate_pixels"
                            ]
                        ),
                        "candidate_area": float(
                            analysis[
                                "candidate_area"
                            ]
                        ),
                        "candidate_coverage_recall": float(
                            analysis[
                                "candidate_recall"
                            ]
                        ),
                        "candidate_precision": float(
                            analysis[
                                "candidate_precision"
                            ]
                        ),

                        # Largest component.
                        "largest_pixels": int(
                            largest[
                                "pixel_count"
                            ]
                        ),
                        "largest_iou": float(
                            largest[
                                "iou"
                            ]
                        ),
                        "largest_precision": float(
                            largest[
                                "precision"
                            ]
                        ),
                        "largest_recall": float(
                            largest[
                                "recall"
                            ]
                        ),

                        # Smallest component.
                        "smallest_pixels": int(
                            smallest[
                                "pixel_count"
                            ]
                        ),
                        "smallest_iou": float(
                            smallest[
                                "iou"
                            ]
                        ),
                        "smallest_precision": float(
                            smallest[
                                "precision"
                            ]
                        ),
                        "smallest_recall": float(
                            smallest[
                                "recall"
                            ]
                        ),

                        # Oracle component.
                        "oracle_best_iou": float(
                            best[
                                "iou"
                            ]
                        ),
                        "oracle_best_precision": float(
                            best[
                                "precision"
                            ]
                        ),
                        "oracle_best_recall": float(
                            best[
                                "recall"
                            ]
                        ),

                        # Actual deterministic selection.
                        "selected_iou": float(
                            selected[
                                "iou"
                            ]
                        ),
                        "selected_precision": float(
                            selected[
                                "precision"
                            ]
                        ),
                        "selected_recall": float(
                            selected[
                                "recall"
                            ]
                        ),

                        "bbox_area": float(
                            bbox_area
                        ),
                    }
                )

            processed_rows += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            patch_number <= 10
            or patch_number % 250 == 0
            or patch_number
            == val["patch_id"].nunique()
        ):

            print(
                f"Processed patches: "
                f"{patch_number}/"
                f"{val['patch_id'].nunique()} | "
                f"rows: "
                f"{processed_rows}/"
                f"{len(val)}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    expected_rows = (
        len(val)
        * len(TOP_K_VALUES)
    )

    if len(result_df) != expected_rows:
        raise RuntimeError(
            "Unexpected result row count:\n"
            f"Expected: {expected_rows}\n"
            f"Got: {len(result_df)}"
        )

    # ========================================================
    # GLOBAL K SUMMARY
    # ========================================================

    k_summaries = []

    for k in TOP_K_VALUES:

        subset = result_df[
            result_df["k"]
            == k
        ]

        k_summaries.append(
            {
                "k": int(k),
                "rows": int(
                    len(subset)
                ),

                "mean_candidate_coverage": float(
                    subset[
                        "candidate_coverage_recall"
                    ].mean()
                ),

                "median_candidate_coverage": float(
                    subset[
                        "candidate_coverage_recall"
                    ].median()
                ),

                "mean_candidate_precision": float(
                    subset[
                        "candidate_precision"
                    ].mean()
                ),

                "median_candidate_precision": float(
                    subset[
                        "candidate_precision"
                    ].median()
                ),

                "mean_component_count": float(
                    subset[
                        "candidate_component_count"
                    ].mean()
                ),

                "median_component_count": float(
                    subset[
                        "candidate_component_count"
                    ].median()
                ),

                "mean_largest_iou": float(
                    subset[
                        "largest_iou"
                    ].mean()
                ),

                "median_largest_iou": float(
                    subset[
                        "largest_iou"
                    ].median()
                ),

                "mean_smallest_iou": float(
                    subset[
                        "smallest_iou"
                    ].mean()
                ),

                "median_smallest_iou": float(
                    subset[
                        "smallest_iou"
                    ].median()
                ),

                "mean_oracle_iou": float(
                    subset[
                        "oracle_best_iou"
                    ].mean()
                ),

                "median_oracle_iou": float(
                    subset[
                        "oracle_best_iou"
                    ].median()
                ),

                "mean_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].mean()
                ),

                "median_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].median()
                ),

                "fraction_selected_ge_025": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.25
                    ).mean()
                ),

                "fraction_selected_ge_050": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.50
                    ).mean()
                ),

                "fraction_oracle_ge_025": float(
                    (
                        subset[
                            "oracle_best_iou"
                        ]
                        >= 0.25
                    ).mean()
                ),

                "fraction_oracle_ge_050": float(
                    (
                        subset[
                            "oracle_best_iou"
                        ]
                        >= 0.50
                    ).mean()
                ),
            }
        )

    k_summary_df = pd.DataFrame(
        k_summaries
    )

    # ========================================================
    # COMPARATOR × K SUMMARY
    # ========================================================

    comparator_summaries = []

    for (
        comparator,
        k,
    ), subset in result_df.groupby(
        [
            "comparator",
            "k",
        ],
        sort=True,
    ):

        comparator_summaries.append(
            {
                "comparator": comparator,
                "k": int(k),
                "rows": int(
                    len(subset)
                ),
                "mean_candidate_coverage": float(
                    subset[
                        "candidate_coverage_recall"
                    ].mean()
                ),
                "mean_candidate_precision": float(
                    subset[
                        "candidate_precision"
                    ].mean()
                ),
                "mean_largest_iou": float(
                    subset[
                        "largest_iou"
                    ].mean()
                ),
                "mean_smallest_iou": float(
                    subset[
                        "smallest_iou"
                    ].mean()
                ),
                "mean_oracle_iou": float(
                    subset[
                        "oracle_best_iou"
                    ].mean()
                ),
                "mean_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].mean()
                ),
                "median_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].median()
                ),
                "fraction_selected_ge_025": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.25
                    ).mean()
                ),
                "fraction_selected_ge_050": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.50
                    ).mean()
                ),
            }
        )

    comparator_df = pd.DataFrame(
        comparator_summaries
    )

    # ========================================================
    # FAMILY × K SUMMARY
    # ========================================================

    family_summaries = []

    for (
        family,
        k,
    ), subset in result_df.groupby(
        [
            "semantic_family",
            "k",
        ],
        sort=True,
    ):

        family_summaries.append(
            {
                "semantic_family": family,
                "k": int(k),
                "rows": int(
                    len(subset)
                ),
                "mean_candidate_coverage": float(
                    subset[
                        "candidate_coverage_recall"
                    ].mean()
                ),
                "mean_candidate_precision": float(
                    subset[
                        "candidate_precision"
                    ].mean()
                ),
                "mean_component_count": float(
                    subset[
                        "candidate_component_count"
                    ].mean()
                ),
                "mean_largest_iou": float(
                    subset[
                        "largest_iou"
                    ].mean()
                ),
                "mean_smallest_iou": float(
                    subset[
                        "smallest_iou"
                    ].mean()
                ),
                "mean_oracle_iou": float(
                    subset[
                        "oracle_best_iou"
                    ].mean()
                ),
                "mean_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].mean()
                ),
                "median_selected_iou": float(
                    subset[
                        "selected_iou"
                    ].median()
                ),
                "fraction_selected_ge_025": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.25
                    ).mean()
                ),
                "fraction_selected_ge_050": float(
                    (
                        subset[
                            "selected_iou"
                        ]
                        >= 0.50
                    ).mean()
                ),
                "mean_selection_gap_vs_oracle": float(
                    (
                        subset[
                            "oracle_best_iou"
                        ]
                        - subset[
                            "selected_iou"
                        ]
                    ).mean()
                ),
            }
        )

    family_df = pd.DataFrame(
        family_summaries
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    family_df.to_csv(
        FAMILY_SUMMARY_FILE,
        index=False,
    )

    k_summary_df.to_csv(
        K_SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # GLOBAL SUMMARY JSON
    # ========================================================

    global_summary = {
        "validation_rows": int(
            len(val)
        ),
        "validation_patches": int(
            val[
                "patch_id"
            ].nunique()
        ),
        "train_rows": int(
            len(train)
        ),
        "train_validation_patch_overlap": int(
            len(overlap)
        ),
        "semantic_families_train": int(
            len(family_profiles)
        ),
        "top_k_values": TOP_K_VALUES,
        "k_summary": k_summaries,
        "comparator_summary": comparator_summaries,
        "output_files": {
            "row_level": str(
                OUTPUT_FILE
            ),
            "family_summary": str(
                FAMILY_SUMMARY_FILE
            ),
            "k_summary": str(
                K_SUMMARY_FILE
            ),
        },
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_summary,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT GLOBAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("GLOBAL COMPONENT-SELECTION RESULTS")
    print("=" * 70)

    print(
        k_summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT COMPARATOR RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPARATOR × K RESULTS")
    print("=" * 70)

    print(
        comparator_df.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT FAMILY RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("FAMILY RESULTS — K=3")
    print("=" * 70)

    print(
        family_df[
            family_df["k"] == 3
        ]
        .sort_values(
            "mean_selected_iou",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # DIAGNOSTIC: SELECTION GAP
    # ========================================================

    print("\n" + "=" * 70)
    print("SELECTION GAP — K=3")
    print("=" * 70)

    gap_df = family_df[
        family_df["k"] == 3
    ].copy()

    gap_df = gap_df.sort_values(
        "mean_selection_gap_vs_oracle",
        ascending=False,
    )

    print(
        gap_df[
            [
                "semantic_family",
                "rows",
                "mean_oracle_iou",
                "mean_selected_iou",
                "mean_selection_gap_vs_oracle",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("C3 COMPONENT-SELECTION ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Row-level results:\n{OUTPUT_FILE}"
    )

    print(
        f"Family results:\n{FAMILY_SUMMARY_FILE}"
    )

    print(
        f"K results:\n{K_SUMMARY_FILE}"
    )

    print(
        f"Summary JSON:\n{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()