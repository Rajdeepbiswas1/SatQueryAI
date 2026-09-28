from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


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

MATRIX_FILE = (
    AUDIT_DIR
    / "phrase_clc_matrix.csv"
)

OUTPUT_FILE = (
    AUDIT_DIR
    / "semantic_family_analysis.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "semantic_family_analysis.json"
)


# ============================================================
# SEMANTIC-FAMILY NORMALIZATION
# ============================================================

def normalize_semantic_family(
    phrase: str,
) -> str:

    text = str(phrase).strip().lower()

    # Remove only structural/reference-selection words.
    # We deliberately do NOT replace semantic words.
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

    # Remove a leading "of" introduced by removing
    # words such as "patch of".
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

    text = text.strip()

    return text


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 SEMANTIC FAMILY / CLC ANALYSIS")
    print("=" * 70)

    if not ROW_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{ROW_FILE}"
        )

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{MATRIX_FILE}"
        )

    rows = pd.read_csv(
        ROW_FILE
    )

    matrix = pd.read_csv(
        MATRIX_FILE
    )

    print(
        f"Audit rows: {len(rows)}"
    )

    print(
        f"Matrix rows: {len(matrix)}"
    )

    # --------------------------------------------------------
    # Create semantic family
    # --------------------------------------------------------

    rows["semantic_family"] = (
        rows["reference_phrase"]
        .map(normalize_semantic_family)
    )

    matrix["semantic_family"] = (
        matrix["reference_phrase"]
        .map(normalize_semantic_family)
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    missing_family = int(
        rows["semantic_family"]
        .isna()
        .sum()
    )

    if missing_family:
        raise RuntimeError(
            "Some rows have no semantic family."
        )

    family_count = rows[
        "semantic_family"
    ].nunique()

    phrase_count = rows[
        "phrase_normalized"
    ].nunique()

    print(
        f"Unique full phrases: {phrase_count}"
    )

    print(
        f"Unique semantic families: {family_count}"
    )

    # --------------------------------------------------------
    # Analyze families
    # --------------------------------------------------------

    results = []

    grouped = rows.groupby(
        "semantic_family",
        sort=True,
    )

    for family, group in grouped:

        # --------------------------------------------
        # Full phrase variants
        # --------------------------------------------

        variants = sorted(
            group[
                "phrase_normalized"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        # --------------------------------------------
        # Dominant CLC across rows
        # --------------------------------------------

        top_counts = (
            group["top_clc"]
            .value_counts()
        )

        dominant_clc = (
            int(top_counts.index[0])
            if len(top_counts)
            else None
        )

        dominant_count = (
            int(top_counts.iloc[0])
            if len(top_counts)
            else 0
        )

        dominant_share = (
            dominant_count
            / len(group)
        )

        # --------------------------------------------
        # Number of distinct CLC values
        # --------------------------------------------

        unique_top_clc = int(
            group[
                "top_clc"
            ].nunique()
        )

        # --------------------------------------------
        # Most frequent top CLCs
        # --------------------------------------------

        top_clc_distribution = []

        for clc_value, count in (
            top_counts
            .head(10)
            .items()
        ):

            top_clc_distribution.append(
                {
                    "clc": int(clc_value),
                    "rows": int(count),
                    "share": float(
                        count
                        / len(group)
                    ),
                }
            )

        # --------------------------------------------
        # Matrix for this semantic family
        # --------------------------------------------

        family_matrix = matrix[
            matrix["semantic_family"]
            == family
        ].copy()

        # Mean fraction by CLC.
        clc_means = (
            family_matrix
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

        mean_fraction_distribution = []

        for clc_value, fraction in (
            clc_means
            .head(10)
            .items()
        ):

            mean_fraction_distribution.append(
                {
                    "clc": int(clc_value),
                    "mean_bbox_fraction": float(
                        fraction
                    ),
                }
            )

        # --------------------------------------------
        # Comparator distribution
        # --------------------------------------------

        comparator_counts = (
            group[
                "comparator"
            ]
            .value_counts()
            .to_dict()
        )

        # --------------------------------------------
        # Quality measures
        # --------------------------------------------

        mean_top_fraction = float(
            group[
                "top_fraction"
            ].mean()
        )

        median_top_fraction = float(
            group[
                "top_fraction"
            ].median()
        )

        mean_class_count = float(
            group[
                "clc_class_count"
            ].mean()
        )

        # --------------------------------------------
        # Family classification
        # --------------------------------------------

        if dominant_share >= 0.80:
            family_consistency = "high"
        elif dominant_share >= 0.50:
            family_consistency = "medium"
        else:
            family_consistency = "low"

        # --------------------------------------------
        # Output
        # --------------------------------------------

        results.append(
            {
                "semantic_family": family,
                "rows": int(len(group)),
                "unique_patches": int(
                    group[
                        "patch_id"
                    ].nunique()
                ),
                "full_phrase_variants": int(
                    len(variants)
                ),
                "variants_json": json.dumps(
                    variants,
                    ensure_ascii=False,
                ),
                "unique_top_clc": unique_top_clc,
                "dominant_clc": dominant_clc,
                "dominant_clc_share": float(
                    dominant_share
                ),
                "top_clc_distribution_json": (
                    json.dumps(
                        top_clc_distribution
                    )
                ),
                "mean_clc_fraction_distribution_json": (
                    json.dumps(
                        mean_fraction_distribution
                    )
                ),
                "mean_top_fraction": (
                    mean_top_fraction
                ),
                "median_top_fraction": (
                    median_top_fraction
                ),
                "mean_clc_class_count": (
                    mean_class_count
                ),
                "largest_rows": int(
                    comparator_counts.get(
                        "largest",
                        0,
                    )
                ),
                "smallest_rows": int(
                    comparator_counts.get(
                        "smallest",
                        0,
                    )
                ),
                "ordinary_rows": int(
                    comparator_counts.get(
                        "ordinary",
                        0,
                    )
                ),
                "family_consistency": (
                    family_consistency
                ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sort by consistency
    # --------------------------------------------------------

    result_df = result_df.sort_values(
        [
            "dominant_clc_share",
            "rows",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    summary = {
        "full_phrase_count": int(
            phrase_count
        ),
        "semantic_family_count": int(
            family_count
        ),
        "high_consistency_families": int(
            (
                result_df[
                    "family_consistency"
                ]
                == "high"
            ).sum()
        ),
        "medium_consistency_families": int(
            (
                result_df[
                    "family_consistency"
                ]
                == "medium"
            ).sum()
        ),
        "low_consistency_families": int(
            (
                result_df[
                    "family_consistency"
                ]
                == "low"
            ).sum()
        ),
        "mean_dominant_clc_share": float(
            result_df[
                "dominant_clc_share"
            ].mean()
        ),
        "median_dominant_clc_share": float(
            result_df[
                "dominant_clc_share"
            ].median()
        ),
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
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("SEMANTIC FAMILY SUMMARY")
    print("=" * 70)

    print(
        f"Full phrases: "
        f"{summary['full_phrase_count']}"
    )

    print(
        f"Semantic families: "
        f"{summary['semantic_family_count']}"
    )

    print(
        f"High consistency families: "
        f"{summary['high_consistency_families']}"
    )

    print(
        f"Medium consistency families: "
        f"{summary['medium_consistency_families']}"
    )

    print(
        f"Low consistency families: "
        f"{summary['low_consistency_families']}"
    )

    print(
        f"Mean dominant CLC share: "
        f"{summary['mean_dominant_clc_share']:.6f}"
    )

    print(
        f"Median dominant CLC share: "
        f"{summary['median_dominant_clc_share']:.6f}"
    )

    # ========================================================
    # LEAST CONSISTENT FAMILIES
    # ========================================================

    print("\n" + "=" * 70)
    print("15 LEAST CONSISTENT SEMANTIC FAMILIES")
    print("=" * 70)

    columns = [
        "semantic_family",
        "rows",
        "full_phrase_variants",
        "dominant_clc",
        "dominant_clc_share",
        "unique_top_clc",
        "mean_top_fraction",
        "mean_clc_class_count",
    ]

    print(
        result_df[
            columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # MOST CONSISTENT FAMILIES
    # ========================================================

    print("\n" + "=" * 70)
    print("15 MOST CONSISTENT SEMANTIC FAMILIES")
    print("=" * 70)

    print(
        result_df[
            columns
        ]
        .sort_values(
            [
                "dominant_clc_share",
                "rows",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("SEMANTIC FAMILY ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Report:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary:\n{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()