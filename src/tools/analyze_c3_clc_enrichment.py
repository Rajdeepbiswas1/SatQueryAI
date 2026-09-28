from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


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

MATRIX_FILE = (
    AUDIT_DIR
    / "phrase_clc_matrix.csv"
)

OUTPUT_FILE = (
    AUDIT_DIR
    / "clc_enrichment_by_family.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "clc_enrichment_summary.json"
)


# ============================================================
# SAME SEMANTIC-FAMILY NORMALIZATION USED IN C3 AUDIT
# ============================================================

def normalize_semantic_family(
    phrase: str,
) -> str:

    text = str(phrase).strip().lower()

    # Remove structural/reference-selection words only.
    # Do NOT remove actual semantic content.
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

    # Remove grammar left behind by structural words.
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
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 CLC ENRICHMENT BY SEMANTIC FAMILY")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not ROW_FILE.exists():
        raise FileNotFoundError(
            f"Missing row audit:\n{ROW_FILE}"
        )

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Missing CLC matrix:\n{MATRIX_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = pd.read_csv(
        ROW_FILE
    )

    matrix = pd.read_csv(
        MATRIX_FILE
    )

    print(
        f"Audit rows loaded: {len(rows)}"
    )

    print(
        f"Matrix rows loaded: {len(matrix)}"
    )

    # --------------------------------------------------------
    # Required row fields
    # --------------------------------------------------------

    required_rows = {
        "reference_phrase",
        "top_clc",
        "top_fraction",
    }

    missing_rows = (
        required_rows
        - set(rows.columns)
    )

    if missing_rows:
        raise ValueError(
            "Missing required row columns:\n"
            f"{sorted(missing_rows)}"
        )

    # --------------------------------------------------------
    # Required matrix fields
    # --------------------------------------------------------

    required_matrix = {
        "reference_phrase",
        "clc_value",
        "bbox_fraction",
    }

    missing_matrix = (
        required_matrix
        - set(matrix.columns)
    )

    if missing_matrix:
        raise ValueError(
            "Missing required matrix columns:\n"
            f"{sorted(missing_matrix)}"
        )

    # ========================================================
    # REBUILD SEMANTIC FAMILY
    # ========================================================

    rows["semantic_family"] = (
        rows["reference_phrase"]
        .map(
            normalize_semantic_family
        )
    )

    matrix["semantic_family"] = (
        matrix["reference_phrase"]
        .map(
            normalize_semantic_family
        )
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    missing_family_rows = int(
        rows["semantic_family"]
        .isna()
        .sum()
    )

    missing_family_matrix = int(
        matrix["semantic_family"]
        .isna()
        .sum()
    )

    if missing_family_rows:
        raise RuntimeError(
            "Missing semantic families in row audit."
        )

    if missing_family_matrix:
        raise RuntimeError(
            "Missing semantic families in CLC matrix."
        )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    full_phrase_count = int(
        rows["reference_phrase"]
        .nunique()
    )

    family_count = int(
        rows["semantic_family"]
        .nunique()
    )

    print(
        f"Unique full phrases: {full_phrase_count}"
    )

    print(
        f"Unique semantic families: {family_count}"
    )

    # ========================================================
    # GLOBAL CLC BASELINE
    # ========================================================

    print("\nCalculating global CLC baseline...")

    # Mean area fraction of each CLC class across
    # all audited bounding boxes.
    global_fraction = (
        matrix
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

    # Frequency with which each CLC class is the
    # dominant class of a bbox.
    global_top_clc_share = (
        rows[
            "top_clc"
        ]
        .value_counts(
            normalize=True
        )
    )

    print(
        f"CLC classes in global baseline: "
        f"{len(global_fraction)}"
    )

    # ========================================================
    # FAMILY × CLC ANALYSIS
    # ========================================================

    results = []

    family_groups = rows.groupby(
        "semantic_family",
        sort=True,
    )

    for family, family_rows in family_groups:

        family_matrix = matrix[
            matrix["semantic_family"]
            == family
        ]

        # ----------------------------------------------------
        # Mean bbox fraction per CLC
        # ----------------------------------------------------

        family_fraction = (
            family_matrix
            .groupby(
                "clc_value"
            )[
                "bbox_fraction"
            ]
            .mean()
        )

        # ----------------------------------------------------
        # Dominant CLC frequency
        # ----------------------------------------------------

        family_top_share = (
            family_rows[
                "top_clc"
            ]
            .value_counts(
                normalize=True
            )
        )

        # ----------------------------------------------------
        # CLC universe
        # ----------------------------------------------------

        clc_values = sorted(
            set(
                global_fraction.index
            )
            |
            set(
                family_fraction.index
            )
        )

        for clc_value in clc_values:

            baseline_fraction = float(
                global_fraction.get(
                    clc_value,
                    0.0,
                )
            )

            family_mean_fraction = float(
                family_fraction.get(
                    clc_value,
                    0.0,
                )
            )

            # ------------------------------------------------
            # Area-fraction lift
            # ------------------------------------------------

            if baseline_fraction > 0.0:

                fraction_lift = (
                    family_mean_fraction
                    / baseline_fraction
                )

            else:

                fraction_lift = np.nan

            # ------------------------------------------------
            # Top-CLC share lift
            # ------------------------------------------------

            baseline_top_share = float(
                global_top_clc_share.get(
                    clc_value,
                    0.0,
                )
            )

            family_top_clc_share = float(
                family_top_share.get(
                    clc_value,
                    0.0,
                )
            )

            if baseline_top_share > 0.0:

                top_share_lift = (
                    family_top_clc_share
                    / baseline_top_share
                )

            else:

                top_share_lift = np.nan

            results.append(
                {
                    "semantic_family": family,
                    "family_rows": int(
                        len(family_rows)
                    ),
                    "clc_value": int(
                        clc_value
                    ),
                    "global_mean_bbox_fraction": (
                        baseline_fraction
                    ),
                    "family_mean_bbox_fraction": (
                        family_mean_fraction
                    ),
                    "fraction_lift": (
                        fraction_lift
                    ),
                    "global_top_clc_share": (
                        baseline_top_share
                    ),
                    "family_top_clc_share": (
                        family_top_clc_share
                    ),
                    "top_share_lift": (
                        top_share_lift
                    ),
                }
            )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # FAMILY-LEVEL SUMMARY
    # ========================================================

    family_summaries = []

    for family, family_rows in family_groups:

        subset = result_df[
            result_df[
                "semantic_family"
            ]
            == family
        ].copy()

        # Only consider classes contributing at least
        # 1% mean bbox fraction within this family.
        meaningful = subset[
            subset[
                "family_mean_bbox_fraction"
            ]
            > 0.01
        ].copy()

        meaningful = meaningful.sort_values(
            [
                "fraction_lift",
                "family_mean_bbox_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

        # ----------------------------------------------------
        # Top enriched classes
        # ----------------------------------------------------

        top_enriched = []

        for _, item in (
            meaningful
            .head(10)
            .iterrows()
        ):

            top_enriched.append(
                {
                    "clc": int(
                        item["clc_value"]
                    ),
                    "family_mean_fraction": float(
                        item[
                            "family_mean_bbox_fraction"
                        ]
                    ),
                    "global_mean_fraction": float(
                        item[
                            "global_mean_bbox_fraction"
                        ]
                    ),
                    "fraction_lift": float(
                        item[
                            "fraction_lift"
                        ]
                    ),
                    "family_top_share": float(
                        item[
                            "family_top_clc_share"
                        ]
                    ),
                    "top_share_lift": float(
                        item[
                            "top_share_lift"
                        ]
                    ),
                }
            )

        # ----------------------------------------------------
        # Normalized family profile
        # ----------------------------------------------------

        profile_values = (
            meaningful[
                "family_mean_bbox_fraction"
            ]
            .clip(
                lower=0.0
            )
            .to_numpy()
        )

        profile_sum = (
            profile_values.sum()
        )

        if profile_sum > 0.0:

            profile = (
                profile_values
                / profile_sum
            )

            entropy = float(
                -np.sum(
                    profile
                    * np.log(
                        np.clip(
                            profile,
                            1e-12,
                            None,
                        )
                    )
                )
            )

        else:

            entropy = 0.0

        # ----------------------------------------------------
        # Strongest enriched CLC
        # ----------------------------------------------------

        if len(meaningful) > 0:

            top_item = (
                meaningful.iloc[0]
            )

            top_mean_clc = int(
                top_item[
                    "clc_value"
                ]
            )

            top_mean_fraction = float(
                top_item[
                    "family_mean_bbox_fraction"
                ]
            )

            top_fraction_lift = float(
                top_item[
                    "fraction_lift"
                ]
            )

        else:

            top_mean_clc = None
            top_mean_fraction = 0.0
            top_fraction_lift = 0.0

        family_summaries.append(
            {
                "semantic_family": family,
                "rows": int(
                    len(family_rows)
                ),
                "unique_top_clc": int(
                    family_rows[
                        "top_clc"
                    ].nunique()
                ),
                "top_mean_fraction_clc": (
                    top_mean_clc
                ),
                "top_mean_fraction": (
                    top_mean_fraction
                ),
                "top_fraction_lift": (
                    top_fraction_lift
                ),
                "family_profile_entropy": (
                    entropy
                ),
                "top_enriched_clc_json": (
                    json.dumps(
                        top_enriched,
                        ensure_ascii=False,
                    )
                ),
            }
        )

    family_df = pd.DataFrame(
        family_summaries
    )

    family_df = family_df.sort_values(
        [
            "top_fraction_lift",
            "rows",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    result_df = result_df.sort_values(
        [
            "semantic_family",
            "fraction_lift",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "full_phrase_count": full_phrase_count,
        "semantic_family_count": family_count,
        "global_clc_classes": int(
            len(global_fraction)
        ),
        "mean_top_fraction_lift": float(
            family_df[
                "top_fraction_lift"
            ].mean()
        ),
        "median_top_fraction_lift": float(
            family_df[
                "top_fraction_lift"
            ].median()
        ),
        "global_clc_baseline": [
            {
                "clc": int(clc_value),
                "mean_bbox_fraction": float(
                    fraction
                ),
            }
            for clc_value, fraction
            in global_fraction.items()
        ],
        "family_summary_file": str(
            OUTPUT_FILE
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "summary": summary,
                "family_summaries": (
                    family_df.to_dict(
                        orient="records"
                    )
                ),
            },
            f,
            indent=2,
        )

    # ========================================================
    # PRINT GLOBAL BASELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("GLOBAL CLC BASELINE")
    print("=" * 70)

    print(
        global_fraction
        .head(15)
        .to_string()
    )

    # ========================================================
    # PRINT FAMILY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP ENRICHED CLC BY SEMANTIC FAMILY")
    print("=" * 70)

    print(
        family_df[
            [
                "semantic_family",
                "rows",
                "top_mean_fraction_clc",
                "top_mean_fraction",
                "top_fraction_lift",
                "family_profile_entropy",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LOWEST ENRICHMENT
    # ========================================================

    print("\n" + "=" * 70)
    print("LOWEST TOP-CLASS ENRICHMENT")
    print("=" * 70)

    print(
        family_df[
            [
                "semantic_family",
                "rows",
                "top_mean_fraction_clc",
                "top_mean_fraction",
                "top_fraction_lift",
            ]
        ]
        .sort_values(
            "top_fraction_lift",
            ascending=True,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("CLC ENRICHMENT ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Detailed report:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary:\n{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()