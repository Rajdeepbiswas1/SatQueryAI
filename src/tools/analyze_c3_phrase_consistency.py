from __future__ import annotations

import json
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
    / "phrase_consistency.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "phrase_consistency_summary.json"
)


def main() -> None:

    print("=" * 70)
    print("C3 PHRASE CONSISTENCY ANALYSIS")
    print("=" * 70)

    if not ROW_FILE.exists():
        raise FileNotFoundError(
            f"Missing audit file:\n{ROW_FILE}"
        )

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Missing matrix file:\n{MATRIX_FILE}"
        )

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

    required_row_columns = {
        "phrase_normalized",
        "comparator",
        "top_clc",
        "top_fraction",
        "clc_class_count",
    }

    missing_row_columns = (
        required_row_columns
        - set(rows.columns)
    )

    if missing_row_columns:
        raise ValueError(
            "Missing audit columns:\n"
            f"{sorted(missing_row_columns)}"
        )

    required_matrix_columns = {
        "phrase_normalized",
        "clc_value",
        "bbox_fraction",
    }

    missing_matrix_columns = (
        required_matrix_columns
        - set(matrix.columns)
    )

    if missing_matrix_columns:
        raise ValueError(
            "Missing matrix columns:\n"
            f"{sorted(missing_matrix_columns)}"
        )

    # ========================================================
    # PHRASE-LEVEL ANALYSIS
    # ========================================================

    results = []

    grouped = rows.groupby(
        [
            "phrase_normalized",
            "comparator",
        ],
        sort=True,
    )

    for (
        phrase,
        comparator,
    ), group in grouped:

        group = group.copy()

        total_rows = len(group)

        # ----------------------------------------------------
        # Dominant top-CLC frequency
        # ----------------------------------------------------

        counts = (
            group["top_clc"]
            .value_counts()
        )

        dominant_clc = (
            int(counts.index[0])
            if len(counts) > 0
            else None
        )

        dominant_count = (
            int(counts.iloc[0])
            if len(counts) > 0
            else 0
        )

        dominant_share = (
            dominant_count
            / total_rows
            if total_rows > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Second-most-common CLC
        # ----------------------------------------------------

        if len(counts) >= 2:

            second_clc = int(
                counts.index[1]
            )

            second_count = int(
                counts.iloc[1]
            )

            second_share = (
                second_count
                / total_rows
            )

        else:

            second_clc = None
            second_count = 0
            second_share = 0.0

        # ----------------------------------------------------
        # Mean bbox fraction for the dominant CLC
        # ----------------------------------------------------

        dominant_matrix = matrix[
            matrix["phrase_normalized"]
            == phrase
        ].copy()

        dominant_matrix = dominant_matrix[
            dominant_matrix["clc_value"]
            == dominant_clc
        ]

        if len(dominant_matrix) > 0:

            dominant_fraction_mean = float(
                dominant_matrix[
                    "bbox_fraction"
                ].mean()
            )

            dominant_fraction_median = float(
                dominant_matrix[
                    "bbox_fraction"
                ].median()
            )

        else:

            dominant_fraction_mean = 0.0
            dominant_fraction_median = 0.0

        # ----------------------------------------------------
        # Overall bbox composition
        # ----------------------------------------------------

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

        median_class_count = float(
            group[
                "clc_class_count"
            ].median()
        )

        # ----------------------------------------------------
        # Comparator information
        # ----------------------------------------------------

        largest_count = int(
            (
                group["comparator"]
                == "largest"
            ).sum()
        )

        smallest_count = int(
            (
                group["comparator"]
                == "smallest"
            ).sum()
        )

        ordinary_count = int(
            (
                group["comparator"]
                == "ordinary"
            ).sum()
        )

        # ----------------------------------------------------
        # Consistency category
        #
        # These are descriptive bins for the audit only.
        # ----------------------------------------------------

        if dominant_share >= 0.80:

            consistency = "high"

        elif dominant_share >= 0.50:

            consistency = "medium"

        else:

            consistency = "low"

        results.append(
            {
                "phrase_normalized": phrase,
                "comparator": comparator,
                "rows": total_rows,
                "unique_top_clc": int(
                    counts.size
                ),
                "dominant_clc": dominant_clc,
                "dominant_count": dominant_count,
                "dominant_share": dominant_share,
                "second_clc": second_clc,
                "second_count": second_count,
                "second_share": second_share,
                "dominant_fraction_mean": (
                    dominant_fraction_mean
                ),
                "dominant_fraction_median": (
                    dominant_fraction_median
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
                "median_clc_class_count": (
                    median_class_count
                ),
                "largest_rows": largest_count,
                "smallest_rows": smallest_count,
                "ordinary_rows": ordinary_count,
                "consistency": consistency,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORTING
    # ========================================================

    result_df = result_df.sort_values(
        [
            "dominant_share",
            "mean_top_fraction",
            "rows",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE FULL PHRASE REPORT
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary = {
        "unique_phrases": int(
            result_df[
                "phrase_normalized"
            ].nunique()
        ),
        "high_consistency_phrases": int(
            (
                result_df["consistency"]
                == "high"
            ).sum()
        ),
        "medium_consistency_phrases": int(
            (
                result_df["consistency"]
                == "medium"
            ).sum()
        ),
        "low_consistency_phrases": int(
            (
                result_df["consistency"]
                == "low"
            ).sum()
        ),
        "mean_dominant_share": float(
            result_df[
                "dominant_share"
            ].mean()
        ),
        "median_dominant_share": float(
            result_df[
                "dominant_share"
            ].median()
        ),
        "mean_dominant_fraction": float(
            result_df[
                "dominant_fraction_mean"
            ].mean()
        ),
        "median_dominant_fraction": float(
            result_df[
                "dominant_fraction_mean"
            ].median()
        ),
        "mean_bbox_top_fraction": float(
            result_df[
                "mean_top_fraction"
            ].mean()
        ),
        "median_bbox_top_fraction": float(
            result_df[
                "median_top_fraction"
            ].median()
        ),
        "mean_clc_class_count": float(
            result_df[
                "mean_clc_class_count"
            ].mean()
        ),
        "median_clc_class_count": float(
            result_df[
                "median_clc_class_count"
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
    # PRINT OVERALL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PHRASE CONSISTENCY SUMMARY")
    print("=" * 70)

    print(
        f"Unique phrases: "
        f"{summary['unique_phrases']}"
    )

    print(
        f"High consistency: "
        f"{summary['high_consistency_phrases']}"
    )

    print(
        f"Medium consistency: "
        f"{summary['medium_consistency_phrases']}"
    )

    print(
        f"Low consistency: "
        f"{summary['low_consistency_phrases']}"
    )

    print(
        f"Mean dominant CLC share: "
        f"{summary['mean_dominant_share']:.6f}"
    )

    print(
        f"Median dominant CLC share: "
        f"{summary['median_dominant_share']:.6f}"
    )

    print(
        f"Mean dominant CLC bbox fraction: "
        f"{summary['mean_dominant_fraction']:.6f}"
    )

    print(
        f"Median dominant CLC bbox fraction: "
        f"{summary['median_dominant_fraction']:.6f}"
    )

    print(
        f"Mean bbox top-class fraction: "
        f"{summary['mean_bbox_top_fraction']:.6f}"
    )

    print(
        f"Mean CLC classes per bbox: "
        f"{summary['mean_clc_class_count']:.6f}"
    )

    # ========================================================
    # LOWEST-CONSISTENCY PHRASES
    # ========================================================

    print("\n" + "=" * 70)
    print("10 LEAST CLC-CONSISTENT PHRASES")
    print("=" * 70)

    low_columns = [
        "phrase_normalized",
        "comparator",
        "rows",
        "dominant_clc",
        "dominant_share",
        "mean_top_fraction",
        "mean_clc_class_count",
    ]

    print(
        result_df[
            low_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # HIGHEST-CONSISTENCY PHRASES
    # ========================================================

    print("\n" + "=" * 70)
    print("10 MOST CLC-CONSISTENT PHRASES")
    print("=" * 70)

    print(
        result_df[
            low_columns
        ]
        .sort_values(
            [
                "dominant_share",
                "mean_top_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Phrase report:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary:\n{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()