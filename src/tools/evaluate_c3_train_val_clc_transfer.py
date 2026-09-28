from __future__ import annotations

import ast
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
    / "train_val_clc_transfer.csv"
)

FAMILY_OUTPUT_FILE = (
    AUDIT_DIR
    / "train_val_clc_transfer_by_family.csv"
)

K_OUTPUT_FILE = (
    AUDIT_DIR
    / "train_val_clc_transfer_by_k.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR
    / "train_val_clc_transfer_summary.json"
)

TOP_K_VALUES = [1, 2, 3, 4, 5]


# ============================================================
# EXACT SAME SEMANTIC-FAMILY NORMALIZATION USED PREVIOUSLY
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
# PARSE CLC FRACTION JSON
# ============================================================

def parse_fraction_json(
    text: str,
) -> dict[int, float]:

    parsed = ast.literal_eval(
        str(text)
    )

    return {
        int(k): float(v)
        for k, v in parsed.items()
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not ROW_FILE.exists():
        raise FileNotFoundError(
            f"Missing row audit:\n{ROW_FILE}"
        )

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Missing CLC matrix:\n{MATRIX_FILE}"
        )

    rows = pd.read_csv(
        ROW_FILE
    )

    matrix = pd.read_csv(
        MATRIX_FILE
    )

    return rows, matrix


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 TRAIN -> VALIDATION CLC TRANSFER")
    print("=" * 70)

    rows, matrix = load_data()

    print(
        f"Rows: {len(rows)}"
    )

    print(
        f"Matrix rows: {len(matrix)}"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_rows = {
        "original_row_index",
        "reference_phrase",
        "audit_split",
        "top_clc",
        "top_fraction",
        "clc_fractions_json",
        "patch_id",
    }

    missing_rows = (
        required_rows
        - set(rows.columns)
    )

    if missing_rows:
        raise ValueError(
            "Missing row columns:\n"
            f"{sorted(missing_rows)}"
        )

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
            "Missing matrix columns:\n"
            f"{sorted(missing_matrix)}"
        )

    # ========================================================
    # REBUILD SEMANTIC FAMILY
    # ========================================================

    print(
        "\nReconstructing semantic families..."
    )

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
    # CHECK ROW IDS
    # ========================================================

    if rows[
        "original_row_index"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate original_row_index detected."
        )

    # ========================================================
    # PARSE CLC DISTRIBUTIONS
    # ========================================================

    print(
        "\nParsing CLC distributions..."
    )

    rows["clc_distribution"] = (
        rows[
            "clc_fractions_json"
        ]
        .map(
            parse_fraction_json
        )
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    rows["audit_split"] = (
        rows["audit_split"]
        .astype(str)
        .str.lower()
    )

    train = rows[
        rows["audit_split"]
        == "train"
    ].copy()

    val = rows[
        rows["audit_split"]
        == "val"
    ].copy()

    print(
        f"Train reference rows: {len(train)}"
    )

    print(
        f"Validation reference rows: {len(val)}"
    )

    if len(train) == 0:
        raise RuntimeError(
            "Training split is empty."
        )

    if len(val) == 0:
        raise RuntimeError(
            "Validation split is empty."
        )

    # ========================================================
    # TRAIN / VALIDATION PATCH OVERLAP
    # ========================================================

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
        f"Train unique patches: "
        f"{len(train_patches)}"
    )

    print(
        f"Validation unique patches: "
        f"{len(val_patches)}"
    )

    print(
        f"Train/validation patch overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train/validation patch overlap detected."
        )

    # ========================================================
    # LEARN FAMILY -> CLC PROFILE FROM TRAIN ONLY
    # ========================================================

    print(
        "\nLearning semantic-family CLC profiles "
        "from TRAIN ONLY..."
    )

    family_profiles: dict[
        str,
        dict,
    ] = {}

    train_family_groups = train.groupby(
        "semantic_family",
        sort=True,
    )

    for family, family_df in (
        train_family_groups
    ):

        sums: dict[int, float] = {}
        counts: dict[int, int] = {}

        for distribution in (
            family_df["clc_distribution"]
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

        means = {}

        for clc_value in sums:

            means[clc_value] = (
                sums[clc_value]
                / counts[clc_value]
            )

        ranked = sorted(
            means.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        family_profiles[str(family)] = {
            "row_count": int(
                len(family_df)
            ),
            "ranked_clc": [
                {
                    "clc": int(
                        clc_value
                    ),
                    "mean_fraction": float(
                        fraction
                    ),
                }
                for clc_value, fraction
                in ranked
            ],
        }

    print(
        f"Families learned: "
        f"{len(family_profiles)}"
    )

    # ========================================================
    # VERIFY ALL VALIDATION FAMILIES EXIST IN TRAIN
    # ========================================================

    val_families = set(
        val["semantic_family"]
        .astype(str)
        .unique()
    )

    missing_val_families = (
        val_families
        - set(
            family_profiles.keys()
        )
    )

    print(
        f"Validation families: "
        f"{len(val_families)}"
    )

    print(
        f"Validation families missing "
        f"from train: "
        f"{len(missing_val_families)}"
    )

    if missing_val_families:

        print(
            "\nMissing families:"
        )

        for family in sorted(
            missing_val_families
        ):
            print(
                family
            )

        raise RuntimeError(
            "Some validation semantic families "
            "are absent from training."
        )

    # ========================================================
    # VALIDATION TRANSFER
    # ========================================================

    print(
        "\nEvaluating validation transfer..."
    )

    results = []

    for _, row in val.iterrows():

        family = str(
            row["semantic_family"]
        )

        ranked = family_profiles[
            family
        ]["ranked_clc"]

        actual_distribution = (
            row["clc_distribution"]
        )

        actual_total = float(
            sum(
                actual_distribution.values()
            )
        )

        if not np.isclose(
            actual_total,
            1.0,
            atol=1e-9,
        ):

            raise RuntimeError(
                "CLC fractions do not sum "
                "to 1.\n"
                f"Row: "
                f"{row['original_row_index']}\n"
                f"Sum: {actual_total}"
            )

        for k in TOP_K_VALUES:

            candidate_codes = {
                int(item["clc"])
                for item in ranked[:k]
            }

            coverage = 0.0

            for clc_value, fraction in (
                actual_distribution.items()
            ):

                if int(clc_value) in candidate_codes:

                    coverage += float(
                        fraction
                    )

            uncovered = (
                1.0
                - coverage
            )

            results.append(
                {
                    "original_row_index": int(
                        row[
                            "original_row_index"
                        ]
                    ),
                    "patch_id": str(
                        row["patch_id"]
                    ),
                    "reference_phrase": str(
                        row[
                            "reference_phrase"
                        ]
                    ),
                    "semantic_family": family,
                    "k": int(k),
                    "candidate_clc_codes": json.dumps(
                        sorted(
                            candidate_codes
                        )
                    ),
                    "bbox_candidate_coverage": float(
                        coverage
                    ),
                    "bbox_uncovered_fraction": float(
                        uncovered
                    ),
                    "ground_truth_top_clc": int(
                        row[
                            "top_clc"
                        ]
                    ),
                    "ground_truth_top_fraction": float(
                        row[
                            "top_fraction"
                        ]
                    ),
                }
            )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # GLOBAL TOP-K SUMMARY
    # ========================================================

    global_summaries = []

    for k in TOP_K_VALUES:

        subset = result_df[
            result_df["k"]
            == k
        ]

        coverage = subset[
            "bbox_candidate_coverage"
        ]

        global_summaries.append(
            {
                "k": int(k),
                "rows": int(len(subset)),
                "mean_bbox_candidate_coverage": float(
                    coverage.mean()
                ),
                "median_bbox_candidate_coverage": float(
                    coverage.median()
                ),
                "p10_bbox_candidate_coverage": float(
                    coverage.quantile(
                        0.10
                    )
                ),
                "p25_bbox_candidate_coverage": float(
                    coverage.quantile(
                        0.25
                    )
                ),
                "p75_bbox_candidate_coverage": float(
                    coverage.quantile(
                        0.75
                    )
                ),
                "p90_bbox_candidate_coverage": float(
                    coverage.quantile(
                        0.90
                    )
                ),
                "fraction_ge_0_50": float(
                    (
                        coverage
                        >= 0.50
                    ).mean()
                ),
                "fraction_ge_0_70": float(
                    (
                        coverage
                        >= 0.70
                    ).mean()
                ),
                "fraction_ge_0_80": float(
                    (
                        coverage
                        >= 0.80
                    ).mean()
                ),
                "fraction_ge_0_90": float(
                    (
                        coverage
                        >= 0.90
                    ).mean()
                ),
            }
        )

    summary_df = pd.DataFrame(
        global_summaries
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

        coverage = subset[
            "bbox_candidate_coverage"
        ]

        family_summaries.append(
            {
                "semantic_family": family,
                "k": int(k),
                "rows": int(len(subset)),
                "mean_coverage": float(
                    coverage.mean()
                ),
                "median_coverage": float(
                    coverage.median()
                ),
                "p25_coverage": float(
                    coverage.quantile(
                        0.25
                    )
                ),
                "p75_coverage": float(
                    coverage.quantile(
                        0.75
                    )
                ),
                "fraction_ge_0_80": float(
                    (
                        coverage
                        >= 0.80
                    ).mean()
                ),
                "fraction_ge_0_90": float(
                    (
                        coverage
                        >= 0.90
                    ).mean()
                ),
            }
        )

    family_df = pd.DataFrame(
        family_summaries
    )

    # ========================================================
    # GAIN VS K=1
    # ========================================================

    k_lookup = (
        summary_df
        .set_index("k")
    )

    improvement = {}

    k1_mean = float(
        k_lookup.loc[
            1,
            "mean_bbox_candidate_coverage",
        ]
    )

    k1_median = float(
        k_lookup.loc[
            1,
            "median_bbox_candidate_coverage",
        ]
    )

    for k in TOP_K_VALUES:

        improvement[str(k)] = {
            "mean_gain_vs_k1": float(
                k_lookup.loc[
                    k,
                    "mean_bbox_candidate_coverage",
                ]
                - k1_mean
            ),
            "median_gain_vs_k1": float(
                k_lookup.loc[
                    k,
                    "median_bbox_candidate_coverage",
                ]
                - k1_median
            ),
        }

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    family_df.to_csv(
        FAMILY_OUTPUT_FILE,
        index=False,
    )

    summary_df.to_csv(
        K_OUTPUT_FILE,
        index=False,
    )

    summary_json = {
        "train_rows": int(
            len(train)
        ),
        "validation_rows": int(
            len(val)
        ),
        "train_unique_patches": int(
            len(train_patches)
        ),
        "validation_unique_patches": int(
            len(val_patches)
        ),
        "train_validation_patch_overlap": int(
            len(overlap)
        ),
        "full_phrase_count": int(
            full_phrase_count
        ),
        "semantic_family_count": int(
            family_count
        ),
        "training_families": int(
            len(family_profiles)
        ),
        "validation_families": int(
            len(val_families)
        ),
        "missing_validation_families": sorted(
            missing_val_families
        ),
        "top_k_values": TOP_K_VALUES,
        "global_summary": global_summaries,
        "improvement_vs_k1": improvement,
        "row_output": str(
            OUTPUT_FILE
        ),
        "family_output": str(
            FAMILY_OUTPUT_FILE
        ),
        "k_output": str(
            K_OUTPUT_FILE
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_json,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT GLOBAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("GLOBAL TOP-K COVERAGE")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT FAMILY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FAMILY RESULTS — K=1")
    print("=" * 70)

    print(
        family_df[
            family_df["k"] == 1
        ]
        .sort_values(
            "mean_coverage",
            ascending=True,
        )
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FAMILY RESULTS — K=3")
    print("=" * 70)

    print(
        family_df[
            family_df["k"] == 3
        ]
        .sort_values(
            "mean_coverage",
            ascending=True,
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "TRAIN -> VALIDATION CLC TRANSFER COMPLETE"
    )
    print("=" * 70)

    print(
        f"Row-level output:\n{OUTPUT_FILE}"
    )

    print(
        f"Family output:\n{FAMILY_OUTPUT_FILE}"
    )

    print(
        f"K summary:\n{K_OUTPUT_FILE}"
    )

    print(
        f"JSON summary:\n{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()