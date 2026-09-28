from pathlib import Path
import json
import re

import pandas as pd
import torch


# ============================================================
# C3 QUERY-CONDITIONED SEMANTIC TARGET CONSTRUCTION
# ============================================================
#
# Purpose
# -------
# For every reference-grounding annotation row:
#
#     query/input
#          |
#          v
#     <ref>...</ref> phrase
#          |
#          v
#     semantic family
#          |
#          v
#     family index
#          |
#          v
#     select the corresponding dense family target
#
# Source family targets:
#
#     [N, 19, 14, 14]
#
# Row-level output:
#
#     [R, 14, 14]
#
# where R is the number of reference grounding rows.
#
# Example:
#
#     "largest patch of mixed forest"
#             |
#             v
#       mixed forest
#             |
#             v
#       family_index = 10
#             |
#             v
#       family_targets[patch, 10]
#
# IMPORTANT
# ---------
# This script performs NO learning.
#
# It does NOT:
#   - modify images
#   - modify reference maps
#   - modify CLC targets
#   - modify family targets
#   - modify checkpoints
#
# It creates new row-level target files only.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

TRAIN_FAMILY_TARGET_PATH = (
    TARGET_DIR
    / "train_family_targets.pt"
)

VAL_FAMILY_TARGET_PATH = (
    TARGET_DIR
    / "val_family_targets.pt"
)

TRAIN_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

VOCAB_PATH = (
    TARGET_DIR
    / "clc_vocabulary.json"
)

FAMILY_MAPPING_JSON = (
    TARGET_DIR
    / "clc_family_mapping.json"
)

TRAIN_OUTPUT_PATH = (
    TARGET_DIR
    / "train_query_conditioned_targets.pt"
)

VAL_OUTPUT_PATH = (
    TARGET_DIR
    / "val_query_conditioned_targets.pt"
)

MANIFEST_OUTPUT_PATH = (
    TARGET_DIR
    / "query_conditioned_target_manifest.json"
)

VERIFICATION_OUTPUT_PATH = (
    TARGET_DIR
    / "query_conditioned_target_verification.json"
)


# ============================================================
# EXPECTED FAMILY -> CLC MAPPING
# ============================================================
#
# This is deliberately repeated here so the script is
# independently auditable.
# ============================================================

FAMILY_TO_CLC = {

    "urban fabric": [
        111,
        112,
    ],

    "industrial or commercial units": [
        121,
    ],

    "arable land": [
        211,
        212,
        213,
    ],

    "permanent crops": [
        221,
        222,
        223,
        241,
    ],

    "pastures": [
        231,
    ],

    "complex cultivation patterns": [
        242,
    ],

    "land principally occupied by agriculture with significant areas of natural vegetation": [
        243,
    ],

    "agro-forestry areas": [
        244,
    ],

    "broad-leaved forest": [
        311,
    ],

    "coniferous forest": [
        312,
    ],

    "mixed forest": [
        313,
    ],

    "natural grassland and sparsely vegetated areas": [
        321,
        333,
    ],

    "moors, heathland and sclerophyllous vegetation": [
        322,
        323,
    ],

    "transitional woodland, shrub": [
        324,
    ],

    "beaches, dunes, sands": [
        331,
    ],

    "inland wetlands": [
        411,
        412,
    ],

    "coastal wetlands": [
        421,
        422,
    ],

    "inland waters": [
        511,
        512,
    ],

    "marine waters": [
        521,
        522,
        523,
    ],
}


FAMILY_NAMES = list(
    FAMILY_TO_CLC.keys()
)

NUM_FAMILIES = len(
    FAMILY_NAMES
)


# ============================================================
# REFERENCE PHRASE PATTERNS
# ============================================================

REF_PATTERN = re.compile(
    r"<ref>(.*?)</ref>",
    flags=re.IGNORECASE | re.DOTALL,
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    text = str(value).lower()

    text = text.replace(
        "\u2013",
        "-",
    )

    text = text.replace(
        "\u2014",
        "-",
    )

    text = text.replace(
        "\u2018",
        "'",
    )

    text = text.replace(
        "\u2019",
        "'",
    )

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# REFERENCE PHRASE EXTRACTION
# ============================================================

def extract_reference_phrase(
    value
):
    if value is None:
        return None

    text = str(
        value
    ).strip()

    match = REF_PATTERN.search(
        text
    )

    if match is None:
        return None

    phrase = (
        match.group(1)
        .strip()
    )

    if not phrase:
        return None

    return phrase


# ============================================================
# SEMANTIC FAMILY RULES
# ============================================================

FAMILY_RULES = [

    (
        "urban fabric",
        [
            "urban fabric",
        ],
    ),

    (
        "industrial or commercial units",
        [
            "industrial or commercial units",
        ],
    ),

    (
        "arable land",
        [
            "arable land",
        ],
    ),

    (
        "permanent crops",
        [
            "permanent crops",
            "annual crops associated with permanent crops",
        ],
    ),

    (
        "pastures",
        [
            "pastures",
        ],
    ),

    (
        "complex cultivation patterns",
        [
            "complex cultivation patterns",
        ],
    ),

    (
        "land principally occupied by agriculture with significant areas of natural vegetation",
        [
            "land principally occupied by agriculture with significant areas of natural vegetation",
        ],
    ),

    (
        "agro-forestry areas",
        [
            "agro-forestry areas",
        ],
    ),

    (
        "broad-leaved forest",
        [
            "broad-leaved forest",
        ],
    ),

    (
        "coniferous forest",
        [
            "coniferous forest",
        ],
    ),

    (
        "mixed forest",
        [
            "mixed forest",
        ],
    ),

    (
        "natural grassland and sparsely vegetated areas",
        [
            "natural grassland and sparsely vegetated areas",
            "natural grasslands or sparsely vegetated areas",
            "natural grassland",
            "sparsely vegetated areas",
        ],
    ),

    (
        "moors, heathland and sclerophyllous vegetation",
        [
            "moors, heathland and sclerophyllous vegetation",
            "moors, heathland, or sclerophyllous vegetation",
            "moors and heathland",
            "sclerophyllous vegetation",
        ],
    ),

    (
        "transitional woodland, shrub",
        [
            "transitional woodland, shrub",
            "transitional woodlands or shrubs",
            "transitional woodland/shrub",
        ],
    ),

    (
        "beaches, dunes, sands",
        [
            "beaches, dunes, sands",
            "beaches, dunes, or sands",
        ],
    ),

    (
        "inland wetlands",
        [
            "inland wetlands",
            "inland marshes",
            "peat bogs",
            "peatbogs",
        ],
    ),

    (
        "coastal wetlands",
        [
            "coastal wetlands",
            "salt marshes",
            "salines",
            "intertidal flats",
        ],
    ),

    (
        "inland waters",
        [
            "inland waters",
            "water courses",
            "water bodies",
        ],
    ),

    (
        "marine waters",
        [
            "marine waters",
            "coastal lagoons",
            "estuaries",
            "sea and ocean",
        ],
    ),
]


# ============================================================
# FAMILY INFERENCE
# ============================================================

def infer_family(
    phrase
):
    if phrase is None:
        return None

    text = normalize_text(
        phrase
    )

    matches = []

    for family, patterns in FAMILY_RULES:

        for pattern in patterns:

            normalized_pattern = (
                normalize_text(
                    pattern
                )
            )

            if normalized_pattern in text:

                matches.append(
                    family
                )

                break

    unique_matches = list(
        dict.fromkeys(
            matches
        )
    )

    if len(unique_matches) == 0:
        return None

    if len(unique_matches) > 1:

        raise RuntimeError(
            "Ambiguous semantic-family assignment.\n"
            f"Phrase: {phrase}\n"
            f"Matches: {unique_matches}"
        )

    return unique_matches[0]


# ============================================================
# LOAD FAMILY TARGET FILE
# ============================================================

def load_family_target_file(
    path,
    split_name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{split_name} family target file not found:\n"
            f"{path}"
        )

    print(
        f"Loading {split_name} family targets:"
    )

    print(
        f"  {path}"
    )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            f"{split_name}: expected dictionary."
        )

    required_keys = [
        "targets",
        "patch_ids",
        "family_names",
        "family_to_clc",
    ]

    missing = [
        key
        for key
        in required_keys
        if key not in data
    ]

    if missing:

        raise RuntimeError(
            f"{split_name}: missing keys: {missing}"
        )

    targets = data[
        "targets"
    ].float()

    patch_ids = [
        str(value)
        for value
        in data[
            "patch_ids"
        ]
    ]

    family_names = list(
        data[
            "family_names"
        ]
    )

    if targets.ndim != 4:

        raise RuntimeError(
            f"{split_name}: expected "
            f"[N,19,H,W], got "
            f"{tuple(targets.shape)}"
        )

    if targets.shape[1] != 19:

        raise RuntimeError(
            f"{split_name}: expected 19 family "
            f"channels, got {targets.shape[1]}"
        )

    if len(patch_ids) != targets.shape[0]:

        raise RuntimeError(
            f"{split_name}: patch ID count "
            "does not match target count."
        )

    if family_names != FAMILY_NAMES:

        raise RuntimeError(
            f"{split_name}: family order differs "
            "from the frozen expected order."
        )

    # Ensure patch IDs are unique.
    if len(
        set(patch_ids)
    ) != len(patch_ids):

        raise RuntimeError(
            f"{split_name}: duplicate patch IDs "
            "inside family target file."
        )

    print(
        f"  Shape: {tuple(targets.shape)}"
    )

    print(
        f"  Patch IDs: {len(patch_ids):,}"
    )

    return (
        targets,
        patch_ids,
    )


# ============================================================
# BUILD PATCH LOOKUP
# ============================================================

def build_patch_lookup(
    patch_ids,
):
    return {
        patch_id: index
        for index, patch_id
        in enumerate(
            patch_ids
        )
    }


# ============================================================
# LOAD SPLIT ANNOTATIONS
# ============================================================

def load_annotations(
    path,
    split_name,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{split_name} parquet not found:\n"
            f"{path}"
        )

    print()
    print(
        f"Loading {split_name} annotations:"
    )

    print(
        f"  {path}"
    )

    df = pd.read_parquet(
        path
    )

    print(
        f"  Rows: {len(df):,}"
    )

    required_columns = [
        "ID",
        "patch_id",
        "input",
        "output",
        "type",
        "category",
        "original_row_index",
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{split_name}: missing required "
            f"columns: {missing}"
        )

    # The split must be reference bounding-box data.
    if not (
        df[
            "type"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "bounding box"
        )
        .all()
    ):

        raise RuntimeError(
            f"{split_name}: contains "
            "non-bounding-box rows."
        )

    if not (
        df[
            "category"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "reference"
        )
        .all()
    ):

        raise RuntimeError(
            f"{split_name}: contains "
            "non-reference rows."
        )

    # original_row_index is the stable mapping to the
    # original training annotation table.
    df[
        "original_row_index"
    ] = df[
        "original_row_index"
    ].astype(int)

    return df.reset_index(
        drop=True
    )


# ============================================================
# BUILD ROW-LEVEL TARGETS
# ============================================================

def build_query_conditioned_targets(
    df,
    family_targets,
    family_patch_ids,
    split_name,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"BUILDING {split_name.upper()} "
        "QUERY-CONDITIONED TARGETS"
    )

    print(
        "=" * 70
    )

    patch_lookup = build_patch_lookup(
        family_patch_ids
    )

    patch_indices = []
    family_indices = []
    phrases = []
    semantic_families = []
    row_indices = []
    annotation_ids = []

    missing_patches = []
    missing_phrases = []
    unmatched_phrases = []

    # --------------------------------------------------------
    # Process every annotation.
    # --------------------------------------------------------

    for row_position in range(
        len(df)
    ):

        row = df.iloc[
            row_position
        ]

        patch_id = str(
            row[
                "patch_id"
            ]
        )

        if patch_id not in patch_lookup:

            missing_patches.append(
                patch_id
            )

            continue

        phrase = (
            extract_reference_phrase(
                row[
                    "input"
                ]
            )
        )

        if phrase is None:

            missing_phrases.append(
                row_position
            )

            continue

        family = infer_family(
            phrase
        )

        if family is None:

            unmatched_phrases.append(
                phrase
            )

            continue

        family_index = FAMILY_NAMES.index(
            family
        )

        patch_index = patch_lookup[
            patch_id
        ]

        patch_indices.append(
            patch_index
        )

        family_indices.append(
            family_index
        )

        phrases.append(
            phrase
        )

        semantic_families.append(
            family
        )

        row_indices.append(
            int(
                row[
                    "original_row_index"
                ]
            )
        )

        annotation_ids.append(
            str(
                row[
                    "ID"
                ]
            )
        )

    # --------------------------------------------------------
    # Hard validation.
    # --------------------------------------------------------

    if missing_patches:

        unique_missing = sorted(
            set(
                missing_patches
            )
        )

        raise RuntimeError(
            f"{split_name}: "
            f"{len(missing_patches):,} rows refer to "
            "patch IDs absent from family targets.\n"
            f"Examples: {unique_missing[:10]}"
        )

    if missing_phrases:

        raise RuntimeError(
            f"{split_name}: "
            f"{len(missing_phrases):,} rows do not contain "
            "<ref>...</ref>."
        )

    if unmatched_phrases:

        unique_unmatched = sorted(
            set(
                normalize_text(
                    phrase
                )
                for phrase
                in unmatched_phrases
            )
        )

        print(
            "Unmatched phrases:"
        )

        for phrase in unique_unmatched:

            print(
                f"  {phrase}"
            )

        raise RuntimeError(
            f"{split_name}: "
            f"{len(unmatched_phrases):,} phrases "
            "could not be assigned to a family."
        )

    if len(
        patch_indices
    ) != len(df):

        raise RuntimeError(
            f"{split_name}: processed "
            f"{len(patch_indices):,} rows but expected "
            f"{len(df):,}."
        )

    # --------------------------------------------------------
    # Convert metadata to tensors.
    # --------------------------------------------------------

    patch_indices_tensor = torch.tensor(
        patch_indices,
        dtype=torch.long,
    )

    family_indices_tensor = torch.tensor(
        family_indices,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Vectorized row-level selection.
    #
    # family_targets:
    #
    #     [N, 19, H, W]
    #
    # First select the patch dimension:
    #
    #     [R, 19, H, W]
    #
    # Then select the family dimension:
    #
    #     [R, H, W]
    # --------------------------------------------------------

    selected_patch_targets = family_targets[
        patch_indices_tensor
    ]

    row_targets = (
        selected_patch_targets[
            torch.arange(
                len(df),
                dtype=torch.long,
            ),
            family_indices_tensor,
        ]
        .contiguous()
        .float()
    )

    expected_height = (
        family_targets.shape[-2]
    )

    expected_width = (
        family_targets.shape[-1]
    )

    if row_targets.shape != (
        len(df),
        expected_height,
        expected_width,
    ):

        raise RuntimeError(
            f"{split_name}: unexpected row target "
            f"shape: {tuple(row_targets.shape)}"
        )

    # --------------------------------------------------------
    # Metadata consistency checks.
    # --------------------------------------------------------

    if len(row_indices) != len(df):

        raise RuntimeError(
            f"{split_name}: original row-index "
            "count mismatch."
        )

    if len(semantic_families) != len(df):

        raise RuntimeError(
            f"{split_name}: semantic family count mismatch."
        )

    if len(phrases) != len(df):

        raise RuntimeError(
            f"{split_name}: phrase count mismatch."
        )

    # --------------------------------------------------------
    # Family counts.
    # --------------------------------------------------------

    family_counts = {}

    for family in FAMILY_NAMES:

        family_counts[
            family
        ] = sum(
            value == family
            for value
            in semantic_families
        )

    print()
    print(
        "Query-conditioned family counts:"
    )

    for family in FAMILY_NAMES:

        print(
            f"  {family:<72}"
            f"{family_counts[family]:>7}"
        )

    # --------------------------------------------------------
    # Patch usage.
    # --------------------------------------------------------

    unique_patch_count = len(
        set(
            family_patch_ids[
                index
            ]
            for index
            in patch_indices
        )
    )

    print()

    print(
        f"Unique patches used: "
        f"{unique_patch_count:,}"
    )

    print(
        f"Rows generated: "
        f"{len(row_targets):,}"
    )

    print(
        f"Target shape: "
        f"{tuple(row_targets.shape)}"
    )

    return {
        "targets":
            row_targets,

        "patch_ids":
            [
                str(
                    df.iloc[
                        i
                    ][
                        "patch_id"
                    ]
                )
                for i in range(
                    len(df)
                )
            ],

        "original_row_index":
            row_indices,

        "family_indices":
            family_indices_tensor,

        "family_names_per_row":
            semantic_families,

        "reference_phrases":
            phrases,

        "annotation_ids":
            annotation_ids,

        "source_patch_indices":
            patch_indices_tensor,

        "family_counts":
            family_counts,

        "unique_patch_count":
            unique_patch_count,
    }


# ============================================================
# VERIFY ROW TARGET AGAINST SOURCE FAMILY TARGET
# ============================================================

def verify_query_targets(
    query_data,
    family_targets,
    family_patch_ids,
    df,
    split_name,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"VERIFYING {split_name.upper()} "
        "QUERY-CONDITIONED TARGETS"
    )

    print(
        "=" * 70
    )

    row_targets = query_data[
        "targets"
    ]

    family_indices = query_data[
        "family_indices"
    ]

    source_patch_indices = query_data[
        "source_patch_indices"
    ]

    # --------------------------------------------------------
    # Exact reconstruction.
    # --------------------------------------------------------

    reconstructed = (
        family_targets[
            source_patch_indices
        ]
        [
            torch.arange(
                len(df),
                dtype=torch.long,
            ),
            family_indices,
        ]
        .float()
    )

    abs_diff = (
        row_targets
        - reconstructed
    ).abs()

    max_exact_error = float(
        abs_diff.max()
    )

    mean_exact_error = float(
        abs_diff.mean()
    )

    print(
        f"Max row-target reconstruction error: "
        f"{max_exact_error:.12e}"
    )

    print(
        f"Mean row-target reconstruction error: "
        f"{mean_exact_error:.12e}"
    )

    if max_exact_error > 0.0:

        # The selection operation should be exact because it
        # references the same tensor values. A tolerance is
        # nevertheless allowed for safe future changes.
        if max_exact_error > 1e-7:

            raise RuntimeError(
                f"{split_name}: row target reconstruction "
                "failed."
            )

    # --------------------------------------------------------
    # Range.
    # --------------------------------------------------------

    minimum = float(
        row_targets.min()
    )

    maximum = float(
        row_targets.max()
    )

    print(
        f"Row target range: "
        f"[{minimum:.10f}, {maximum:.10f}]"
    )

    if minimum < -1e-6:

        raise RuntimeError(
            f"{split_name}: target values below 0."
        )

    if maximum > 1.0 + 1e-6:

        raise RuntimeError(
            f"{split_name}: target values above 1."
        )

    # --------------------------------------------------------
    # Every row's heatmap should be the complete spatial
    # semantic distribution for exactly one family.
    # --------------------------------------------------------

    row_mass = (
        row_targets
        .double()
        .sum(
            dim=(1, 2)
        )
    )

    if not torch.isfinite(
        row_mass
    ).all():

        raise RuntimeError(
            f"{split_name}: row target masses "
            "contain NaN/Inf."
        )

    # --------------------------------------------------------
    # Verify every row patch exists.
    # --------------------------------------------------------

    patch_lookup = build_patch_lookup(
        family_patch_ids
    )

    for row_position in range(
        len(df)
    ):

        patch_id = str(
            df.iloc[
                row_position
            ][
                "patch_id"
            ]
        )

        if patch_id not in patch_lookup:

            raise RuntimeError(
                f"{split_name}: patch {patch_id} "
                "missing during verification."
            )

        expected_patch_index = (
            patch_lookup[
                patch_id
            ]
        )

        actual_patch_index = int(
            source_patch_indices[
                row_position
            ]
        )

        if (
            expected_patch_index
            != actual_patch_index
        ):

            raise RuntimeError(
                f"{split_name}: patch-index mismatch "
                f"for row {row_position}."
            )

    # --------------------------------------------------------
    # Family verification.
    # --------------------------------------------------------

    for row_position in range(
        len(df)
    ):

        phrase = (
            query_data[
                "reference_phrases"
            ][
                row_position
            ]
        )

        family = infer_family(
            phrase
        )

        recorded_family = (
            query_data[
                "family_names_per_row"
            ][
                row_position
            ]
        )

        if family != recorded_family:

            raise RuntimeError(
                f"{split_name}: family mismatch "
                f"at row {row_position}."
            )

        expected_index = FAMILY_NAMES.index(
            family
        )

        actual_index = int(
            family_indices[
                row_position
            ]
        )

        if expected_index != actual_index:

            raise RuntimeError(
                f"{split_name}: family-index mismatch "
                f"at row {row_position}."
            )

    # --------------------------------------------------------
    # Patch overlap statistics.
    # --------------------------------------------------------

    unique_patch_ids = len(
        set(
            query_data[
                "patch_ids"
            ]
        )
    )

    print(
        f"Rows verified: "
        f"{len(df):,}"
    )

    print(
        f"Unique patches: "
        f"{unique_patch_ids:,}"
    )

    print(
        f"Max row reconstruction error: "
        f"{max_exact_error:.12e}"
    )

    print(
        "Query-conditioned target verification: PASS"
    )

    return {
        "split":
            split_name,

        "rows":
            int(
                len(df)
            ),

        "unique_patches":
            int(
                unique_patch_ids
            ),

        "max_reconstruction_error":
            max_exact_error,

        "mean_reconstruction_error":
            mean_exact_error,

        "target_min":
            minimum,

        "target_max":
            maximum,

        "mean_target_mass":
            float(
                row_mass.mean()
            ),

        "median_target_mass":
            float(
                row_mass.median()
            ),

        "max_target_mass":
            float(
                row_mass.max()
            ),

        "min_target_mass":
            float(
                row_mass.min()
            ),
    }


# ============================================================
# SAVE QUERY TARGET FILE
# ============================================================

def save_query_targets(
    path,
    query_data,
    split_name,
):

    payload = {
        "targets":
            query_data[
                "targets"
            ],

        "patch_ids":
            query_data[
                "patch_ids"
            ],

        "original_row_index":
            query_data[
                "original_row_index"
            ],

        "family_indices":
            query_data[
                "family_indices"
            ],

        "family_names_per_row":
            query_data[
                "family_names_per_row"
            ],

        "reference_phrases":
            query_data[
                "reference_phrases"
            ],

        "annotation_ids":
            query_data[
                "annotation_ids"
            ],

        "source_patch_indices":
            query_data[
                "source_patch_indices"
            ],

        "family_names":
            FAMILY_NAMES,

        "family_to_clc":
            FAMILY_TO_CLC,

        "num_families":
            NUM_FAMILIES,

        "split":
            split_name,

        "target_size":
            [
                int(
                    query_data[
                        "targets"
                    ].shape[-2]
                ),
                int(
                    query_data[
                        "targets"
                    ].shape[-1]
                ),
            ],

        "dtype":
            "float32",

        "source":
            "C3 verified 19-family dense semantic targets",
    }

    torch.save(
        payload,
        path,
    )

    print()
    print(
        f"Saved {split_name} query-conditioned targets:"
    )

    print(
        f"  {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "C3 QUERY-CONDITIONED SEMANTIC TARGET CONSTRUCTION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Mode:"
    )

    print(
        "  LOCAL"
    )

    print(
        "  GPU OFF"
    )

    print(
        "  KAGGLE OFF"
    )

    print()

    # --------------------------------------------------------
    # Required files.
    # --------------------------------------------------------

    required_paths = [

        TRAIN_FAMILY_TARGET_PATH,

        VAL_FAMILY_TARGET_PATH,

        TRAIN_PARQUET,

        VAL_PARQUET,

        VOCAB_PATH,

        FAMILY_MAPPING_JSON,
    ]

    print(
        "CHECKING REQUIRED FILES"
    )

    print(
        "-" * 70
    )

    for path in required_paths:

        if path.exists():

            print(
                f"FOUND: {path}"
            )

        else:

            raise FileNotFoundError(
                f"MISSING: {path}"
            )

    # --------------------------------------------------------
    # Frozen vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "LOADING FROZEN VOCABULARY"
    )

    print(
        "-" * 70
    )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        vocab = json.load(
            f
        )

    frozen_clc = sorted(
        int(value)
        for value
        in vocab[
            "clc_vocabulary"
        ]
    )

    expected_clc = sorted(
        clc
        for family in FAMILY_NAMES
        for clc
        in FAMILY_TO_CLC[
            family
        ]
    )

    if frozen_clc != expected_clc:

        raise RuntimeError(
            "Frozen CLC vocabulary does not match "
            "the audited 32-class grouping."
        )

    print(
        f"Frozen CLC classes: "
        f"{len(frozen_clc)}"
    )

    print(
        f"Semantic families: "
        f"{NUM_FAMILIES}"
    )

    print(
        "Vocabulary check: PASS"
    )

    # --------------------------------------------------------
    # Family mapping JSON sanity check.
    # --------------------------------------------------------

    print()
    print(
        "CHECKING FAMILY MAPPING REPORT"
    )

    print(
        "-" * 70
    )

    with open(
        FAMILY_MAPPING_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        mapping_report = json.load(
            f
        )

    mapping_family_to_clc = {
        family:
            [
                int(value)
                for value
                in values
            ]
        for family, values
        in mapping_report[
            "family_to_clc"
        ].items()
    }

    if mapping_family_to_clc != FAMILY_TO_CLC:

        raise RuntimeError(
            "Saved clc_family_mapping.json does not "
            "match the audited mapping."
        )

    print(
        "Saved family mapping: PASS"
    )

    # --------------------------------------------------------
    # Load family targets.
    # --------------------------------------------------------

    (
        train_family_targets,
        train_family_patch_ids,
    ) = load_family_target_file(
        TRAIN_FAMILY_TARGET_PATH,
        "train",
    )

    (
        val_family_targets,
        val_family_patch_ids,
    ) = load_family_target_file(
        VAL_FAMILY_TARGET_PATH,
        "validation",
    )

    # --------------------------------------------------------
    # Load annotations.
    # --------------------------------------------------------

    train_df = load_annotations(
        TRAIN_PARQUET,
        "train",
    )

    val_df = load_annotations(
        VAL_PARQUET,
        "validation",
    )

    # --------------------------------------------------------
    # Verify patch sets.
    # --------------------------------------------------------

    train_patch_set = set(
        train_family_patch_ids
    )

    val_patch_set = set(
        val_family_patch_ids
    )

    overlap = (
        train_patch_set
        & val_patch_set
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TRAIN / VALIDATION PATCH ISOLATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Train family-target patches: "
        f"{len(train_patch_set):,}"
    )

    print(
        f"Validation family-target patches: "
        f"{len(val_patch_set):,}"
    )

    print(
        f"Overlap: "
        f"{len(overlap):,}"
    )

    if overlap:

        raise RuntimeError(
            "Train/validation patch overlap detected."
        )

    print(
        "Patch isolation: PASS"
    )

    # --------------------------------------------------------
    # Check every annotation patch belongs to its split's
    # family target set.
    # --------------------------------------------------------

    missing_train_annotation_patches = sorted(
        set(
            train_df[
                "patch_id"
            ]
            .astype(str)
        )
        - train_patch_set
    )

    missing_val_annotation_patches = sorted(
        set(
            val_df[
                "patch_id"
            ]
            .astype(str)
        )
        - val_patch_set
    )

    if missing_train_annotation_patches:

        raise RuntimeError(
            "Training annotation patches missing "
            "from family targets:\n"
            + "\n".join(
                missing_train_annotation_patches[
                    :10
                ]
            )
        )

    if missing_val_annotation_patches:

        raise RuntimeError(
            "Validation annotation patches missing "
            "from family targets:\n"
            + "\n".join(
                missing_val_annotation_patches[
                    :10
                ]
            )
        )

    print(
        "Annotation patch coverage: PASS"
    )

    # --------------------------------------------------------
    # Build train row targets.
    # --------------------------------------------------------

    train_query_data = (
        build_query_conditioned_targets(
            train_df,
            train_family_targets,
            train_family_patch_ids,
            "train",
        )
    )

    # --------------------------------------------------------
    # Build validation row targets.
    # --------------------------------------------------------

    val_query_data = (
        build_query_conditioned_targets(
            val_df,
            val_family_targets,
            val_family_patch_ids,
            "validation",
        )
    )

    # --------------------------------------------------------
    # Verify.
    # --------------------------------------------------------

    train_verification = (
        verify_query_targets(
            train_query_data,
            train_family_targets,
            train_family_patch_ids,
            train_df,
            "train",
        )
    )

    val_verification = (
        verify_query_targets(
            val_query_data,
            val_family_targets,
            val_family_patch_ids,
            val_df,
            "validation",
        )
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SAVING QUERY-CONDITIONED TARGETS"
    )

    print(
        "=" * 70
    )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_query_targets(
        TRAIN_OUTPUT_PATH,
        train_query_data,
        "train",
    )

    save_query_targets(
        VAL_OUTPUT_PATH,
        val_query_data,
        "validation",
    )

    # --------------------------------------------------------
    # Manifest.
    # --------------------------------------------------------

    manifest = {

        "project":
            "SatQueryAI",

        "stage":
            "C3 query-conditioned semantic target construction",

        "source_representation":
            {
                "classes":
                    19,

                "spatial_shape":
                    [
                        int(
                            train_family_targets.shape[-2]
                        ),
                        int(
                            train_family_targets.shape[-1]
                        ),
                    ],

                "train_patches":
                    int(
                        train_family_targets.shape[0]
                    ),

                "validation_patches":
                    int(
                        val_family_targets.shape[0]
                    ),
            },

        "row_representation":
            {
                "train_rows":
                    int(
                        train_query_data[
                            "targets"
                        ].shape[0]
                    ),

                "validation_rows":
                    int(
                        val_query_data[
                            "targets"
                        ].shape[0]
                    ),

                "target_shape_train":
                    list(
                        train_query_data[
                            "targets"
                        ].shape
                    ),

                "target_shape_validation":
                    list(
                        val_query_data[
                            "targets"
                        ].shape
                    ),

                "target_spatial_size":
                    [
                        int(
                            train_query_data[
                                "targets"
                            ].shape[-2]
                        ),
                        int(
                            train_query_data[
                                "targets"
                            ].shape[-1]
                        ),
                    ],
            },

        "family_names":
            FAMILY_NAMES,

        "family_to_clc":
            FAMILY_TO_CLC,

        "train_family_counts":
            train_query_data[
                "family_counts"
            ],

        "validation_family_counts":
            val_query_data[
                "family_counts"
            ],

        "train_unique_query_patches":
            train_query_data[
                "unique_patch_count"
            ],

        "validation_unique_query_patches":
            val_query_data[
                "unique_patch_count"
            ],

        "patch_overlap":
            0,

        "outputs":
            {
                "train":
                    str(
                        TRAIN_OUTPUT_PATH
                    ),

                "validation":
                    str(
                        VAL_OUTPUT_PATH
                    ),
            },

        "verification":
            {
                "train":
                    train_verification,

                "validation":
                    val_verification,
            },

        "construction":
            (
                "For each reference annotation row, "
                "extract <ref>...</ref>, infer its semantic "
                "family, find the row patch in the corresponding "
                "19-family target tensor, and select the target "
                "channel for that family."
            ),
    }

    with open(
        MANIFEST_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Verification JSON.
    # --------------------------------------------------------

    verification = {
        "train":
            train_verification,

        "validation":
            val_verification,

        "overall":
            {
                "train_passed":
                    True,

                "validation_passed":
                    True,

                "train_validation_patch_overlap":
                    0,
            },
    }

    with open(
        VERIFICATION_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            verification,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "C3 QUERY-CONDITIONED TARGET BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Train rows: "
        f"{len(train_query_data['targets']):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_query_data['targets']):,}"
    )

    print()

    print(
        f"Train target shape: "
        f"{tuple(train_query_data['targets'].shape)}"
    )

    print(
        f"Validation target shape: "
        f"{tuple(val_query_data['targets'].shape)}"
    )

    print()

    print(
        "Train verification: PASS"
    )

    print(
        "Validation verification: PASS"
    )

    print()

    print(
        "Created:"
    )

    print(
        f"  {TRAIN_OUTPUT_PATH}"
    )

    print(
        f"  {VAL_OUTPUT_PATH}"
    )

    print(
        f"  {MANIFEST_OUTPUT_PATH}"
    )

    print(
        f"  {VERIFICATION_OUTPUT_PATH}"
    )

    print()

    print(
        "No existing C3 targets, images, maps, "
        "or checkpoints were modified."
    )


if __name__ == "__main__":
    main()