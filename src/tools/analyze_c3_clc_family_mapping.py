from pathlib import Path
import json
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# C3 CLC -> SEMANTIC FAMILY MAPPING AUDIT
# ============================================================
#
# Purpose:
#   Audit the relationship between:
#
#       BigEarthNet V2 32-class CLC vocabulary
#
#   and:
#
#       BigEarthNet 19-class semantic families
#
# Important:
#   The grounding parquet does NOT contain a dedicated
#   "phrase" column.
#
#   The actual reference phrase is stored inside the
#   "input" column as:
#
#       <ref>...</ref>
#
#   This is consistent with the verified grounding dataset
#   implementation already used in the project.
#
# This script:
#
#   1. Loads the frozen 32-class CLC vocabulary.
#   2. Validates the official 32 -> 19 grouping.
#   3. Loads reference_train.parquet.
#   4. Extracts the semantic phrase from input.
#   5. Assigns each phrase to one of 19 families.
#   6. Audits unmatched / ambiguous phrases.
#   7. Counts family occurrences.
#   8. Checks validation-family availability.
#   9. Saves a CSV and JSON audit report.
#
# No original imagery, reference maps, vocabulary, or
# semantic target tensors are modified.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
    / "clc_vocabulary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "clc_family_mapping.csv"
)

JSON_OUTPUT = (
    OUTPUT_DIR
    / "clc_family_mapping.json"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_NUM_CLC = 32
EXPECTED_NUM_FAMILIES = 19


# ============================================================
# 19 SEMANTIC FAMILIES
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


EXPECTED_FAMILIES = [
    family
    for family, _patterns in FAMILY_RULES
]


# ============================================================
# OFFICIAL 32 CLC -> 19 SEMANTIC FAMILY GROUPING
# ============================================================

EXPECTED_CLC_GROUPS = {

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


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for semantic-family matching.
    """

    if value is None:
        return ""

    text = str(value).lower()

    # Unicode punctuation normalization.
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

    # Normalize underscores.
    text = text.replace(
        "_",
        " ",
    )

    # Normalize slash spacing.
    text = text.replace(
        " / ",
        "/",
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# EXTRACT <ref>...</ref> FROM INPUT
# ============================================================

REF_PATTERN = re.compile(
    r"<ref>(.*?)</ref>",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_reference_phrase(value):
    """
    Extract the semantic reference phrase from the
    grounding input string.

    Example:

        <ref>largest patch of transitional woodlands or shrubs</ref>

    becomes:

        largest patch of transitional woodlands or shrubs
    """

    if value is None:
        return None

    text = str(value).strip()

    match = REF_PATTERN.search(
        text
    )

    if match is None:
        return None

    phrase = match.group(1).strip()

    if not phrase:
        return None

    return phrase


# ============================================================
# INFER SEMANTIC FAMILY
# ============================================================

def infer_family(phrase):
    """
    Match a reference phrase to exactly one semantic family.

    Returns:
        family name
        None for no match

    Raises:
        RuntimeError for multiple-family ambiguity.
    """

    if phrase is None:
        return None

    text = normalize_text(
        phrase
    )

    matches = []

    for family, patterns in FAMILY_RULES:

        family_match = False

        for pattern in patterns:

            normalized_pattern = normalize_text(
                pattern
            )

            if normalized_pattern in text:

                family_match = True
                break

        if family_match:

            matches.append(
                family
            )

    # No family matched.
    if len(matches) == 0:
        return None

    # More than one family matched.
    unique_matches = list(
        dict.fromkeys(matches)
    )

    if len(unique_matches) > 1:

        raise RuntimeError(
            "Ambiguous semantic-family match.\n"
            f"Phrase: {phrase}\n"
            f"Matched families: {unique_matches}"
        )

    return unique_matches[0]


# ============================================================
# VALIDATE FROZEN VOCABULARY
# ============================================================

print(
    "=" * 70
)

print(
    "C3 CLC -> SEMANTIC FAMILY MAPPING AUDIT"
)

print(
    "=" * 70
)

print()


if not VOCAB_PATH.exists():

    raise FileNotFoundError(
        "Frozen CLC vocabulary not found:\n"
        f"{VOCAB_PATH}"
    )


with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:

    vocab = json.load(
        f
    )


if not isinstance(
    vocab,
    dict,
):

    raise RuntimeError(
        "Frozen vocabulary JSON must contain an object."
    )


if "clc_vocabulary" not in vocab:

    raise RuntimeError(
        "Frozen vocabulary JSON does not contain "
        "'clc_vocabulary'."
    )


clc_values = sorted(
    int(value)
    for value in vocab[
        "clc_vocabulary"
    ]
)


print(
    f"Frozen CLC vocabulary: "
    f"{len(clc_values)} classes"
)

print(
    clc_values
)

print()


# ============================================================
# CHECK EXPECTED FAMILY COUNT
# ============================================================

print(
    "EXPECTED FAMILY STRUCTURE CHECK"
)

print(
    "-" * 70
)

print(
    f"Expected families: "
    f"{len(EXPECTED_FAMILIES)}"
)


if len(EXPECTED_FAMILIES) != EXPECTED_NUM_FAMILIES:

    raise RuntimeError(
        "Incorrect number of semantic families: "
        f"{len(EXPECTED_FAMILIES)}"
    )


# ============================================================
# CHECK GROUPED CLC COUNT
# ============================================================

expected_flat_clc = sorted(
    clc
    for family in EXPECTED_FAMILIES
    for clc in EXPECTED_CLC_GROUPS[
        family
    ]
)


print(
    f"Expected grouped CLC values: "
    f"{len(expected_flat_clc)}"
)


if len(clc_values) != EXPECTED_NUM_CLC:

    raise RuntimeError(
        f"Expected {EXPECTED_NUM_CLC} "
        f"CLC classes, found {len(clc_values)}."
    )


# ============================================================
# EXACT VOCABULARY MATCH
# ============================================================

missing_clc = sorted(
    set(expected_flat_clc)
    - set(clc_values)
)

unexpected_clc = sorted(
    set(clc_values)
    - set(expected_flat_clc)
)


if missing_clc or unexpected_clc:

    print()

    print(
        "Frozen vocabulary:"
    )

    print(
        clc_values
    )

    print()

    print(
        "Expected grouped CLC values:"
    )

    print(
        expected_flat_clc
    )

    print()

    print(
        f"Missing expected CLC values: "
        f"{missing_clc}"
    )

    print(
        f"Unexpected vocabulary values: "
        f"{unexpected_clc}"
    )

    raise RuntimeError(
        "Frozen vocabulary does not match the "
        "explicit 32-class grouping."
    )


# ============================================================
# CHECK DUPLICATE CLC ASSIGNMENTS
# ============================================================

clc_occurrences = defaultdict(
    int
)

for family in EXPECTED_FAMILIES:

    for clc in EXPECTED_CLC_GROUPS[
        family
    ]:

        clc_occurrences[
            clc
        ] += 1


duplicate_assignments = sorted(
    clc
    for clc, count
    in clc_occurrences.items()
    if count != 1
)


if duplicate_assignments:

    raise RuntimeError(
        "CLC values assigned to multiple families: "
        f"{duplicate_assignments}"
    )


print(
    "32-class vocabulary <-> 19-family grouping: PASS"
)

print()


# ============================================================
# BUILD REVERSE MAPPING
# ============================================================

clc_to_family = {}

for family in EXPECTED_FAMILIES:

    for clc in EXPECTED_CLC_GROUPS[
        family
    ]:

        if clc in clc_to_family:

            raise RuntimeError(
                f"CLC {clc} assigned more than once."
            )

        clc_to_family[
            clc
        ] = family


# ============================================================
# LOAD TRAINING PARQUET
# ============================================================

print(
    "LOADING TRAINING REFERENCE ANNOTATIONS"
)

print(
    "-" * 70
)


if not TRAIN_PARQUET.exists():

    raise FileNotFoundError(
        "Training reference parquet not found:\n"
        f"{TRAIN_PARQUET}"
    )


train_df = pd.read_parquet(
    TRAIN_PARQUET
)


print(
    f"Rows loaded: "
    f"{len(train_df):,}"
)

print()


print(
    "Columns:"
)

for column in train_df.columns:

    print(
        f"  {column}"
    )

print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ID",
    "patch_id",
    "input",
    "output",
    "type",
    "category",
    "original_row_index",
]


missing_columns = [
    column
    for column in required_columns
    if column not in train_df.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# ============================================================
# VERIFY THIS IS THE REFERENCE GROUNDING SPLIT
# ============================================================

type_values = sorted(
    train_df[
        "type"
    ]
    .astype(str)
    .str.lower()
    .unique()
    .tolist()
)

category_values = sorted(
    train_df[
        "category"
    ]
    .astype(str)
    .str.lower()
    .unique()
    .tolist()
)


print(
    f"Unique type values: "
    f"{type_values}"
)

print(
    f"Unique category values: "
    f"{category_values}"
)

print()


if not all(
    str(value).lower()
    == "bounding box"
    for value in train_df[
        "type"
    ]
):

    raise RuntimeError(
        "reference_train.parquet contains "
        "non-bounding-box rows."
    )


if not all(
    str(value).lower()
    == "reference"
    for value in train_df[
        "category"
    ]
):

    raise RuntimeError(
        "reference_train.parquet contains "
        "non-reference rows."
    )


# ============================================================
# EXTRACT REFERENCE PHRASES
# ============================================================

print(
    "EXTRACTING <ref>...</ref> PHRASES FROM INPUT"
)

print(
    "-" * 70
)


train_df[
    "reference_phrase"
] = train_df[
    "input"
].apply(
    extract_reference_phrase
)


missing_reference_phrase = train_df[
    train_df[
        "reference_phrase"
    ].isna()
]


print(
    f"Rows with extracted reference phrase: "
    f"{len(train_df) - len(missing_reference_phrase):,}"
)

print(
    f"Rows without <ref>...</ref>: "
    f"{len(missing_reference_phrase):,}"
)

print()


if len(missing_reference_phrase) > 0:

    print(
        "Examples without a reference tag:"
    )

    print(
        "-" * 70
    )

    for value in (
        missing_reference_phrase[
            "input"
        ]
        .astype(str)
        .head(20)
        .tolist()
    ):

        print(
            f"  {value}"
        )

    print()

    raise RuntimeError(
        "Some reference grounding rows do not contain "
        "a <ref>...</ref> phrase in the input column."
    )


# ============================================================
# PARSE SEMANTIC FAMILIES
# ============================================================

print(
    "PARSING SEMANTIC FAMILIES"
)

print(
    "-" * 70
)


train_df[
    "semantic_family"
] = train_df[
    "reference_phrase"
].apply(
    infer_family
)


unmatched_df = train_df[
    train_df[
        "semantic_family"
    ].isna()
].copy()


matched_count = (
    len(train_df)
    - len(unmatched_df)
)


print(
    f"Matched rows:   "
    f"{matched_count:,}"
)

print(
    f"Unmatched rows: "
    f"{len(unmatched_df):,}"
)

print()


# ============================================================
# HANDLE UNMATCHED PHRASES
# ============================================================

if len(unmatched_df) > 0:

    print(
        "UNMATCHED REFERENCE PHRASES"
    )

    print(
        "-" * 70
    )

    unmatched_phrases = sorted(
        set(
            normalize_text(
                phrase
            )
            for phrase
            in unmatched_df[
                "reference_phrase"
            ]
            .tolist()
        )
    )

    for phrase in unmatched_phrases:

        print(
            f"  {phrase}"
        )

    print()

    raise RuntimeError(
        "Some reference phrases could not be "
        "assigned to a semantic family."
    )


# ============================================================
# OBSERVED FAMILY SET
# ============================================================

observed_families = sorted(
    set(
        train_df[
            "semantic_family"
        ]
        .tolist()
    )
)


print(
    f"Unique semantic families observed: "
    f"{len(observed_families)}"
)

print()


# ============================================================
# CHECK EXPECTED 19 FAMILIES
# ============================================================

missing_train_families = sorted(
    set(EXPECTED_FAMILIES)
    - set(observed_families)
)


if missing_train_families:

    print(
        "Families missing from training reference data:"
    )

    for family in missing_train_families:

        print(
            f"  {family}"
        )

    print()

    raise RuntimeError(
        "Not all expected semantic families "
        "are represented in training reference data."
    )


# ============================================================
# FAMILY COUNTS
# ============================================================

train_family_counts = (
    train_df[
        "semantic_family"
    ]
    .value_counts()
    .to_dict()
)


print(
    "TRAIN REFERENCE FAMILY COUNTS"
)

print(
    "-" * 70
)

for family in EXPECTED_FAMILIES:

    count = int(
        train_family_counts.get(
            family,
            0,
        )
    )

    print(
        f"{family:<70} "
        f"{count:>8}"
    )

print()


# ============================================================
# PRINT CLC -> FAMILY STRUCTURE
# ============================================================

print(
    "=" * 70
)

print(
    "CLC <-> SEMANTIC FAMILY STRUCTURE"
)

print(
    "=" * 70
)

print()


for family in EXPECTED_FAMILIES:

    family_clcs = (
        EXPECTED_CLC_GROUPS[
            family
        ]
    )

    family_rows = train_df[
        train_df[
            "semantic_family"
        ]
        == family
    ]

    unique_phrases = (
        family_rows[
            "reference_phrase"
        ]
        .astype(str)
        .nunique()
    )

    print(
        family
    )

    print(
        f"  CLC codes:      {family_clcs}"
    )

    print(
        f"  CLC count:      {len(family_clcs)}"
    )

    print(
        f"  Reference rows: {len(family_rows):,}"
    )

    print(
        f"  Unique phrases: {unique_phrases:,}"
    )

    print()


# ============================================================
# BUILD OUTPUT MAPPING TABLE
# ============================================================

mapping_rows = []


for family in EXPECTED_FAMILIES:

    family_clcs = (
        EXPECTED_CLC_GROUPS[
            family
        ]
    )

    family_rows = train_df[
        train_df[
            "semantic_family"
        ]
        == family
    ]

    unique_phrases = (
        family_rows[
            "reference_phrase"
        ]
        .astype(str)
        .nunique()
    )

    for clc in family_clcs:

        mapping_rows.append(
            {
                "semantic_family":
                    family,

                "clc":
                    clc,

                "clc_count_in_family":
                    len(
                        family_clcs
                    ),

                "family_type":
                    (
                        "single_clc"
                        if len(family_clcs) == 1
                        else "multi_clc"
                    ),

                "train_reference_rows":
                    len(
                        family_rows
                    ),

                "train_unique_reference_phrases":
                    unique_phrases,
            }
        )


mapping_df = pd.DataFrame(
    mapping_rows
)


# ============================================================
# SINGLE VS MULTI CLC FAMILY SUMMARY
# ============================================================

single_clc_families = []

multi_clc_families = []


for family in EXPECTED_FAMILIES:

    count = len(
        EXPECTED_CLC_GROUPS[
            family
        ]
    )

    if count == 1:

        single_clc_families.append(
            family
        )

    else:

        multi_clc_families.append(
            family
        )


print(
    "=" * 70
)

print(
    "STRUCTURAL AUDIT"
)

print(
    "=" * 70
)

print()


print(
    f"Single-CLC families: "
    f"{len(single_clc_families)}"
)

for family in single_clc_families:

    print(
        f"  {family}: "
        f"{EXPECTED_CLC_GROUPS[family]}"
    )


print()


print(
    f"Multi-CLC families: "
    f"{len(multi_clc_families)}"
)

for family in multi_clc_families:

    print(
        f"  {family}: "
        f"{EXPECTED_CLC_GROUPS[family]}"
    )


print()


# ============================================================
# EXACT CLC COVERAGE CHECK
# ============================================================

print(
    "CLC COVERAGE AUDIT"
)

print(
    "-" * 70
)


mapped_clc_values = sorted(
    clc_to_family.keys()
)


print(
    f"Frozen vocabulary classes: "
    f"{len(clc_values)}"
)

print(
    f"Mapped vocabulary classes: "
    f"{len(mapped_clc_values)}"
)


if mapped_clc_values != clc_values:

    raise RuntimeError(
        "CLC -> family mapping does not cover "
        "the frozen vocabulary exactly."
    )


print(
    "CLC coverage: PASS"
)

print()


# ============================================================
# VALIDATION FAMILY AVAILABILITY AUDIT
# ============================================================
#
# IMPORTANT:
#   The mapping itself is NOT learned from validation.
#
#   Validation is checked only to determine which semantic
#   families happen to occur there.
# ============================================================

print(
    "CHECKING VALIDATION FAMILY AVAILABILITY"
)

print(
    "-" * 70
)


validation_family_counts = {}


if not VAL_PARQUET.exists():

    print(
        "Validation parquet not found."
    )

    print(
        "Skipping validation family audit."
    )

else:

    val_df = pd.read_parquet(
        VAL_PARQUET
    )

    print(
        f"Validation rows loaded: "
        f"{len(val_df):,}"
    )

    print()

    required_val_columns = [
        "input",
        "type",
        "category",
        "original_row_index",
    ]

    missing_val_columns = [
        column
        for column in required_val_columns
        if column not in val_df.columns
    ]

    if missing_val_columns:

        raise RuntimeError(
            "Validation parquet missing required columns: "
            + ", ".join(
                missing_val_columns
            )
        )

    # The split should already be only reference bounding boxes.
    val_type_ok = (
        val_df[
            "type"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "bounding box"
        )
    )

    val_category_ok = (
        val_df[
            "category"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "reference"
        )
    )

    if not val_type_ok.all():

        raise RuntimeError(
            "Validation parquet contains "
            "non-bounding-box rows."
        )

    if not val_category_ok.all():

        raise RuntimeError(
            "Validation parquet contains "
            "non-reference rows."
        )

    val_df[
        "reference_phrase"
    ] = val_df[
        "input"
    ].apply(
        extract_reference_phrase
    )

    missing_val_phrase = val_df[
        val_df[
            "reference_phrase"
        ].isna()
    ]

    print(
        f"Validation rows with <ref> phrase: "
        f"{len(val_df) - len(missing_val_phrase):,}"
    )

    print(
        f"Validation rows without <ref>: "
        f"{len(missing_val_phrase):,}"
    )

    print()

    if len(missing_val_phrase) > 0:

        raise RuntimeError(
            "Some validation reference rows do not "
            "contain a <ref>...</ref> phrase."
        )

    val_df[
        "semantic_family"
    ] = val_df[
        "reference_phrase"
    ].apply(
        infer_family
    )

    unmatched_val = val_df[
        val_df[
            "semantic_family"
        ].isna()
    ]

    if len(unmatched_val) > 0:

        print(
            "Unmatched validation phrases:"
        )

        print(
            "-" * 70
        )

        for phrase in sorted(
            set(
                normalize_text(
                    value
                )
                for value
                in unmatched_val[
                    "reference_phrase"
                ].tolist()
            )
        ):

            print(
                f"  {phrase}"
            )

        print()

        raise RuntimeError(
            "Validation contains reference phrases "
            "that cannot be assigned to a family."
        )

    validation_family_counts = (
        val_df[
            "semantic_family"
        ]
        .value_counts()
        .to_dict()
    )

    validation_observed_families = sorted(
        set(
            val_df[
                "semantic_family"
            ]
            .tolist()
        )
    )

    validation_missing_families = sorted(
        set(EXPECTED_FAMILIES)
        - set(validation_observed_families)
    )

    print(
        f"Validation families observed: "
        f"{len(validation_observed_families)}"
    )

    print(
        f"Validation families absent: "
        f"{len(validation_missing_families)}"
    )

    if validation_missing_families:

        for family in validation_missing_families:

            print(
                f"  Absent: {family}"
            )

    else:

        print(
            "All 19 semantic families occur "
            "in validation reference data."
        )

    print()


# ============================================================
# SAVE DETAILED CSV
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


mapping_df.to_csv(
    CSV_OUTPUT,
    index=False,
)


# ============================================================
# BUILD JSON REPORT
# ============================================================

json_report = {

    "mapping_source":
        "BigEarthNet V2 32-class CLC vocabulary "
        "to 19-class semantic family grouping",

    "frozen_vocabulary_path":
        str(
            VOCAB_PATH
        ),

    "num_clc_classes":
        len(
            clc_values
        ),

    "clc_values":
        clc_values,

    "num_semantic_families":
        len(
            EXPECTED_FAMILIES
        ),

    "semantic_families":
        EXPECTED_FAMILIES,

    "clc_to_family":
        {
            str(clc):
                family
            for clc, family
            in sorted(
                clc_to_family.items()
            )
        },

    "family_to_clc":
        {
            family:
                EXPECTED_CLC_GROUPS[
                    family
                ]
            for family
            in EXPECTED_FAMILIES
        },

    "single_clc_families":
        single_clc_families,

    "multi_clc_families":
        multi_clc_families,

    "train_reference_rows":
        int(
            len(train_df)
        ),

    "train_unique_reference_phrases":
        int(
            train_df[
                "reference_phrase"
            ]
            .astype(str)
            .nunique()
        ),

    "train_family_counts":
        {
            family:
                int(
                    train_family_counts.get(
                        family,
                        0,
                    )
                )
            for family
            in EXPECTED_FAMILIES
        },

    "validation_family_counts":
        {
            family:
                int(
                    validation_family_counts.get(
                        family,
                        0,
                    )
                )
            for family
            in EXPECTED_FAMILIES
        },

    "phrase_source_column":
        "input",

    "phrase_extraction_pattern":
        "<ref>...</ref>",
}


with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_report,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "=" * 70
)

print(
    "C3 CLC -> SEMANTIC FAMILY MAPPING COMPLETE"
)

print(
    "=" * 70
)

print()


print(
    f"CLC classes: "
    f"{len(clc_values)}"
)

print(
    f"Semantic families: "
    f"{len(EXPECTED_FAMILIES)}"
)

print(
    f"Training reference rows audited: "
    f"{len(train_df):,}"
)

print(
    f"Training unique reference phrases: "
    f"{train_df['reference_phrase'].astype(str).nunique():,}"
)

print()


print(
    "All training reference phrases matched "
    "exactly one semantic family."
)

print(
    "Every frozen CLC class belongs to exactly "
    "one semantic family."
)

print()


print(
    "CSV output:"
)

print(
    CSV_OUTPUT
)

print()


print(
    "JSON output:"
)

print(
    JSON_OUTPUT
)

print()


print(
    "No original imagery, reference maps, "
    "CLC vocabulary, or semantic target tensors "
    "were modified."
)

print()