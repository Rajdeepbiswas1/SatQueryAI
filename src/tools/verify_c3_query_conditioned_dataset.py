from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd
import torch


# ============================================================
# C3 QUERY-CONDITIONED DATASET INTEGRATION VERIFICATION
# ============================================================
#
# This is a VERIFICATION script only.
#
# It does NOT:
#   - train a model
#   - modify a checkpoint
#   - modify source datasets
#   - modify existing targets
#   - modify cached features
#
# It verifies:
#
#   1. Query-conditioned targets align exactly with parquet rows.
#   2. Family indices / family names align.
#   3. Target tensors exactly match their source family target.
#   4. Train/validation patch isolation remains intact.
#   5. A small REAL batch can be assembled with:
#
#          S1          [B,196,768]
#          S2          [B,196,768]
#          text        [B,768]
#          target      [B,14,14]
#          target_box  [B,4]
#
# The complete DOFA cache is NOT loaded into memory.
# Only the shards needed for a tiny verification batch are
# read sequentially.
# ============================================================


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)


# ============================================================
# ANNOTATION FILES
# ============================================================

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


# ============================================================
# QUERY-CONDITIONED TARGETS
# ============================================================

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

TRAIN_QUERY_TARGETS = (
    TARGET_DIR
    / "train_query_conditioned_targets.pt"
)

VAL_QUERY_TARGETS = (
    TARGET_DIR
    / "val_query_conditioned_targets.pt"
)

TRAIN_FAMILY_TARGETS = (
    TARGET_DIR
    / "train_family_targets.pt"
)

VAL_FAMILY_TARGETS = (
    TARGET_DIR
    / "val_family_targets.pt"
)

VOCAB_PATH = (
    TARGET_DIR
    / "clc_vocabulary.json"
)

FAMILY_MAPPING_PATH = (
    TARGET_DIR
    / "clc_family_mapping.json"
)


# ============================================================
# CACHE PATHS
# ============================================================

DOFA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

TEXT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grounding_text_embeddings"
    / "grounding_text_embedding_shards"
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

# Small enough to keep local verification lightweight.
BATCH_SIZE = 4

FLOAT_TOLERANCE = 1e-7


# ============================================================
# FROZEN FAMILY ORDER
# ============================================================

FAMILY_NAMES = [
    "urban fabric",
    "industrial or commercial units",
    "arable land",
    "permanent crops",
    "pastures",
    "complex cultivation patterns",
    "land principally occupied by agriculture with significant areas of natural vegetation",
    "agro-forestry areas",
    "broad-leaved forest",
    "coniferous forest",
    "mixed forest",
    "natural grassland and sparsely vegetated areas",
    "moors, heathland and sclerophyllous vegetation",
    "transitional woodland, shrub",
    "beaches, dunes, sands",
    "inland wetlands",
    "coastal wetlands",
    "inland waters",
    "marine waters",
]


# ============================================================
# REGEX FOR <ref>...</ref>
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

    text = str(
        value
    ).lower()

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
    )

    match = REF_PATTERN.search(
        text
    )

    if match is None:
        return None

    return (
        match.group(1)
        .strip()
    )


# ============================================================
# FAMILY INFERENCE
# ============================================================

FAMILY_PATTERNS = {

    "urban fabric": [
        "urban fabric",
    ],

    "industrial or commercial units": [
        "industrial or commercial units",
    ],

    "arable land": [
        "arable land",
    ],

    "permanent crops": [
        "permanent crops",
        "annual crops associated with permanent crops",
    ],

    "pastures": [
        "pastures",
    ],

    "complex cultivation patterns": [
        "complex cultivation patterns",
    ],

    "land principally occupied by agriculture with significant areas of natural vegetation": [
        "land principally occupied by agriculture with significant areas of natural vegetation",
    ],

    "agro-forestry areas": [
        "agro-forestry areas",
    ],

    "broad-leaved forest": [
        "broad-leaved forest",
    ],

    "coniferous forest": [
        "coniferous forest",
    ],

    "mixed forest": [
        "mixed forest",
    ],

    "natural grassland and sparsely vegetated areas": [
        "natural grassland and sparsely vegetated areas",
        "natural grasslands or sparsely vegetated areas",
        "natural grassland",
        "sparsely vegetated areas",
    ],

    "moors, heathland and sclerophyllous vegetation": [
        "moors, heathland and sclerophyllous vegetation",
        "moors, heathland, or sclerophyllous vegetation",
        "moors and heathland",
        "sclerophyllous vegetation",
    ],

    "transitional woodland, shrub": [
        "transitional woodland, shrub",
        "transitional woodlands or shrubs",
        "transitional woodland/shrub",
    ],

    "beaches, dunes, sands": [
        "beaches, dunes, sands",
        "beaches, dunes, or sands",
    ],

    "inland wetlands": [
        "inland wetlands",
        "inland marshes",
        "peat bogs",
        "peatbogs",
    ],

    "coastal wetlands": [
        "coastal wetlands",
        "salt marshes",
        "salines",
        "intertidal flats",
    ],

    "inland waters": [
        "inland waters",
        "water courses",
        "water bodies",
    ],

    "marine waters": [
        "marine waters",
        "coastal lagoons",
        "estuaries",
        "sea and ocean",
    ],
}


def infer_family(
    phrase
):

    if phrase is None:
        return None

    text = normalize_text(
        phrase
    )

    matches = []

    for family in FAMILY_NAMES:

        patterns = FAMILY_PATTERNS[
            family
        ]

        for pattern in patterns:

            if (
                normalize_text(
                    pattern
                )
                in text
            ):
                matches.append(
                    family
                )
                break

    matches = list(
        dict.fromkeys(
            matches
        )
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:

        raise RuntimeError(
            "Ambiguous family assignment.\n"
            f"Phrase: {phrase}\n"
            f"Families: {matches}"
        )

    return matches[0]


# ============================================================
# BOX PARSING
# ============================================================

def parse_box(
    value
):

    if isinstance(
        value,
        str,
    ):

        numbers = re.findall(
            r"-?(?:\d+(?:\.\d*)?|\.\d+)",
            value,
        )

        if len(numbers) < 4:

            raise RuntimeError(
                f"Could not parse target box:\n"
                f"{value}"
            )

        values = [
            float(
                numbers[i]
            )
            for i in range(4)
        ]

    else:

        values = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1).tolist()

        if len(values) < 4:

            raise RuntimeError(
                f"Invalid target box:\n"
                f"{value}"
            )

        values = values[:4]

    values = [
        float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )
        for value in values
    ]

    x1, y1, x2, y2 = values

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return torch.tensor(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dtype=torch.float32,
    )


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

def require_file(
    path,
    label,
):

    print(
        f"{label}:"
    )

    print(
        f"  {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n"
            f"{path}"
        )

    print(
        "  FOUND"
    )


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations(
    path,
    split_name,
):

    df = pd.read_parquet(
        path
    )

    required = [
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
        in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{split_name}: missing columns: "
            f"{missing}"
        )

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

    df[
        "patch_id"
    ] = df[
        "patch_id"
    ].astype(str)

    df[
        "original_row_index"
    ] = df[
        "original_row_index"
    ].astype(int)

    return df.reset_index(
        drop=True
    )


# ============================================================
# LOAD QUERY TARGET PAYLOAD
# ============================================================

def load_query_targets(
    path,
    split_name,
):

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
            f"{split_name}: query target payload "
            "is not a dictionary."
        )

    required = [
        "targets",
        "patch_ids",
        "original_row_index",
        "family_indices",
        "family_names_per_row",
        "reference_phrases",
        "annotation_ids",
        "source_patch_indices",
        "family_names",
    ]

    missing = [
        key
        for key
        in required
        if key not in data
    ]

    if missing:

        raise RuntimeError(
            f"{split_name}: query target payload "
            f"is missing keys: {missing}"
        )

    targets = data[
        "targets"
    ]

    if not torch.is_tensor(
        targets
    ):

        raise RuntimeError(
            f"{split_name}: targets is not tensor."
        )

    targets = targets.float()

    if targets.ndim != 3:

        raise RuntimeError(
            f"{split_name}: expected target shape "
            "[rows,14,14], got "
            f"{tuple(targets.shape)}"
        )

    if targets.shape[-2:] != (
        14,
        14,
    ):

        raise RuntimeError(
            f"{split_name}: expected 14x14 targets, "
            f"got {tuple(targets.shape[-2:])}"
        )

    return data


# ============================================================
# VERIFY QUERY TARGET ALIGNMENT
# ============================================================

def verify_query_alignment(
    df,
    data,
    split_name,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{split_name.upper()} QUERY-TARGET ALIGNMENT"
    )

    print(
        "=" * 70
    )

    rows = len(
        df
    )

    targets = data[
        "targets"
    ]

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if targets.shape[0] != rows:

        raise RuntimeError(
            f"{split_name}: target row count "
            f"{targets.shape[0]} != parquet rows {rows}"
        )

    print(
        f"Rows: {rows:,}"
    )

    print(
        f"Target tensor: "
        f"{tuple(targets.shape)}"
    )

    # --------------------------------------------------------
    # Patch IDs
    # --------------------------------------------------------

    expected_patch_ids = [
        str(value)
        for value
        in df[
            "patch_id"
        ].tolist()
    ]

    actual_patch_ids = [
        str(value)
        for value
        in data[
            "patch_ids"
        ]
    ]

    if (
        expected_patch_ids
        != actual_patch_ids
    ):

        for i, (
            expected,
            actual,
        ) in enumerate(
            zip(
                expected_patch_ids,
                actual_patch_ids,
            )
        ):

            if expected != actual:

                raise RuntimeError(
                    f"{split_name}: patch_id mismatch "
                    f"at row {i}: "
                    f"expected {expected}, "
                    f"got {actual}"
                )

    print(
        "Patch-ID alignment: PASS"
    )

    # --------------------------------------------------------
    # Original annotation row
    # --------------------------------------------------------

    expected_source_indices = [
        int(value)
        for value
        in df[
            "original_row_index"
        ].tolist()
    ]

    actual_source_indices = [
        int(value)
        for value
        in data[
            "original_row_index"
        ]
    ]

    if (
        expected_source_indices
        != actual_source_indices
    ):

        for i, (
            expected,
            actual,
        ) in enumerate(
            zip(
                expected_source_indices,
                actual_source_indices,
            )
        ):

            if expected != actual:

                raise RuntimeError(
                    f"{split_name}: original_row_index "
                    f"mismatch at row {i}: "
                    f"expected {expected}, "
                    f"got {actual}"
                )

    print(
        "Original-row-index alignment: PASS"
    )

    # --------------------------------------------------------
    # Annotation IDs
    # --------------------------------------------------------

    expected_ids = [
        str(value)
        for value
        in df[
            "ID"
        ].tolist()
    ]

    actual_ids = [
        str(value)
        for value
        in data[
            "annotation_ids"
        ]
    ]

    if (
        expected_ids
        != actual_ids
    ):

        for i, (
            expected,
            actual,
        ) in enumerate(
            zip(
                expected_ids,
                actual_ids,
            )
        ):

            if expected != actual:

                raise RuntimeError(
                    f"{split_name}: annotation ID "
                    f"mismatch at row {i}."
                )

    print(
        "Annotation-ID alignment: PASS"
    )

    # --------------------------------------------------------
    # Reference phrases
    # --------------------------------------------------------

    mismatched_phrases = 0

    for i in range(
        rows
    ):

        expected_phrase = (
            extract_reference_phrase(
                df.iloc[i][
                    "input"
                ]
            )
        )

        actual_phrase = (
            data[
                "reference_phrases"
            ][
                i
            ]
        )

        if normalize_text(
            expected_phrase
        ) != normalize_text(
            actual_phrase
        ):

            if mismatched_phrases < 5:

                print(
                    "\nPhrase mismatch:"
                )

                print(
                    f"  Row: {i}"
                )

                print(
                    f"  Expected: {expected_phrase}"
                )

                print(
                    f"  Actual:   {actual_phrase}"
                )

            mismatched_phrases += 1

    if mismatched_phrases:

        raise RuntimeError(
            f"{split_name}: "
            f"{mismatched_phrases:,} phrase mismatches."
        )

    print(
        "Reference-phrase alignment: PASS"
    )

    # --------------------------------------------------------
    # Semantic family
    # --------------------------------------------------------

    family_indices = (
        data[
            "family_indices"
        ]
    )

    if not torch.is_tensor(
        family_indices
    ):

        family_indices = torch.tensor(
            family_indices,
            dtype=torch.long,
        )

    family_indices = (
        family_indices
        .long()
        .cpu()
    )

    if family_indices.numel() != rows:

        raise RuntimeError(
            f"{split_name}: family-index count mismatch."
        )

    actual_families = list(
        data[
            "family_names_per_row"
        ]
    )

    if len(
        actual_families
    ) != rows:

        raise RuntimeError(
            f"{split_name}: family-name count mismatch."
        )

    family_errors = 0

    for i in range(
        rows
    ):

        phrase = extract_reference_phrase(
            df.iloc[i][
                "input"
            ]
        )

        inferred = infer_family(
            phrase
        )

        stored = actual_families[
            i
        ]

        stored_index = int(
            family_indices[
                i
            ]
        )

        if inferred != stored:

            if family_errors < 5:

                print()
                print(
                    "Family mismatch:"
                )

                print(
                    f"  Row: {i}"
                )

                print(
                    f"  Phrase: {phrase}"
                )

                print(
                    f"  Inferred: {inferred}"
                )

                print(
                    f"  Stored: {stored}"
                )

            family_errors += 1

        if stored_index < 0:
            raise RuntimeError(
                f"{split_name}: negative family index "
                f"at row {i}."
            )

        if stored_index >= len(
            FAMILY_NAMES
        ):
            raise RuntimeError(
                f"{split_name}: family index "
                f"{stored_index} out of range."
            )

        expected_family_from_index = (
            FAMILY_NAMES[
                stored_index
            ]
        )

        if (
            expected_family_from_index
            != stored
        ):

            raise RuntimeError(
                f"{split_name}: family index/name "
                f"mismatch at row {i}."
            )

    if family_errors:

        raise RuntimeError(
            f"{split_name}: "
            f"{family_errors:,} semantic-family mismatches."
        )

    print(
        "Semantic-family alignment: PASS"
    )

    # --------------------------------------------------------
    # Source patch indices
    # --------------------------------------------------------

    source_patch_indices = (
        data[
            "source_patch_indices"
        ]
    )

    if not torch.is_tensor(
        source_patch_indices
    ):

        source_patch_indices = torch.tensor(
            source_patch_indices,
            dtype=torch.long,
        )

    source_patch_indices = (
        source_patch_indices
        .long()
        .cpu()
    )

    if source_patch_indices.numel() != rows:

        raise RuntimeError(
            f"{split_name}: source patch-index "
            "count mismatch."
        )

    print(
        "Source patch-index alignment: PASS"
    )

    # --------------------------------------------------------
    # Target numeric checks
    # --------------------------------------------------------

    minimum = float(
        targets.min()
    )

    maximum = float(
        targets.max()
    )

    if minimum < -FLOAT_TOLERANCE:

        raise RuntimeError(
            f"{split_name}: target minimum "
            f"is {minimum}"
        )

    if maximum > 1.0 + FLOAT_TOLERANCE:

        raise RuntimeError(
            f"{split_name}: target maximum "
            f"is {maximum}"
        )

    if not torch.isfinite(
        targets
    ).all():

        raise RuntimeError(
            f"{split_name}: target tensor contains "
            "NaN or Inf."
        )

    print(
        f"Target range: [{minimum:.8f}, {maximum:.8f}]"
    )

    print(
        "Target numeric validity: PASS"
    )

    return {
        "rows": rows,
        "target_shape": list(
            targets.shape
        ),
        "target_min": minimum,
        "target_max": maximum,
    }


# ============================================================
# VERIFY TARGETS AGAINST FAMILY SOURCE
# ============================================================

def verify_against_family_targets(
    query_data,
    family_target_path,
    split_name,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{split_name.upper()} SOURCE-TARGET "
        "RECONSTRUCTION"
    )

    print(
        "=" * 70
    )

    family_data = torch.load(
        family_target_path,
        map_location="cpu",
        weights_only=False,
    )

    source_targets = (
        family_data[
            "targets"
        ]
        .float()
    )

    query_targets = (
        query_data[
            "targets"
        ]
        .float()
    )

    patch_indices = (
        query_data[
            "source_patch_indices"
        ]
        .long()
    )

    family_indices = (
        query_data[
            "family_indices"
        ]
        .long()
    )

    rows = query_targets.shape[0]

    # --------------------------------------------------------
    # Reconstruct in chunks to keep temporary memory small.
    # --------------------------------------------------------

    max_error = 0.0
    mean_error_sum = 0.0
    element_count = 0

    chunk_size = 512

    for start in range(
        0,
        rows,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            rows,
        )

        chunk_patch_indices = (
            patch_indices[
                start:end
            ]
        )

        chunk_family_indices = (
            family_indices[
                start:end
            ]
        )

        chunk_source = (
            source_targets[
                chunk_patch_indices
            ]
        )

        chunk_source = (
            chunk_source[
                torch.arange(
                    end - start,
                    dtype=torch.long,
                ),
                chunk_family_indices,
            ]
        )

        chunk_query = (
            query_targets[
                start:end
            ]
        )

        diff = (
            chunk_query
            - chunk_source
        ).abs()

        chunk_max = float(
            diff.max()
        )

        if chunk_max > max_error:
            max_error = chunk_max

        mean_error_sum += float(
            diff.sum()
        )

        element_count += diff.numel()

    mean_error = (
        mean_error_sum
        / element_count
    )

    print(
        f"Max reconstruction error: "
        f"{max_error:.12e}"
    )

    print(
        f"Mean reconstruction error: "
        f"{mean_error:.12e}"
    )

    if max_error > FLOAT_TOLERANCE:

        raise RuntimeError(
            f"{split_name}: query-conditioned "
            "targets do not exactly reconstruct "
            "from family targets."
        )

    print(
        "Family-target reconstruction: PASS"
    )

    return {
        "max_error":
            max_error,

        "mean_error":
            mean_error,
    }


# ============================================================
# BOXES FROM PARQUET
# ============================================================

def build_target_boxes(
    df,
):

    boxes = []

    for i in range(
        len(df)
    ):

        boxes.append(
            parse_box(
                df.iloc[i][
                    "output"
                ]
            )
        )

    return torch.stack(
        boxes,
        dim=0,
    )


# ============================================================
# DOFA SHARD SEARCH
# ============================================================

def find_dofa_features_for_patch_ids(
    patch_ids,
    max_required,
):

    patch_ids = [
        str(value)
        for value
        in patch_ids[
            :max_required
        ]
    ]

    remaining = set(
        patch_ids
    )

    found = {}

    shard_files = sorted(
        [
            path
            for path
            in DOFA_ROOT.iterdir()
            if path.suffix.lower() == ".pt"
        ]
    )

    if not shard_files:

        raise RuntimeError(
            f"No DOFA .pt shards found in:\n"
            f"{DOFA_ROOT}"
        )

    print()
    print(
        "Searching DOFA shards for verification batch..."
    )

    print(
        f"Total DOFA shards available: "
        f"{len(shard_files)}"
    )

    for shard_number, path in enumerate(
        shard_files,
        start=1,
    ):

        if not remaining:
            break

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
                f"Invalid DOFA shard:\n{path}"
            )

        if (
            "patch_ids" not in data
            or "s1" not in data
            or "s2" not in data
        ):

            raise RuntimeError(
                f"Missing DOFA keys in:\n{path}"
            )

        shard_patch_ids = [
            str(value)
            for value
            in data[
                "patch_ids"
            ]
        ]

        for local_index, patch_id in enumerate(
            shard_patch_ids
        ):

            if patch_id in remaining:

                found[
                    patch_id
                ] = {
                    "s1":
                        data[
                            "s1"
                        ][
                            local_index
                        ].float(),

                    "s2":
                        data[
                            "s2"
                        ][
                            local_index
                        ].float(),
                }

                remaining.remove(
                    patch_id
                )

        del data

        if shard_number % 5 == 0:

            print(
                f"  Scanned {shard_number} "
                f"shards; found "
                f"{len(found)}/{len(patch_ids)}"
            )

    if remaining:

        raise RuntimeError(
            "Could not find all verification "
            f"patches in DOFA cache.\n"
            f"Missing: {sorted(remaining)}"
        )

    print(
        f"Found all {len(found)} verification "
        "patch feature sets."
    )

    return found


# ============================================================
# TEXT SHARD SEARCH
# ============================================================

def find_text_embeddings_for_row_indices(
    row_indices,
):

    row_indices = [
        int(value)
        for value
        in row_indices
    ]

    remaining = set(
        row_indices
    )

    found = {}

    shard_files = sorted(
        [
            path
            for path
            in TEXT_ROOT.iterdir()
            if path.suffix.lower() == ".pt"
        ]
    )

    if not shard_files:

        raise RuntimeError(
            f"No text-embedding .pt shards found in:\n"
            f"{TEXT_ROOT}"
        )

    print()
    print(
        "Searching text-embedding shards "
        "for verification batch..."
    )

    print(
        f"Total text shards available: "
        f"{len(shard_files)}"
    )

    for shard_number, path in enumerate(
        shard_files,
        start=1,
    ):

        if not remaining:
            break

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
                f"Invalid text shard:\n{path}"
            )

        if (
            "source_row_index" not in data
            or "text_embeddings" not in data
        ):

            raise RuntimeError(
                f"Missing text keys in:\n{path}"
            )

        source_indices = [
            int(value)
            for value
            in data[
                "source_row_index"
            ]
        ]

        for local_index, row_index in enumerate(
            source_indices
        ):

            if row_index in remaining:

                found[
                    row_index
                ] = (
                    data[
                        "text_embeddings"
                    ][
                        local_index
                    ]
                    .float()
                )

                remaining.remove(
                    row_index
                )

        del data

    if remaining:

        raise RuntimeError(
            "Could not find all verification "
            "text embeddings.\n"
            f"Missing source rows: "
            f"{sorted(remaining)}"
        )

    print(
        f"Found all {len(found)} verification "
        "text embeddings."
    )

    return found


# ============================================================
# REAL BATCH ASSEMBLY
# ============================================================

def build_real_batch(
    df,
    query_data,
    split_name,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{split_name.upper()} REAL BATCH INTEGRATION TEST"
    )

    print(
        "=" * 70
    )

    batch_count = min(
        BATCH_SIZE,
        len(df),
    )

    # --------------------------------------------------------
    # Use the first rows deliberately.
    # --------------------------------------------------------

    batch_df = df.iloc[
        :batch_count
    ].reset_index(
        drop=True
    )

    patch_ids = [
        str(value)
        for value
        in batch_df[
            "patch_id"
        ]
    ]

    row_indices = [
        int(value)
        for value
        in batch_df[
            "original_row_index"
        ]
    ]

    # --------------------------------------------------------
    # DOFA.
    # --------------------------------------------------------

    dofa_found = (
        find_dofa_features_for_patch_ids(
            patch_ids,
            batch_count,
        )
    )

    s1_list = []
    s2_list = []

    for patch_id in patch_ids:

        item = dofa_found[
            patch_id
        ]

        s1 = item[
            "s1"
        ]

        s2 = item[
            "s2"
        ]

        if tuple(
            s1.shape
        ) != (
            196,
            768,
        ):

            raise RuntimeError(
                f"{split_name}: unexpected S1 shape "
                f"for {patch_id}: "
                f"{tuple(s1.shape)}"
            )

        if tuple(
            s2.shape
        ) != (
            196,
            768,
        ):

            raise RuntimeError(
                f"{split_name}: unexpected S2 shape "
                f"for {patch_id}: "
                f"{tuple(s2.shape)}"
            )

        s1_list.append(
            s1
        )

        s2_list.append(
            s2
        )

    s1_batch = torch.stack(
        s1_list,
        dim=0,
    )

    s2_batch = torch.stack(
        s2_list,
        dim=0,
    )

    # --------------------------------------------------------
    # Text.
    # --------------------------------------------------------

    text_found = (
        find_text_embeddings_for_row_indices(
            row_indices
        )
    )

    text_list = []

    for row_index in row_indices:

        embedding = text_found[
            row_index
        ]

        if tuple(
            embedding.shape
        ) != (
            768,
        ):

            raise RuntimeError(
                f"{split_name}: unexpected text "
                f"embedding shape for row "
                f"{row_index}: "
                f"{tuple(embedding.shape)}"
            )

        text_list.append(
            embedding
        )

    text_batch = torch.stack(
        text_list,
        dim=0,
    )

    # --------------------------------------------------------
    # Query-conditioned target.
    # --------------------------------------------------------

    query_target_batch = (
        query_data[
            "targets"
        ][
            :batch_count
        ]
        .float()
        .contiguous()
    )

    if tuple(
        query_target_batch.shape
    ) != (
        batch_count,
        14,
        14,
    ):

        raise RuntimeError(
            f"{split_name}: unexpected query target "
            f"batch shape: "
            f"{tuple(query_target_batch.shape)}"
        )

    # --------------------------------------------------------
    # Ground-truth boxes.
    # --------------------------------------------------------

    target_boxes = build_target_boxes(
        batch_df
    )

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    family_indices = (
        query_data[
            "family_indices"
        ][
            :batch_count
        ]
        .long()
    )

    family_names = [
        str(value)
        for value
        in query_data[
            "family_names_per_row"
        ][
            :batch_count
        ]
    ]

    phrases = [
        str(value)
        for value
        in query_data[
            "reference_phrases"
        ][
            :batch_count
        ]
    ]

    # --------------------------------------------------------
    # FINAL BATCH SHAPES.
    # --------------------------------------------------------

    print()
    print(
        "REAL BATCH:"
    )

    print(
        f"  S1:            {tuple(s1_batch.shape)}"
    )

    print(
        f"  S2:            {tuple(s2_batch.shape)}"
    )

    print(
        f"  Text:          {tuple(text_batch.shape)}"
    )

    print(
        f"  Query target:  {tuple(query_target_batch.shape)}"
    )

    print(
        f"  Target box:    {tuple(target_boxes.shape)}"
    )

    # --------------------------------------------------------
    # Print actual examples.
    # --------------------------------------------------------

    print()
    print(
        "BATCH ROWS:"
    )

    for i in range(
        batch_count
    ):

        print(
            f"  Row {i}:"
        )

        print(
            f"    patch_id:  "
            f"{patch_ids[i]}"
        )

        print(
            f"    row_index: "
            f"{row_indices[i]}"
        )

        print(
            f"    phrase:    "
            f"{phrases[i]}"
        )

        print(
            f"    family:    "
            f"{family_names[i]}"
        )

        print(
            f"    family_idx:"
            f" {int(family_indices[i])}"
        )

        print(
            f"    box:       "
            f"{target_boxes[i].tolist()}"
        )

        print(
            f"    target min/max: "
            f"{float(query_target_batch[i].min()):.6f} / "
            f"{float(query_target_batch[i].max()):.6f}"
        )

    # --------------------------------------------------------
    # Final shape assertions.
    # --------------------------------------------------------

    assert s1_batch.dtype == torch.float32
    assert s2_batch.dtype == torch.float32
    assert text_batch.dtype == torch.float32
    assert query_target_batch.dtype == torch.float32
    assert target_boxes.dtype == torch.float32

    assert s1_batch.shape == (
        batch_count,
        196,
        768,
    )

    assert s2_batch.shape == (
        batch_count,
        196,
        768,
    )

    assert text_batch.shape == (
        batch_count,
        768,
    )

    assert query_target_batch.shape == (
        batch_count,
        14,
        14,
    )

    assert target_boxes.shape == (
        batch_count,
        4,
    )

    print()
    print(
        "Real-batch tensor interface: PASS"
    )

    return {
        "batch_size":
            batch_count,

        "s1_shape":
            list(
                s1_batch.shape
            ),

        "s2_shape":
            list(
                s2_batch.shape
            ),

        "text_shape":
            list(
                text_batch.shape
            ),

        "query_target_shape":
            list(
                query_target_batch.shape
            ),

        "target_box_shape":
            list(
                target_boxes.shape
            ),

        "patch_ids":
            patch_ids,

        "row_indices":
            row_indices,

        "phrases":
            phrases,

        "families":
            family_names,
    }


# ============================================================
# TRAIN / VALIDATION PATCH ISOLATION
# ============================================================

def verify_split_isolation(
    train_df,
    val_df,
    train_query,
    val_query,
):

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

    train_df_patches = set(
        train_df[
            "patch_id"
        ]
    )

    val_df_patches = set(
        val_df[
            "patch_id"
        ]
    )

    train_query_patches = set(
        train_query[
            "patch_ids"
        ]
    )

    val_query_patches = set(
        val_query[
            "patch_ids"
        ]
    )

    df_overlap = (
        train_df_patches
        &
        val_df_patches
    )

    query_overlap = (
        train_query_patches
        &
        val_query_patches
    )

    print(
        f"Train annotation patches: "
        f"{len(train_df_patches):,}"
    )

    print(
        f"Validation annotation patches: "
        f"{len(val_df_patches):,}"
    )

    print(
        f"Train query-target patches: "
        f"{len(train_query_patches):,}"
    )

    print(
        f"Validation query-target patches: "
        f"{len(val_query_patches):,}"
    )

    print(
        f"Annotation split overlap: "
        f"{len(df_overlap):,}"
    )

    print(
        f"Query-target split overlap: "
        f"{len(query_overlap):,}"
    )

    if df_overlap:

        raise RuntimeError(
            "Train/validation overlap detected "
            "in annotation files."
        )

    if query_overlap:

        raise RuntimeError(
            "Train/validation overlap detected "
            "in query-conditioned targets."
        )

    print(
        "Patch isolation: PASS"
    )

    return {
        "annotation_overlap":
            len(df_overlap),

        "query_target_overlap":
            len(query_overlap),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "C3 QUERY-CONDITIONED DATASET "
        "INTEGRATION VERIFICATION"
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

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "REQUIRED FILE CHECK"
    )

    print(
        "=" * 70
    )

    required = [

        (
            TRAIN_PARQUET,
            "Train annotation file",
        ),

        (
            VAL_PARQUET,
            "Validation annotation file",
        ),

        (
            TRAIN_QUERY_TARGETS,
            "Train query-conditioned targets",
        ),

        (
            VAL_QUERY_TARGETS,
            "Validation query-conditioned targets",
        ),

        (
            TRAIN_FAMILY_TARGETS,
            "Train family targets",
        ),

        (
            VAL_FAMILY_TARGETS,
            "Validation family targets",
        ),

        (
            VOCAB_PATH,
            "CLC vocabulary",
        ),

        (
            FAMILY_MAPPING_PATH,
            "CLC family mapping",
        ),

        (
            DOFA_ROOT,
            "DOFA cache directory",
        ),

        (
            TEXT_ROOT,
            "Text embedding cache directory",
        ),
    ]

    for path, label in required:

        if path.exists():

            print(
                f"FOUND: {label}"
            )

        else:

            raise FileNotFoundError(
                f"MISSING: {label}\n"
                f"{path}"
            )

    # ========================================================
    # VOCABULARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FROZEN VOCABULARY CHECK"
    )

    print(
        "=" * 70
    )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        vocab = json.load(
            f
        )

    clc_classes = [
        int(value)
        for value
        in vocab[
            "clc_vocabulary"
        ]
    ]

    if len(
        clc_classes
    ) != 32:

        raise RuntimeError(
            "Expected 32 frozen CLC classes."
        )

    print(
        "Frozen CLC vocabulary: 32 classes"
    )

    print(
        "Vocabulary check: PASS"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = load_annotations(
        TRAIN_PARQUET,
        "train",
    )

    val_df = load_annotations(
        VAL_PARQUET,
        "validation",
    )

    train_query = load_query_targets(
        TRAIN_QUERY_TARGETS,
        "train",
    )

    val_query = load_query_targets(
        VAL_QUERY_TARGETS,
        "validation",
    )

    print()
    print(
        f"Train annotations: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation annotations: "
        f"{len(val_df):,}"
    )

    print(
        f"Train query targets: "
        f"{train_query['targets'].shape[0]:,}"
    )

    print(
        f"Validation query targets: "
        f"{val_query['targets'].shape[0]:,}"
    )

    # ========================================================
    # QUERY ALIGNMENT
    # ========================================================

    train_alignment = (
        verify_query_alignment(
            train_df,
            train_query,
            "train",
        )
    )

    val_alignment = (
        verify_query_alignment(
            val_df,
            val_query,
            "validation",
        )
    )

    # ========================================================
    # SOURCE FAMILY TARGET RECONSTRUCTION
    # ========================================================

    train_reconstruction = (
        verify_against_family_targets(
            train_query,
            TRAIN_FAMILY_TARGETS,
            "train",
        )
    )

    val_reconstruction = (
        verify_against_family_targets(
            val_query,
            VAL_FAMILY_TARGETS,
            "validation",
        )
    )

    # ========================================================
    # SPLIT ISOLATION
    # ========================================================

    split_result = (
        verify_split_isolation(
            train_df,
            val_df,
            train_query,
            val_query,
        )
    )

    # ========================================================
    # REAL BATCH
    # ========================================================

    train_batch_result = (
        build_real_batch(
            train_df,
            train_query,
            "train",
        )
    )

    val_batch_result = (
        build_real_batch(
            val_df,
            val_query,
            "validation",
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {

        "project":
            "SatQueryAI",

        "stage":
            "C3 query-conditioned dataset integration verification",

        "device":
            "cpu",

        "gpu":
            False,

        "train_alignment":
            train_alignment,

        "validation_alignment":
            val_alignment,

        "train_reconstruction":
            train_reconstruction,

        "validation_reconstruction":
            val_reconstruction,

        "split_isolation":
            split_result,

        "train_real_batch":
            train_batch_result,

        "validation_real_batch":
            val_batch_result,

        "overall":
            {
                "query_target_alignment":
                    True,

                "source_family_target_reconstruction":
                    True,

                "patch_isolation":
                    True,

                "real_batch_tensor_interface":
                    True,
            },
    }

    output_path = (
        TARGET_DIR
        / "query_conditioned_dataset_verification.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "C3 QUERY-CONDITIONED DATASET VERIFICATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "QUERY-TARGET ALIGNMENT: PASS"
    )

    print(
        "FAMILY-TARGET RECONSTRUCTION: PASS"
    )

    print(
        "TRAIN/VALIDATION ISOLATION: PASS"
    )

    print(
        "REAL BATCH INTERFACE: PASS"
    )

    print()

    print(
        "VERIFICATION REPORT:"
    )

    print(
        f"  {output_path}"
    )

    print()

    print(
        "GPU was not used."
    )

    print(
        "No model was trained."
    )

    print(
        "No existing project data or checkpoint "
        "was modified."
    )


if __name__ == "__main__":
    main()