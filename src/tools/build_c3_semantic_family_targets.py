from pathlib import Path
import json
import csv

import torch


# ============================================================
# C3 32-CLASS CLC -> 19-CLASS SEMANTIC FAMILY TARGETS
# ============================================================
#
# Purpose
# -------
# Convert the already-verified dense CLC targets:
#
#       [N, 32, 14, 14]
#
# into semantic-family targets:
#
#       [N, 19, 14, 14]
#
# using the frozen BigEarthNet CLC -> semantic-family mapping.
#
# Mathematical operation:
#
#     Family_k(x,y) =
#         sum CLC_c(x,y), for c belonging to family k
#
# This is an aggregation only.
#
# Important:
#   - The original 32-class tensors are READ ONLY.
#   - The frozen CLC vocabulary is READ ONLY.
#   - No images are modified.
#   - No reference maps are modified.
#   - No model/checkpoint is modified.
#
# The script also verifies:
#
#   1. source tensor shape
#   2. source class order
#   3. source patch IDs
#   4. output shape
#   5. output range
#   6. per-cell probability/mass conservation
#   7. per-patch family-area conservation
#   8. total-area conservation
#
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

TRAIN_TARGET_PATH = (
    TARGET_DIR
    / "train_semantic_targets.pt"
)

VAL_TARGET_PATH = (
    TARGET_DIR
    / "val_semantic_targets.pt"
)

VOCAB_PATH = (
    TARGET_DIR
    / "clc_vocabulary.json"
)

TRAIN_OUTPUT_PATH = (
    TARGET_DIR
    / "train_family_targets.pt"
)

VAL_OUTPUT_PATH = (
    TARGET_DIR
    / "val_family_targets.pt"
)

MANIFEST_OUTPUT_PATH = (
    TARGET_DIR
    / "family_target_manifest.json"
)

DISTRIBUTION_CSV_PATH = (
    TARGET_DIR
    / "family_target_distribution.csv"
)


# ============================================================
# OFFICIAL 32 -> 19 GROUPING ALREADY AUDITED
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


EXPECTED_CLC = sorted(
    clc
    for family in FAMILY_NAMES
    for clc in FAMILY_TO_CLC[
        family
    ]
)


# ============================================================
# BASIC VALIDATION HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(
        message
    )


def load_json(path):
    if not path.exists():
        fail(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_pt(path):
    if not path.exists():
        fail(
            f"Required tensor file not found:\n{path}"
        )

    print(
        f"Loading:\n  {path}"
    )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return data


# ============================================================
# INSPECT / EXTRACT SOURCE TARGET TENSOR
# ============================================================

def extract_target_tensor(
    data,
    path,
):
    """
    Extract the [N,32,H,W] tensor from the saved .pt file.

    The script supports either:

        torch.Tensor

    or a dictionary containing one of several explicit target
    tensor keys.

    We do not silently accept arbitrary tensors:
    the final extracted tensor must be exactly 4-D with
    channel dimension 32.
    """

    if torch.is_tensor(data):

        tensor = data

        print(
            "Saved object type: Tensor"
        )

    elif isinstance(
        data,
        dict,
    ):

        print(
            "Saved object type: dict"
        )

        print(
            "Dictionary keys:"
        )

        for key in data.keys():

            print(
                f"  {key}"
            )

        candidate_keys = [
            "targets",
            "target",
            "semantic_targets",
            "clc_targets",
            "maps",
            "tensor",
        ]

        tensor = None

        for key in candidate_keys:

            if key in data and torch.is_tensor(
                data[key]
            ):

                candidate = data[key]

                if (
                    candidate.ndim == 4
                    and candidate.shape[1] == 32
                ):

                    tensor = candidate

                    print(
                        f"Using target tensor key: "
                        f"{key}"
                    )

                    break

        if tensor is None:

            fail(
                "Could not identify a [N,32,H,W] "
                f"target tensor in:\n{path}"
            )

    else:

        fail(
            "Unsupported saved object type "
            f"{type(data)} in:\n{path}"
        )

    if tensor.ndim != 4:

        fail(
            "Expected 4-D target tensor but found "
            f"shape {tuple(tensor.shape)} in:\n{path}"
        )

    if tensor.shape[1] != 32:

        fail(
            "Expected exactly 32 CLC channels but found "
            f"{tensor.shape[1]} in:\n{path}"
        )

    if tensor.dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
    ):

        fail(
            "Expected floating-point target tensor but found "
            f"dtype {tensor.dtype} in:\n{path}"
        )

    return tensor.float()


# ============================================================
# PATCH-ID EXTRACTION
# ============================================================

def extract_patch_ids(
    data,
    expected_count,
    split_name,
):
    """
    Try to recover patch IDs when they were stored alongside
    the tensor.

    Patch IDs are metadata only; the aggregation itself depends
    only on tensor channel order.
    """

    if not isinstance(
        data,
        dict,
    ):

        return None

    candidate_keys = [
        "patch_ids",
        "patch_id",
        "ids",
    ]

    for key in candidate_keys:

        if key not in data:
            continue

        values = data[key]

        if torch.is_tensor(values):

            values = values.tolist()

        values = list(
            values
        )

        if len(values) != expected_count:

            fail(
                f"{split_name}: patch ID count under key "
                f"'{key}' is {len(values):,}, expected "
                f"{expected_count:,}."
            )

        values = [
            str(value)
            for value in values
        ]

        return values

    return None


# ============================================================
# VOCABULARY VALIDATION
# ============================================================

def validate_vocabulary(
    vocab,
):
    if not isinstance(
        vocab,
        dict,
    ):

        fail(
            "Frozen vocabulary JSON must contain an object."
        )

    if "clc_vocabulary" not in vocab:

        fail(
            "Frozen vocabulary does not contain "
            "'clc_vocabulary'."
        )

    frozen_clc = sorted(
        int(value)
        for value in vocab[
            "clc_vocabulary"
        ]
    )

    if len(frozen_clc) != 32:

        fail(
            "Frozen vocabulary does not contain 32 classes."
        )

    if frozen_clc != EXPECTED_CLC:

        missing = sorted(
            set(EXPECTED_CLC)
            - set(frozen_clc)
        )

        extra = sorted(
            set(frozen_clc)
            - set(EXPECTED_CLC)
        )

        fail(
            "Frozen vocabulary does not match the audited "
            "32-class grouping.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    print(
        "Frozen CLC vocabulary: PASS"
    )

    print(
        f"CLC classes: {len(frozen_clc)}"
    )

    print(
        f"Semantic families: {len(FAMILY_NAMES)}"
    )

    return frozen_clc


# ============================================================
# BUILD CLC INDEX
# ============================================================

def build_clc_index(
    frozen_clc,
):
    return {
        clc: index
        for index, clc
        in enumerate(frozen_clc)
    }


# ============================================================
# AGGREGATE 32 -> 19
# ============================================================

def aggregate_family_targets(
    clc_targets,
    frozen_clc,
):
    """
    Convert:

        [N,32,H,W]

    into:

        [N,19,H,W]
    """

    n, channels, height, width = (
        clc_targets.shape
    )

    if channels != 32:

        fail(
            "Source target tensor must have "
            "32 channels."
        )

    clc_to_index = build_clc_index(
        frozen_clc
    )

    family_targets = torch.zeros(
        (
            n,
            len(FAMILY_NAMES),
            height,
            width,
        ),
        dtype=torch.float32,
    )

    for family_index, family in enumerate(
        FAMILY_NAMES
    ):

        clc_codes = FAMILY_TO_CLC[
            family
        ]

        clc_indices = [
            clc_to_index[
                clc
            ]
            for clc in clc_codes
        ]

        family_targets[
            :,
            family_index,
            :,
            :,
        ] = clc_targets[
            :,
            clc_indices,
            :,
            :,
        ].sum(
            dim=1
        )

    return family_targets


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(
    clc_targets,
    family_targets,
    frozen_clc,
    split_name,
):
    print()
    print(
        "=" * 70
    )

    print(
        f"{split_name.upper()} FAMILY TARGET VALIDATION"
    )

    print(
        "=" * 70
    )

    n, c, h, w = (
        clc_targets.shape
    )

    fn, fc, fh, fw = (
        family_targets.shape
    )

    print(
        f"CLC target shape:      "
        f"{tuple(clc_targets.shape)}"
    )

    print(
        f"Family target shape:   "
        f"{tuple(family_targets.shape)}"
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if c != 32:

        fail(
            f"{split_name}: expected 32 CLC channels, "
            f"found {c}."
        )

    if fc != 19:

        fail(
            f"{split_name}: expected 19 family channels, "
            f"found {fc}."
        )

    if (
        n != fn
        or h != fh
        or w != fw
    ):

        fail(
            f"{split_name}: source/output spatial shape "
            "mismatch."
        )

    # --------------------------------------------------------
    # Finite check
    # --------------------------------------------------------

    if not torch.isfinite(
        clc_targets
    ).all():

        fail(
            f"{split_name}: CLC targets contain "
            "NaN/Inf."
        )

    if not torch.isfinite(
        family_targets
    ).all():

        fail(
            f"{split_name}: family targets contain "
            "NaN/Inf."
        )

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    clc_min = float(
        clc_targets.min()
    )

    clc_max = float(
        clc_targets.max()
    )

    family_min = float(
        family_targets.min()
    )

    family_max = float(
        family_targets.max()
    )

    print(
        f"CLC target range:      "
        f"[{clc_min:.10f}, {clc_max:.10f}]"
    )

    print(
        f"Family target range:   "
        f"[{family_min:.10f}, {family_max:.10f}]"
    )

    tolerance = 1e-5

    if (
        clc_min < -tolerance
        or clc_max > 1.0 + tolerance
    ):

        fail(
            f"{split_name}: CLC targets fall outside "
            "[0,1]."
        )

    if (
        family_min < -tolerance
        or family_max > 1.0 + tolerance
    ):

        fail(
            f"{split_name}: family targets fall outside "
            "[0,1]."
        )

    # --------------------------------------------------------
    # Per-cell conservation
    #
    # Each heatmap cell represents an aggregate spatial
    # contribution. Summing all 32 CLC classes should equal
    # summing all 19 family classes.
    # --------------------------------------------------------

    clc_cell_mass = (
        clc_targets
        .double()
        .sum(
            dim=1
        )
    )

    family_cell_mass = (
        family_targets
        .double()
        .sum(
            dim=1
        )
    )

    cell_mass_error = (
        clc_cell_mass
        - family_cell_mass
    ).abs()

    max_cell_mass_error = float(
        cell_mass_error.max()
    )

    mean_cell_mass_error = float(
        cell_mass_error.mean()
    )

    print(
        f"Max cell-mass conservation error: "
        f"{max_cell_mass_error:.12e}"
    )

    print(
        f"Mean cell-mass conservation error: "
        f"{mean_cell_mass_error:.12e}"
    )

    if max_cell_mass_error > 1e-5:

        fail(
            f"{split_name}: cell-mass conservation "
            "failed."
        )

    # --------------------------------------------------------
    # Per-patch total mass conservation
    # --------------------------------------------------------

    clc_patch_mass = (
        clc_targets
        .double()
        .sum(
            dim=(1, 2, 3)
        )
    )

    family_patch_mass = (
        family_targets
        .double()
        .sum(
            dim=(1, 2, 3)
        )
    )

    patch_mass_error = (
        clc_patch_mass
        - family_patch_mass
    ).abs()

    max_patch_mass_error = float(
        patch_mass_error.max()
    )

    mean_patch_mass_error = float(
        patch_mass_error.mean()
    )

    print(
        f"Max patch-mass conservation error: "
        f"{max_patch_mass_error:.12e}"
    )

    print(
        f"Mean patch-mass conservation error: "
        f"{mean_patch_mass_error:.12e}"
    )

    if max_patch_mass_error > 1e-4:

        fail(
            f"{split_name}: patch-mass conservation "
            "failed."
        )

    # --------------------------------------------------------
    # Family-by-family source conservation
    # --------------------------------------------------------

    clc_to_index = build_clc_index(
        frozen_clc
    )

    family_area_errors = []

    for family_index, family in enumerate(
        FAMILY_NAMES
    ):

        source_indices = [
            clc_to_index[
                clc
            ]
            for clc
            in FAMILY_TO_CLC[
                family
            ]
        ]

        source_area = (
            clc_targets[
                :,
                source_indices,
                :,
                :,
            ]
            .double()
            .sum(
                dim=(1, 2, 3)
            )
        )

        family_area = (
            family_targets[
                :,
                family_index,
                :,
                :,
            ]
            .double()
            .sum(
                dim=(1, 2)
            )
        )

        error = (
            source_area
            - family_area
        ).abs()

        max_error = float(
            error.max()
        )

        mean_error = float(
            error.mean()
        )

        family_area_errors.append(
            (
                family,
                max_error,
                mean_error,
            )
        )

        print(
            f"{family:<72} "
            f"max={max_error:.12e} "
            f"mean={mean_error:.12e}"
        )

        if max_error > 1e-4:

            fail(
                f"{split_name}: area conservation "
                f"failed for family '{family}'."
            )

    # --------------------------------------------------------
    # Check number of active families
    # --------------------------------------------------------

    family_total_area = (
        family_targets
        .double()
        .sum(
            dim=(0, 2, 3)
        )
    )

    active_families = int(
        (
            family_total_area
            > 1e-10
        )
        .sum()
    )

    print()

    print(
        f"Active semantic families: "
        f"{active_families}/19"
    )

    # --------------------------------------------------------
    # Check all heatmap cells still sum to one
    #
    # The original 32-class targets represent the partition
    # of each spatial cell. After grouping the classes into
    # 19 families, the partition must remain complete.
    # --------------------------------------------------------

    expected_cell_mass = torch.ones_like(
        family_cell_mass
    )

    complete_cell_error = (
        family_cell_mass
        - expected_cell_mass
    ).abs()

    max_complete_error = float(
        complete_cell_error.max()
    )

    print(
        f"Max family cell-sum error vs 1: "
        f"{max_complete_error:.12e}"
    )

    if max_complete_error > 1e-4:

        fail(
            f"{split_name}: family cell sums are not "
            "approximately 1."
        )

    print()

    print(
        f"{split_name}: ALL FAMILY TARGET CHECKS PASSED"
    )

    return {
        "split": split_name,
        "num_examples": n,
        "source_shape": list(
            clc_targets.shape
        ),
        "family_shape": list(
            family_targets.shape
        ),
        "clc_min": clc_min,
        "clc_max": clc_max,
        "family_min": family_min,
        "family_max": family_max,
        "max_cell_mass_error": max_cell_mass_error,
        "mean_cell_mass_error": mean_cell_mass_error,
        "max_patch_mass_error": max_patch_mass_error,
        "mean_patch_mass_error": mean_patch_mass_error,
        "active_families": active_families,
        "max_complete_cell_error": max_complete_error,
        "family_area_errors": [
            {
                "family": family,
                "max_error": max_error,
                "mean_error": mean_error,
            }
            for (
                family,
                max_error,
                mean_error,
            )
            in family_area_errors
        ],
    }


# ============================================================
# DISTRIBUTION
# ============================================================

def build_distribution_rows(
    train_family_targets,
    val_family_targets,
):
    rows = []

    splits = [
        (
            "train",
            train_family_targets,
        ),
        (
            "validation",
            val_family_targets,
        ),
    ]

    for split_name, tensor in splits:

        n = tensor.shape[0]

        total_area = float(
            tensor
            .double()
            .sum()
        )

        for family_index, family in enumerate(
            FAMILY_NAMES
        ):

            area = float(
                tensor[
                    :,
                    family_index,
                    :,
                    :,
                ]
                .double()
                .sum()
            )

            mean_area_per_patch = (
                area
                / n
            )

            fraction = (
                area
                / total_area
                if total_area > 0
                else 0.0
            )

            patch_presence = float(
                (
                    tensor[
                        :,
                        family_index,
                        :,
                        :,
                    ]
                    .double()
                    .sum(
                        dim=(1, 2)
                    )
                    > 1e-10
                )
                .float()
                .mean()
            )

            rows.append(
                {
                    "split":
                        split_name,

                    "family_index":
                        family_index,

                    "semantic_family":
                        family,

                    "clc_codes":
                        ",".join(
                            str(clc)
                            for clc
                            in FAMILY_TO_CLC[
                                family
                            ]
                        ),

                    "examples":
                        n,

                    "total_area":
                        area,

                    "mean_area_per_patch":
                        mean_area_per_patch,

                    "fraction_of_total_area":
                        fraction,

                    "patch_presence_fraction":
                        patch_presence,
                }
            )

    return rows


# ============================================================
# SAVE SPLIT
# ============================================================

def save_family_targets(
    path,
    family_targets,
    patch_ids,
    split_name,
):
    payload = {
        "targets":
            family_targets.float(),

        "patch_ids":
            patch_ids,

        "family_names":
            FAMILY_NAMES,

        "family_to_clc":
            FAMILY_TO_CLC,

        "num_classes":
            len(FAMILY_NAMES),

        "spatial_shape":
            [
                int(
                    family_targets.shape[-2]
                ),
                int(
                    family_targets.shape[-1]
                ),
            ],

        "split":
            split_name,

        "source":
            "C3 verified 32-class CLC targets",

        "dtype":
            "float32",
    }

    torch.save(
        payload,
        path,
    )

    print(
        f"Saved {split_name} family targets:"
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
        "C3 32-CLASS CLC -> 19-CLASS FAMILY TARGET BUILD"
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
    # Check source files
    # --------------------------------------------------------

    required_paths = [
        TRAIN_TARGET_PATH,
        VAL_TARGET_PATH,
        VOCAB_PATH,
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

            fail(
                f"MISSING: {path}"
            )

    print()

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------

    print(
        "LOADING FROZEN CLC VOCABULARY"
    )

    print(
        "-" * 70
    )

    vocab = load_json(
        VOCAB_PATH
    )

    frozen_clc = validate_vocabulary(
        vocab
    )

    print()

    # --------------------------------------------------------
    # Load train
    # --------------------------------------------------------

    print(
        "LOADING TRAIN CLC TARGETS"
    )

    print(
        "-" * 70
    )

    train_data = load_pt(
        TRAIN_TARGET_PATH
    )

    train_clc = extract_target_tensor(
        train_data,
        TRAIN_TARGET_PATH,
    )

    train_patch_ids = extract_patch_ids(
        train_data,
        train_clc.shape[0],
        "train",
    )

    print(
        f"Train CLC target shape: "
        f"{tuple(train_clc.shape)}"
    )

    if train_patch_ids is not None:

        print(
            f"Train patch IDs: "
            f"{len(train_patch_ids):,}"
        )

    else:

        print(
            "Train patch IDs: not stored in source file"
        )

    print()

    # --------------------------------------------------------
    # Load validation
    # --------------------------------------------------------

    print(
        "LOADING VALIDATION CLC TARGETS"
    )

    print(
        "-" * 70
    )

    val_data = load_pt(
        VAL_TARGET_PATH
    )

    val_clc = extract_target_tensor(
        val_data,
        VAL_TARGET_PATH,
    )

    val_patch_ids = extract_patch_ids(
        val_data,
        val_clc.shape[0],
        "validation",
    )

    print(
        f"Validation CLC target shape: "
        f"{tuple(val_clc.shape)}"
    )

    if val_patch_ids is not None:

        print(
            f"Validation patch IDs: "
            f"{len(val_patch_ids):,}"
        )

    else:

        print(
            "Validation patch IDs: "
            "not stored in source file"
        )

    print()

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "AGGREGATING 32 CLC CLASSES -> 19 FAMILIES"
    )

    print(
        "=" * 70
    )

    train_family_targets = aggregate_family_targets(
        train_clc,
        frozen_clc,
    )

    val_family_targets = aggregate_family_targets(
        val_clc,
        frozen_clc,
    )

    print(
        f"Train family shape: "
        f"{tuple(train_family_targets.shape)}"
    )

    print(
        f"Validation family shape: "
        f"{tuple(val_family_targets.shape)}"
    )

    print()

    # --------------------------------------------------------
    # Validate train
    # --------------------------------------------------------

    train_report = validate_split(
        train_clc,
        train_family_targets,
        frozen_clc,
        "train",
    )

    # --------------------------------------------------------
    # Validate validation
    # --------------------------------------------------------

    val_report = validate_split(
        val_clc,
        val_family_targets,
        frozen_clc,
        "validation",
    )

    # --------------------------------------------------------
    # Build distribution
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING FAMILY DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    distribution_rows = build_distribution_rows(
        train_family_targets,
        val_family_targets,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SAVING FAMILY TARGETS"
    )

    print(
        "=" * 70
    )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_family_targets(
        TRAIN_OUTPUT_PATH,
        train_family_targets,
        train_patch_ids,
        "train",
    )

    save_family_targets(
        VAL_OUTPUT_PATH,
        val_family_targets,
        val_patch_ids,
        "validation",
    )

    # --------------------------------------------------------
    # Save distribution CSV
    # --------------------------------------------------------

    with open(
        DISTRIBUTION_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "family_index",
                "semantic_family",
                "clc_codes",
                "examples",
                "total_area",
                "mean_area_per_patch",
                "fraction_of_total_area",
                "patch_presence_fraction",
            ],
        )

        writer.writeheader()

        writer.writerows(
            distribution_rows
        )

    print(
        "Saved family distribution:"
    )

    print(
        f"  {DISTRIBUTION_CSV_PATH}"
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = {

        "project":
            "SatQueryAI",

        "stage":
            "C3 semantic family target aggregation",

        "source_target_representation":
            {
                "classes":
                    32,

                "spatial_shape":
                    [
                        int(
                            train_clc.shape[-2]
                        ),
                        int(
                            train_clc.shape[-1]
                        ),
                    ],

                "train_examples":
                    int(
                        train_clc.shape[0]
                    ),

                "validation_examples":
                    int(
                        val_clc.shape[0]
                    ),
            },

        "output_target_representation":
            {
                "classes":
                    19,

                "family_names":
                    FAMILY_NAMES,

                "spatial_shape":
                    [
                        int(
                            train_family_targets.shape[-2]
                        ),
                        int(
                            train_family_targets.shape[-1]
                        ),
                    ],

                "train_examples":
                    int(
                        train_family_targets.shape[0]
                    ),

                "validation_examples":
                    int(
                        val_family_targets.shape[0]
                    ),

                "dtype":
                    "float32",
            },

        "family_to_clc":
            FAMILY_TO_CLC,

        "clc_vocabulary":
            frozen_clc,

        "source_files":
            {
                "train":
                    str(
                        TRAIN_TARGET_PATH
                    ),

                "validation":
                    str(
                        VAL_TARGET_PATH
                    ),

                "vocabulary":
                    str(
                        VOCAB_PATH
                    ),
            },

        "output_files":
            {
                "train":
                    str(
                        TRAIN_OUTPUT_PATH
                    ),

                "validation":
                    str(
                        VAL_OUTPUT_PATH
                    ),

                "distribution":
                    str(
                        DISTRIBUTION_CSV_PATH
                    ),

                "manifest":
                    str(
                        MANIFEST_OUTPUT_PATH
                    ),
            },

        "train_validation_patch_id_status":
            {
                "train_patch_ids_available":
                    train_patch_ids is not None,

                "validation_patch_ids_available":
                    val_patch_ids is not None,

                "train_validation_overlap_checked":
                    False,
            },

        "verification":
            {
                "train":
                    train_report,

                "validation":
                    val_report,
            },

        "aggregation_formula":
            "family_k(x,y) = sum of constituent CLC classes",
    }

    with open(
        MANIFEST_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print()
    print(
        "Saved family target manifest:"
    )

    print(
        f"  {MANIFEST_OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "C3 FAMILY TARGET BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Train:      "
        f"{tuple(train_family_targets.shape)}"
    )

    print(
        f"Validation: "
        f"{tuple(val_family_targets.shape)}"
    )

    print()

    print(
        "32 CLC classes have been aggregated into "
        "19 semantic families."
    )

    print(
        "All train/validation conservation checks passed."
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
        f"  {DISTRIBUTION_CSV_PATH}"
    )

    print(
        f"  {MANIFEST_OUTPUT_PATH}"
    )

    print()

    print(
        "Original 32-class targets were not modified."
    )


if __name__ == "__main__":
    main()