from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\SatQueryAI")

BEN_TXT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BigEarthNet.txt.parquet"
)

METADATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BigEarthNet_LMDB"
    / "metadata.parquet"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "selected_pairs.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "evaluation_manifest.csv"
)

SEED = 42
VALIDATION_TARGET = 1000
TEST_TARGET = 1000


print("=== SatQueryAI evaluation-set selection ===")
print()

# --------------------------------------------------
# Existing training patches
# --------------------------------------------------

train = pd.read_csv(
    TRAIN_FILE,
    usecols=["patch_id"],
)

train_ids = set(
    train["patch_id"].astype(str)
)

print(
    f"Existing training patches: {len(train_ids):,}"
)

# --------------------------------------------------
# BigEarthNet.txt split information
# --------------------------------------------------

df = pd.read_parquet(
    BEN_TXT,
    columns=[
        "patch_id",
        "split",
    ],
)

df["patch_id"] = df["patch_id"].astype(str)
df["split"] = df["split"].astype(str)

# One row per image pair.
df = df.drop_duplicates(
    subset=["patch_id", "split"]
)

benchmark = (
    df[df["split"] == "bench"]
    [["patch_id"]]
    .drop_duplicates()
)

validation = (
    df[df["split"] == "validation"]
    [["patch_id"]]
    .drop_duplicates()
)

test = (
    df[df["split"] == "test"]
    [["patch_id"]]
    .drop_duplicates()
)

print(
    f"Available validation patches: {len(validation):,}"
)

print(
    f"Available test patches:       {len(test):,}"
)

print(
    f"Available benchmark patches:   {len(benchmark):,}"
)

# --------------------------------------------------
# Metadata for stratified selection
# --------------------------------------------------

metadata = pd.read_parquet(
    BEN_TXT,
    columns=[
        "patch_id",
        "country",
        "season",
        "climate_zone",
    ],
)

metadata["patch_id"] = (
    metadata["patch_id"]
    .astype(str)
)

metadata = metadata.drop_duplicates(
    subset=["patch_id"]
)


def stratified_sample(
    candidates,
    target,
    excluded_ids,
    seed,
):
    """
    Select approximately proportionally across
    country / season / climate zone.
    """

    candidates = candidates[
        ~candidates["patch_id"].isin(excluded_ids)
    ].copy()

    if len(candidates) < target:
        raise RuntimeError(
            f"Only {len(candidates):,} eligible patches "
            f"available, but {target:,} required."
        )

    candidates = candidates.merge(
        metadata,
        on="patch_id",
        how="left",
    )

    candidates = candidates.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    strata = [
        "country",
        "season",
        "climate_zone",
    ]

    available_strata = [
        column
        for column in strata
        if column in candidates.columns
        and candidates[column].notna().any()
    ]

    if not available_strata:
        return candidates.sample(
            n=target,
            random_state=seed,
        )[["patch_id"]]

    keys = (
        candidates[available_strata]
        .fillna("UNKNOWN")
        .astype(str)
        .agg("|".join, axis=1)
    )

    counts = keys.value_counts()

    # Initial proportional quotas.
    quotas = (
        counts / counts.sum() * target
    ).astype(int)

    # Distribute remaining slots.
    remaining = (
        target
        - int(quotas.sum())
    )

    for key in counts.sort_values(
        ascending=False
    ).index:

        if remaining == 0:
            break

        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1

    rng = np.random.default_rng(seed)

    selected_parts = []

    for key, quota in quotas.items():

        if quota <= 0:
            continue

        indices = candidates.index[
            keys == key
        ]

        if len(indices) == 0:
            continue

        take = min(
            quota,
            len(indices),
        )

        random_state = int(
            rng.integers(
                0,
                2**31 - 1,
            )
        )

        selected = candidates.loc[
            indices
        ].sample(
            n=take,
            random_state=random_state,
        )

        selected_parts.append(
            selected
        )

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # Safety fill if rounding left us short.
    if len(result) < target:

        already_selected = set(
            result["patch_id"]
        )

        remaining_candidates = candidates[
            ~candidates["patch_id"].isin(
                already_selected
            )
        ]

        extra = remaining_candidates.sample(
            n=target - len(result),
            random_state=seed,
        )

        result = pd.concat(
            [
                result,
                extra,
            ],
            ignore_index=True,
        )

    return (
        result[
            ["patch_id"]
        ]
        .drop_duplicates()
        .head(target)
    )


# --------------------------------------------------
# Select validation
# --------------------------------------------------

used_ids = set(train_ids)

validation_selected = stratified_sample(
    validation,
    VALIDATION_TARGET,
    used_ids,
    SEED,
)

used_ids.update(
    validation_selected["patch_id"]
)

# --------------------------------------------------
# Select test
# --------------------------------------------------

test_selected = stratified_sample(
    test,
    TEST_TARGET,
    used_ids,
    SEED + 1,
)

used_ids.update(
    test_selected["patch_id"]
)

# --------------------------------------------------
# Use the entire official benchmark
# --------------------------------------------------

benchmark_selected = benchmark[
    ~benchmark["patch_id"].isin(
        train_ids
    )
].copy()

# --------------------------------------------------
# Build final manifest
# --------------------------------------------------

validation_selected[
    "dataset_split"
] = "validation"

test_selected[
    "dataset_split"
] = "test"

benchmark_selected[
    "dataset_split"
] = "benchmark"

manifest = pd.concat(
    [
        validation_selected[
            ["patch_id", "dataset_split"]
        ],
        test_selected[
            ["patch_id", "dataset_split"]
        ],
        benchmark_selected[
            ["patch_id", "dataset_split"]
        ],
    ],
    ignore_index=True,
)

# --------------------------------------------------
# Hard checks
# --------------------------------------------------

if manifest["patch_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate patch IDs found."
    )

overlap = (
    set(manifest["patch_id"])
    .intersection(train_ids)
)

if overlap:
    raise RuntimeError(
        f"{len(overlap):,} evaluation patches "
        "overlap with training."
    )

validation_count = (
    manifest["dataset_split"]
    == "validation"
).sum()

test_count = (
    manifest["dataset_split"]
    == "test"
).sum()

benchmark_count = (
    manifest["dataset_split"]
    == "benchmark"
).sum()

if validation_count != VALIDATION_TARGET:
    raise RuntimeError(
        "Incorrect validation count."
    )

if test_count != TEST_TARGET:
    raise RuntimeError(
        "Incorrect test count."
    )

if benchmark_count != len(benchmark_selected):
    raise RuntimeError(
        "Incorrect benchmark count."
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

manifest.to_csv(
    OUTPUT_FILE,
    index=False,
)

# --------------------------------------------------
# Report
# --------------------------------------------------

print()
print("=== Selection complete ===")
print(
    f"Training:      {len(train_ids):,}"
)
print(
    f"Validation:    {validation_count:,}"
)
print(
    f"Test:          {test_count:,}"
)
print(
    f"Benchmark:     {benchmark_count:,}"
)
print(
    f"Evaluation:    {len(manifest):,}"
)
print(
    f"Train overlap: {len(overlap):,}"
)
print()
print(
    f"Manifest saved to:\n{OUTPUT_FILE}"
)

print()
print("=== Split counts ===")
print(
    manifest[
        "dataset_split"
    ].value_counts().to_string()
)