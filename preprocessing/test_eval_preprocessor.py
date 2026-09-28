from pathlib import Path
import importlib.util
import pandas as pd
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

SCRIPT_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "preprocess_eval_dataset.py"
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "preprocess_eval_dataset",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load evaluation "
            "preprocessor."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


preprocessor = load_module(
    SCRIPT_PATH
)

manifest = pd.read_csv(
    PROJECT_ROOT
    / "data"
    / "splits"
    / "evaluation_manifest.csv"
)

row = manifest.iloc[0]

PATCH_ID = str(
    row["patch_id"]
)

REFERENCE_SPLIT = str(
    row["dataset_split"]
)

metadata = pd.read_parquet(
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BigEarthNet_LMDB"
    / "metadata.parquet",
    columns=[
        "patch_id",
        "s1_name",
    ],
)

metadata["patch_id"] = (
    metadata["patch_id"]
    .astype(str)
)

match = metadata[
    metadata["patch_id"]
    == PATCH_ID
]

if len(match) != 1:
    raise RuntimeError(
        f"Expected exactly one metadata "
        f"row for {PATCH_ID}, found "
        f"{len(match)}."
    )

S1_NAME = str(
    match.iloc[0]["s1_name"]
)

print("Processing one evaluation pair...")
print(
    f"Patch: {PATCH_ID}"
)
print(
    f"Split: {REFERENCE_SPLIT}"
)
print(
    f"S1:    {S1_NAME}"
)
print()

tensor = preprocessor.process_pair(
    PATCH_ID,
    S1_NAME,
)

print("=== Result ===")
print(
    "Shape:",
    tuple(tensor.shape),
)
print(
    "Dtype:",
    tensor.dtype,
)
print(
    "Min:",
    tensor.min().item(),
)
print(
    "Max:",
    tensor.max().item(),
)
print(
    "Finite:",
    bool(
        torch.isfinite(tensor).all()
    ),
)

if (
    tuple(tensor.shape)
    == (14, 120, 120)
    and tensor.dtype
    == torch.float32
    and torch.isfinite(tensor).all()
    and tensor.min() >= 0
    and tensor.max() <= 1
):
    print()
    print(
        "SUCCESS: evaluation "
        "preprocessing test passed."
    )
else:
    print()
    print(
        "WARNING: evaluation "
        "preprocessing test failed."
    )