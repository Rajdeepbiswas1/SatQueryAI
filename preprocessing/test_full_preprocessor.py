from pathlib import Path
import importlib.util
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

SCRIPT_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "preprocess_dataset.py"
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "preprocess_dataset",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load preprocessing script.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


preprocessor = load_module(SCRIPT_PATH)


PATCH_ID = "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"

S1_NAME = (
    "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_test"
    / f"{PATCH_ID}.pt"
)


print("Processing one pair...")
print(f"Patch: {PATCH_ID}")
print()

tensor_new = preprocessor.process_pair(
    PATCH_ID,
    S1_NAME,
)

reference = torch.load(
    REFERENCE_FILE,
    map_location="cpu",
)

tensor_old = reference["image"]

print("=== Comparison ===")
print("Old shape:", tuple(tensor_old.shape))
print("New shape:", tuple(tensor_new.shape))

print("Old dtype:", tensor_old.dtype)
print("New dtype:", tensor_new.dtype)

print(
    "Maximum absolute difference:",
    torch.max(torch.abs(tensor_old - tensor_new)).item(),
)

print(
    "Mean absolute difference:",
    torch.mean(torch.abs(tensor_old - tensor_new)).item(),
)

print(
    "Old min/max:",
    tensor_old.min().item(),
    tensor_old.max().item(),
)

print(
    "New min/max:",
    tensor_new.min().item(),
    tensor_new.max().item(),
)

if (
    tensor_old.shape == tensor_new.shape
    and torch.allclose(
        tensor_old,
        tensor_new,
        atol=1e-5,
        rtol=1e-5,
    )
):
    print("\nSUCCESS: The preprocessing implementations match.")
else:
    print("\nWARNING: The preprocessing outputs differ.")