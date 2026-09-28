import time
from pathlib import Path

import torch

from agent.dofa_extractor import SatQueryAIDofaExtractor


PROJECT_ROOT = Path(r"D:\SatQueryAI")

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
    / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90.pt"
)


print("=" * 80)
print("SATQUERYAI — CPU DOFA INFERENCE BENCHMARK")
print("=" * 80)

print()
print("Input:")
print(INPUT_PATH)

if not INPUT_PATH.is_file():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}"
    )

# ------------------------------------------------------------------
# Load one already-preprocessed validated sample.
# ------------------------------------------------------------------

data = torch.load(
    INPUT_PATH,
    map_location="cpu",
    weights_only=False,
)

image = data["image"]

print()
print("Patch ID:")
print(data["patch_id"])

print("Input shape:")
print(tuple(image.shape))

print("Input dtype:")
print(image.dtype)

print("Input finite:")
print(torch.isfinite(image).all().item())

if tuple(image.shape) != (14, 120, 120):
    raise RuntimeError(
        f"Unexpected input shape: {tuple(image.shape)}"
    )

# ------------------------------------------------------------------
# Load frozen DOFA.
# ------------------------------------------------------------------

print()
print("Loading frozen DOFA...")

load_start = time.perf_counter()

extractor = SatQueryAIDofaExtractor(
    device="cpu"
)

load_time = time.perf_counter() - load_start

print(
    f"DOFA load time: {load_time:.3f} seconds"
)

# ------------------------------------------------------------------
# First extraction.
# ------------------------------------------------------------------

print()
print("Running first CPU extraction...")

start = time.perf_counter()

s1, s2 = extractor.extract(
    image
)

first_time = time.perf_counter() - start

print(
    f"First extraction time: {first_time:.3f} seconds"
)

print("S1 shape:")
print(tuple(s1.shape))

print("S2 shape:")
print(tuple(s2.shape))

# ------------------------------------------------------------------
# Second extraction to measure warm inference.
# ------------------------------------------------------------------

print()
print("Running second CPU extraction...")

start = time.perf_counter()

s1_second, s2_second = extractor.extract(
    image
)

second_time = time.perf_counter() - start

print(
    f"Second extraction time: {second_time:.3f} seconds"
)

# ------------------------------------------------------------------
# Validate outputs.
# ------------------------------------------------------------------

expected_shape = (
    1,
    196,
    768,
)

if tuple(s1.shape) != expected_shape:
    raise RuntimeError(
        f"Unexpected S1 shape: {tuple(s1.shape)}"
    )

if tuple(s2.shape) != expected_shape:
    raise RuntimeError(
        f"Unexpected S2 shape: {tuple(s2.shape)}"
    )

if not torch.isfinite(s1).all().item():
    raise RuntimeError(
        "S1 contains NaN or Inf."
    )

if not torch.isfinite(s2).all().item():
    raise RuntimeError(
        "S2 contains NaN or Inf."
    )

# Check deterministic inference on the same input.
s1_difference = torch.max(
    torch.abs(s1 - s1_second)
).item()

s2_difference = torch.max(
    torch.abs(s2 - s2_second)
).item()

print()
print("Maximum S1 difference between runs:")
print(s1_difference)

print("Maximum S2 difference between runs:")
print(s2_difference)

print()
print("=" * 80)
print("DOFA CPU BENCHMARK COMPLETE")
print("=" * 80)