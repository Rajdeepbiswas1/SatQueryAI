from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\SatQueryAI\data\processed")

TRAIN = ROOT / "reference_grounding_split" / "reference_train.parquet"
VAL = ROOT / "reference_grounding_split" / "reference_val.parquet"

S1_ROOT = ROOT / "images_raw" / "BigEarthNet-S1"
S2_ROOT = ROOT / "images_raw" / "BigEarthNet-S2"

print("=" * 70)
print("SATQUERYAI — GROUNDING IMAGE SUBSET SIZE CALCULATION")
print("=" * 70)

# ---------------------------------------------------------------------
# Load exact grounding patch list
# ---------------------------------------------------------------------

train = pd.read_parquet(TRAIN)
val = pd.read_parquet(VAL)

df = pd.concat([train, val], ignore_index=True)

patch_map = (
    df[["patch_id", "s1_name"]]
    .drop_duplicates()
)

print(f"\nUnique S2 patches: {patch_map['patch_id'].nunique():,}")
print(f"Unique S1 names:   {patch_map['s1_name'].nunique():,}")

# ---------------------------------------------------------------------
# Build S2 directory index
# ---------------------------------------------------------------------

print("\nIndexing S2 patch directories...")

s2_dirs = {
    p.name: p
    for p in S2_ROOT.rglob("*")
    if p.is_dir()
}

# ---------------------------------------------------------------------
# Build S1 directory index
# ---------------------------------------------------------------------

print("Indexing S1 patch directories...")

s1_dirs = {
    p.name: p
    for p in S1_ROOT.rglob("*")
    if p.is_dir()
}

# ---------------------------------------------------------------------
# Calculate required files and sizes
# ---------------------------------------------------------------------

required_s1 = []
required_s2 = []

missing_s1 = []
missing_s2 = []

s1_bytes = 0
s2_bytes = 0

for _, row in patch_map.iterrows():

    patch_id = str(row["patch_id"])
    s1_name = str(row["s1_name"])

    # -------------------------------------------------------------
    # S2
    # -------------------------------------------------------------

    s2_dir = s2_dirs.get(patch_id)

    if s2_dir is None:
        missing_s2.append(patch_id)
    else:
        files = [
            p for p in s2_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".tif"
        ]

        required_s2.extend(files)

        for p in files:
            s2_bytes += p.stat().st_size

    # -------------------------------------------------------------
    # S1
    # -------------------------------------------------------------

    s1_dir = s1_dirs.get(s1_name)

    if s1_dir is None:
        missing_s1.append(s1_name)
    else:
        files = [
            p for p in s1_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".tif"
        ]

        required_s1.extend(files)

        for p in files:
            s1_bytes += p.stat().st_size

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

total_bytes = s1_bytes + s2_bytes


def gib(n):
    return n / (1024 ** 3)


def gib_to_string(n):
    return f"{gib(n):.3f} GiB"


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nRequired patches:")
print(f"  S1/S2 patch pairs: {len(patch_map):,}")

print("\nS1:")
print(f"  Files:   {len(required_s1):,}")
print(f"  Size:    {gib_to_string(s1_bytes)}")
print(f"  Missing: {len(missing_s1):,}")

print("\nS2:")
print(f"  Files:   {len(required_s2):,}")
print(f"  Size:    {gib_to_string(s2_bytes)}")
print(f"  Missing: {len(missing_s2):,}")

print("\nTOTAL:")
print(f"  Files:   {len(required_s1) + len(required_s2):,}")
print(f"  Size:    {gib_to_string(total_bytes)}")

# ---------------------------------------------------------------------
# File-count sanity checks
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("FILE COUNT BY PATCH")
print("-" * 70)

if required_s1:
    print(
        "S1 files per patch:",
        len(required_s1) / len(patch_map)
    )

if required_s2:
    print(
        "S2 files per patch:",
        len(required_s2) / len(patch_map)
    )

# ---------------------------------------------------------------------
# Example patch contents
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("EXAMPLE PATCH CONTENTS")
print("-" * 70)

example = patch_map.iloc[0]

print("\nS2 patch:")
print(example["patch_id"])

s2_example = s2_dirs.get(str(example["patch_id"]))

if s2_example:
    for p in sorted(s2_example.iterdir()):
        if p.is_file():
            print(
                f"  {p.name:20s}"
                f" {p.stat().st_size / (1024 ** 2):8.2f} MB"
            )

print("\nS1 patch:")
print(example["s1_name"])

s1_example = s1_dirs.get(str(example["s1_name"]))

if s1_example:
    for p in sorted(s1_example.iterdir()):
        if p.is_file():
            print(
                f"  {p.name:20s}"
                f" {p.stat().st_size / (1024 ** 2):8.2f} MB"
            )

# ---------------------------------------------------------------------
# Missing examples
# ---------------------------------------------------------------------

if missing_s1:
    print("\nFirst missing S1 names:")
    for x in missing_s1[:10]:
        print(" ", x)

if missing_s2:
    print("\nFirst missing S2 patch IDs:")
    for x in missing_s2[:10]:
        print(" ", x)

print("\n" + "=" * 70)
print("SIZE CALCULATION COMPLETE")
print("=" * 70)