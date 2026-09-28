from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(r"D:\SatQueryAI\data\processed")

TRAIN = ROOT / "reference_grounding_split" / "reference_train.parquet"
VAL = ROOT / "reference_grounding_split" / "reference_val.parquet"

S1_ROOT = ROOT / "images_raw" / "BigEarthNet-S1"
S2_ROOT = ROOT / "images_raw" / "BigEarthNet-S2"

OUT = ROOT / "grounding_images"

S1_OUT = OUT / "BigEarthNet-S1"
S2_OUT = OUT / "BigEarthNet-S2"

print("=" * 70)
print("SATQUERYAI — CREATE GROUNDING IMAGE SUBSET")
print("=" * 70)

# ---------------------------------------------------------------------
# Load exact reference split
# ---------------------------------------------------------------------

train = pd.read_parquet(TRAIN)
val = pd.read_parquet(VAL)

df = pd.concat([train, val], ignore_index=True)

patch_map = (
    df[["patch_id", "s1_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"\nUnique patch pairs: {len(patch_map):,}")

# ---------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------

if OUT.exists():
    print(f"\nOutput directory already exists:")
    print(OUT)

    answer = input(
        "\nDelete and recreate it? [y/N]: "
    ).strip().lower()

    if answer != "y":
        print("\nAborted. Existing files were not modified.")
        raise SystemExit

    print("\nDeleting existing staging directory...")
    shutil.rmtree(OUT)

S1_OUT.mkdir(parents=True, exist_ok=True)
S2_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Build source directory indexes
# ---------------------------------------------------------------------

print("\nIndexing S1 directories...")
s1_dirs = {
    p.name: p
    for p in S1_ROOT.rglob("*")
    if p.is_dir()
}

print("Indexing S2 directories...")
s2_dirs = {
    p.name: p
    for p in S2_ROOT.rglob("*")
    if p.is_dir()
}

# ---------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------

copied_s1_files = 0
copied_s2_files = 0

copied_s1_bytes = 0
copied_s2_bytes = 0

missing_s1 = []
missing_s2 = []

for i, row in patch_map.iterrows():

    patch_id = str(row["patch_id"])
    s1_name = str(row["s1_name"])

    # ================================================================
    # S2
    # ================================================================

    s2_src = s2_dirs.get(patch_id)

    if s2_src is None:
        missing_s2.append(patch_id)
    else:
        s2_dst = S2_OUT / patch_id
        s2_dst.mkdir(parents=True, exist_ok=True)

        for src in s2_src.iterdir():

            if not src.is_file():
                continue

            if src.suffix.lower() != ".tif":
                continue

            dst = s2_dst / src.name

            shutil.copy2(src, dst)

            size = dst.stat().st_size

            copied_s2_files += 1
            copied_s2_bytes += size

    # ================================================================
    # S1
    # ================================================================

    s1_src = s1_dirs.get(s1_name)

    if s1_src is None:
        missing_s1.append(s1_name)
    else:
        s1_dst = S1_OUT / s1_name
        s1_dst.mkdir(parents=True, exist_ok=True)

        for src in s1_src.iterdir():

            if not src.is_file():
                continue

            if src.suffix.lower() != ".tif":
                continue

            dst = s1_dst / src.name

            shutil.copy2(src, dst)

            size = dst.stat().st_size

            copied_s1_files += 1
            copied_s1_bytes += size

    # Progress every 250 patches
    if (i + 1) % 250 == 0 or (i + 1) == len(patch_map):
        print(
            f"Processed {i + 1:,}/{len(patch_map):,} patches"
        )

# ---------------------------------------------------------------------
# Copy manifests too
# ---------------------------------------------------------------------

print("\nCopying reference manifests...")

shutil.copy2(
    TRAIN,
    OUT / "reference_train.parquet"
)

shutil.copy2(
    VAL,
    OUT / "reference_val.parquet"
)

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

total_files = copied_s1_files + copied_s2_files
total_bytes = copied_s1_bytes + copied_s2_bytes

print("\n" + "=" * 70)
print("SUBSET CREATION COMPLETE")
print("=" * 70)

print(f"\nPatch pairs:")
print(f"  {len(patch_map):,}")

print("\nS1:")
print(f"  Files: {copied_s1_files:,}")
print(
    f"  Size:  {copied_s1_bytes / (1024 ** 3):.3f} GiB"
)
print(f"  Missing: {len(missing_s1):,}")

print("\nS2:")
print(f"  Files: {copied_s2_files:,}")
print(
    f"  Size:  {copied_s2_bytes / (1024 ** 3):.3f} GiB"
)
print(f"  Missing: {len(missing_s2):,}")

print("\nTOTAL:")
print(f"  Files: {total_files:,}")
print(
    f"  Size:  {total_bytes / (1024 ** 3):.3f} GiB"
)

print("\nOutput:")
print(f"  {OUT}")

# ---------------------------------------------------------------------
# Final safety assertions
# ---------------------------------------------------------------------

assert len(missing_s1) == 0, (
    f"Missing {len(missing_s1)} S1 patches."
)

assert len(missing_s2) == 0, (
    f"Missing {len(missing_s2)} S2 patches."
)

assert copied_s1_files == len(patch_map) * 2, (
    f"Expected {len(patch_map) * 2:,} S1 files, "
    f"found {copied_s1_files:,}."
)

assert copied_s2_files == len(patch_map) * 12, (
    f"Expected {len(patch_map) * 12:,} S2 files, "
    f"found {copied_s2_files:,}."
)

print("\nAll file-count safety checks passed.")
print("=" * 70)