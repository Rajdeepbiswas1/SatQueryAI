from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\SatQueryAI\data\processed")

TRAIN = ROOT / "reference_grounding_split" / "reference_train.parquet"
VAL = ROOT / "reference_grounding_split" / "reference_val.parquet"

S1 = ROOT / "images_raw" / "BigEarthNet-S1"
S2 = ROOT / "images_raw" / "BigEarthNet-S2"

print("=" * 70)
print("REFERENCE GROUNDING → BIGEARTHNET IMAGE MAPPING")
print("=" * 70)

# ----------------------------------------------------------------------
# Load reference split
# ----------------------------------------------------------------------

train = pd.read_parquet(TRAIN)
val = pd.read_parquet(VAL)

df = pd.concat([train, val], ignore_index=True)

print("\nTRAIN rows:", len(train))
print("VAL rows:", len(val))
print("TOTAL rows:", len(df))

print("\nParquet columns:")
for col in df.columns:
    print(" ", col)

# ----------------------------------------------------------------------
# Use patch_id, NOT ID
# ----------------------------------------------------------------------

if "patch_id" not in df.columns:
    raise RuntimeError(
        "patch_id column was not found in the reference split."
    )

patch_ids = set(
    df["patch_id"]
    .dropna()
    .astype(str)
)

print("\nUnique patch IDs:", len(patch_ids))

print("\nFirst 10 patch IDs:")
for pid in sorted(patch_ids)[:10]:
    print(" ", pid)

# ----------------------------------------------------------------------
# Inspect actual image files
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("S1 IMAGE FILES")
print("-" * 70)

s1_files = [
    p for p in S1.rglob("*")
    if p.is_file()
]

print("S1 files:", len(s1_files))

for p in s1_files[:10]:
    print(" ", p)

print("\n" + "-" * 70)
print("S2 IMAGE FILES")
print("-" * 70)

s2_files = [
    p for p in S2.rglob("*")
    if p.is_file()
]

print("S2 files:", len(s2_files))

for p in s2_files[:10]:
    print(" ", p)

# ----------------------------------------------------------------------
# Build identifiers found in paths
#
# We check:
#   - filename stem
#   - filename
#   - every parent directory name
#
# This avoids making assumptions about the exact directory layout.
# ----------------------------------------------------------------------

def build_path_identifiers(files):
    identifiers = {}

    for p in files:
        candidates = set()

        candidates.add(p.name)
        candidates.add(p.stem)

        for parent in p.parents:
            if parent.name:
                candidates.add(parent.name)

        for candidate in candidates:
            identifiers.setdefault(candidate, p)

    return identifiers


print("\nBuilding S1 identifier index...")
s1_index = build_path_identifiers(s1_files)

print("Building S2 identifier index...")
s2_index = build_path_identifiers(s2_files)

# ----------------------------------------------------------------------
# Exact matching
# ----------------------------------------------------------------------

s1_matches = {}
s2_matches = {}

for pid in patch_ids:

    if pid in s1_index:
        s1_matches[pid] = s1_index[pid]

    if pid in s2_index:
        s2_matches[pid] = s2_index[pid]

print("\n" + "=" * 70)
print("EXACT PATCH-ID MATCH RESULTS")
print("=" * 70)

print(f"\nReference unique patches: {len(patch_ids):,}")
print(f"S1 matches:               {len(s1_matches):,}")
print(f"S2 matches:               {len(s2_matches):,}")

both = set(s1_matches) & set(s2_matches)
only_s1 = set(s1_matches) - set(s2_matches)
only_s2 = set(s2_matches) - set(s1_matches)
missing = patch_ids - set(s1_matches) - set(s2_matches)

print(f"\nBoth S1 + S2:             {len(both):,}")
print(f"S1 only:                  {len(only_s1):,}")
print(f"S2 only:                  {len(only_s2):,}")
print(f"Missing from both:        {len(missing):,}")

# ----------------------------------------------------------------------
# Show examples
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("MATCH EXAMPLES")
print("-" * 70)

for pid in sorted(both)[:10]:
    print("\nPatch ID:")
    print(" ", pid)

    print("S1:")
    print(" ", s1_matches[pid])

    print("S2:")
    print(" ", s2_matches[pid])

print("\n" + "-" * 70)
print("MISSING EXAMPLES")
print("-" * 70)

for pid in sorted(missing)[:20]:
    print(" ", pid)

print("\n" + "=" * 70)
print("MAPPING CHECK COMPLETE")
print("=" * 70)