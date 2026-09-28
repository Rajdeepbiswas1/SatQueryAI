from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\SatQueryAI\data\processed")

TRAIN = ROOT / "reference_grounding_split" / "reference_train.parquet"
VAL = ROOT / "reference_grounding_split" / "reference_val.parquet"

S1_ROOT = ROOT / "images_raw" / "BigEarthNet-S1"

print("=" * 70)
print("BIGEARTHNET S1 MAPPING VIA s1_name")
print("=" * 70)

train = pd.read_parquet(TRAIN)
val = pd.read_parquet(VAL)

df = pd.concat([train, val], ignore_index=True)

# One S1 name should correspond to each BigEarthNet patch.
mapping = (
    df[["patch_id", "s1_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"\nReference rows: {len(df):,}")
print(f"Unique S2 patches: {df['patch_id'].nunique():,}")
print(f"Unique patch/S1 pairs: {len(mapping):,}")
print(f"Unique S1 names: {mapping['s1_name'].nunique():,}")

print("\nFirst 15 mappings:")
print(mapping.head(15).to_string(index=False))

# ------------------------------------------------------------------
# Index actual S1 directories/files
# ------------------------------------------------------------------

print("\nIndexing S1 directories...")

s1_dirs = [
    p for p in S1_ROOT.rglob("*")
    if p.is_dir()
]

s1_files = [
    p for p in S1_ROOT.rglob("*.tif")
    if p.is_file()
]

print(f"S1 directories: {len(s1_dirs):,}")
print(f"S1 tif files:   {len(s1_files):,}")

# Directory names are normally the patch/acquisition identifiers.
dir_names = {p.name: p for p in s1_dirs}

file_names = {p.stem: p for p in s1_files}

# ------------------------------------------------------------------
# Match s1_name
# ------------------------------------------------------------------

matched_dir = 0
matched_file = 0
unmatched = []

matches = []

for _, row in mapping.iterrows():

    patch_id = str(row["patch_id"])
    s1_name = str(row["s1_name"])

    found = None

    # Exact directory name
    if s1_name in dir_names:
        found = dir_names[s1_name]
        matched_dir += 1

    # Exact file stem
    elif s1_name in file_names:
        found = file_names[s1_name]
        matched_file += 1

    # Sometimes the actual S1 patch directory is represented
    # without an extension but has the same stem.
    if found is None:
        unmatched.append((patch_id, s1_name))
    else:
        matches.append((patch_id, s1_name, found))

print("\n" + "=" * 70)
print("S1 MATCH RESULTS")
print("=" * 70)

print(f"\nUnique patch/S1 pairs: {len(mapping):,}")
print(f"Matched by directory:  {matched_dir:,}")
print(f"Matched by file stem:  {matched_file:,}")
print(f"Unmatched:             {len(unmatched):,}")

print("\n" + "-" * 70)
print("MATCH EXAMPLES")
print("-" * 70)

for patch_id, s1_name, path in matches[:10]:
    print("\nS2 patch:")
    print(" ", patch_id)

    print("s1_name:")
    print(" ", s1_name)

    print("Matched path:")
    print(" ", path)

print("\n" + "-" * 70)
print("UNMATCHED EXAMPLES")
print("-" * 70)

for patch_id, s1_name in unmatched[:20]:
    print("\nS2 patch:")
    print(" ", patch_id)
    print("s1_name:")
    print(" ", s1_name)

print("\n" + "=" * 70)
print("S1 MAPPING CHECK COMPLETE")
print("=" * 70)