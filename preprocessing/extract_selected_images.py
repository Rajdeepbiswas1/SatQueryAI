from pathlib import Path
import subprocess
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(r"D:\SatQueryAI")

SELECTED_PAIRS = PROJECT_ROOT / "data" / "splits" / "selected_pairs.csv"
METADATA = PROJECT_ROOT / "data" / "raw" / "BigEarthNet_LMDB" / "metadata.parquet"

ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "BigEarthNet_v2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "images"

S1_ARCHIVE = ARCHIVE_DIR / "BigEarthNet-S1.tar.zst"
S2_ARCHIVE = ARCHIVE_DIR / "BigEarthNet-S2.tar.zst"

S1_LIST = PROJECT_ROOT / "data" / "raw" / "s1_extract_list.txt"
S2_LIST = PROJECT_ROOT / "data" / "raw" / "s2_extract_list.txt"


# ---------------------------------------------------------
# Load our 10,000 selected pairs
# ---------------------------------------------------------
selected = pd.read_csv(SELECTED_PAIRS)

metadata = pd.read_parquet(
    METADATA,
    columns=["patch_id", "s1_name", "s2v1_name"]
)

pairs = selected.merge(
    metadata,
    on="patch_id",
    how="inner"
)

print(f"Selected pairs: {len(selected):,}")
print(f"Matched metadata: {len(pairs):,}")

if len(pairs) != len(selected):
    raise RuntimeError(
        "Not all selected patch IDs were found in metadata."
    )


# ---------------------------------------------------------
# Build exact archive member paths
# ---------------------------------------------------------
s1_members = []
s2_members = []

for row in pairs.itertuples(index=False):

    # -----------------------------
    # Sentinel-1
    # -----------------------------
    s1_name = row.s1_name

    # Parent acquisition name:
    # remove final "_row_col"
    s1_parts = s1_name.split("_")
    s1_acquisition = "_".join(s1_parts[:-2])

    s1_base = (
        f"BigEarthNet-S1/"
        f"{s1_acquisition}/"
        f"{s1_name}/"
        f"{s1_name}"
    )

    s1_members.append(s1_base + "_VH.tif")
    s1_members.append(s1_base + "_VV.tif")

    # -----------------------------
    # Sentinel-2
    # -----------------------------
    patch_id = row.patch_id

    # Parent acquisition name:
    # remove final "_row_col"
    patch_parts = patch_id.split("_")
    s2_acquisition = "_".join(patch_parts[:-2])

    s2_name = row.s2v1_name

    s2_base = (
        f"BigEarthNet-S2/"
        f"{s2_acquisition}/"
        f"{s2_name}"
    )

    for band in [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B11",
        "B12",
        "B8A",
    ]:
        s2_members.append(
            f"{s2_base}/{s2_name}_{band}.tif"
        )


# Remove duplicates while preserving order
s1_members = list(dict.fromkeys(s1_members))
s2_members = list(dict.fromkeys(s2_members))


# ---------------------------------------------------------
# Write tar member lists
# ---------------------------------------------------------
S1_LIST.write_text(
    "\n".join(s1_members) + "\n",
    encoding="utf-8"
)

S2_LIST.write_text(
    "\n".join(s2_members) + "\n",
    encoding="utf-8"
)

print(f"S1 files requested: {len(s1_members):,}")
print(f"S2 files requested: {len(s2_members):,}")

print(f"S1 list: {S1_LIST}")
print(f"S2 list: {S2_LIST}")


# ---------------------------------------------------------
# Check that archives exist
# ---------------------------------------------------------
if not S1_ARCHIVE.exists():
    raise FileNotFoundError(S1_ARCHIVE)

if not S2_ARCHIVE.exists():
    raise FileNotFoundError(S2_ARCHIVE)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Extract S1
# ---------------------------------------------------------
print("\n=== Extracting Sentinel-1 ===")

subprocess.run(
    [
        "tar",
        "-xf",
        str(S1_ARCHIVE),
        "-C",
        str(OUTPUT_DIR),
        "-T",
        str(S1_LIST),
    ],
    check=True,
)

print("Sentinel-1 extraction finished.")


# ---------------------------------------------------------
# Extract S2
# ---------------------------------------------------------
print("\n=== Extracting Sentinel-2 ===")

subprocess.run(
    [
        "tar",
        "-xf",
        str(S2_ARCHIVE),
        "-C",
        str(OUTPUT_DIR),
        "-T",
        str(S2_LIST),
    ],
    check=True,
)

print("Sentinel-2 extraction finished.")


print("\nAll selected imagery has been extracted.")