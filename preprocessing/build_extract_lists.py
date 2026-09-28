from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\SatQueryAI")

SELECTED_PAIRS = PROJECT_ROOT / "data" / "splits" / "selected_pairs.csv"
METADATA = PROJECT_ROOT / "data" / "raw" / "BigEarthNet_LMDB" / "metadata.parquet"

S1_LIST = PROJECT_ROOT / "data" / "raw" / "s1_extract_list.txt"
S2_LIST = PROJECT_ROOT / "data" / "raw" / "s2_extract_list.txt"

selected = pd.read_csv(SELECTED_PAIRS)

metadata = pd.read_parquet(
    METADATA,
    columns=["patch_id", "s1_name", "s2v1_name"],
)

pairs = selected.merge(
    metadata,
    on="patch_id",
    how="inner",
)

print(f"Selected pairs: {len(selected):,}")
print(f"Matched pairs:  {len(pairs):,}")

if len(pairs) != len(selected):
    raise RuntimeError("Some selected patch IDs were not found in metadata.")

s1_members = []
s2_members = []

for row in pairs.itertuples(index=False):

    # -----------------------------
    # Sentinel-1
    # -----------------------------
    s1_name = row.s1_name

    s1_parts = s1_name.split("_")
    s1_acquisition = "_".join(s1_parts[:-3])

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
    patch_parts = patch_id.split("_")
    s2_acquisition = "_".join(patch_parts[:-2])

    s2_base = (
        f"BigEarthNet-S2/"
        f"{s2_acquisition}/"
        f"{patch_id}"
    )

    for band in [
        "B01", "B02", "B03", "B04",
        "B05", "B06", "B07", "B08",
        "B09", "B11", "B12", "B8A",
    ]:
        s2_members.append(
            f"{s2_base}/{patch_id}_{band}.tif"
        )

# Remove duplicates while preserving order.
s1_members = list(dict.fromkeys(s1_members))
s2_members = list(dict.fromkeys(s2_members))

S1_LIST.write_text(
    "\n".join(s1_members) + "\n",
    encoding="utf-8",
)

S2_LIST.write_text(
    "\n".join(s2_members) + "\n",
    encoding="utf-8",
)

print(f"S1 archive members: {len(s1_members):,}")
print(f"S2 archive members: {len(s2_members):,}")
print(f"\nS1 list written to: {S1_LIST}")
print(f"S2 list written to: {S2_LIST}")

print("\nFirst 2 S1 entries:")
for x in s1_members[:2]:
    print(x)

print("\nFirst 12 S2 entries:")
for x in s2_members[:12]:
    print(x)