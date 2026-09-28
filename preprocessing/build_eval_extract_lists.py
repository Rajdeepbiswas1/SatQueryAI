from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\SatQueryAI")

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "evaluation_manifest.csv"
)

METADATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BigEarthNet_LMDB"
    / "metadata.parquet"
)

S1_LIST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "s1_eval_extract_list.txt"
)

S2_LIST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "s2_eval_extract_list.txt"
)

S2_BANDS = [
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
]


print("=== Building evaluation extraction lists ===")
print()

manifest = pd.read_csv(
    MANIFEST,
    dtype={"patch_id": str},
)

metadata = pd.read_parquet(
    METADATA,
    columns=[
        "patch_id",
        "s1_name",
        "s2v1_name",
    ],
)

metadata["patch_id"] = metadata["patch_id"].astype(str)

pairs = manifest.merge(
    metadata,
    on="patch_id",
    how="left",
)

if pairs["s1_name"].isna().any():
    missing = pairs.loc[
        pairs["s1_name"].isna(),
        "patch_id",
    ].tolist()

    raise RuntimeError(
        f"Missing S1 metadata for {len(missing):,} patches."
    )

if pairs["s2v1_name"].isna().any():
    missing = pairs.loc[
        pairs["s2v1_name"].isna(),
        "patch_id",
    ].tolist()

    raise RuntimeError(
        f"Missing S2 metadata for {len(missing):,} patches."
    )


s1_members = []
s2_members = []

for row in pairs.itertuples(index=False):

    patch_id = row.patch_id
    s1_name = row.s1_name
    s2_name = row.patch_id

    # Sentinel-1 archive member paths.
    s1_parts = s1_name.split("_")

    if len(s1_parts) < 3:
        raise RuntimeError(
            f"Unexpected S1 name: {s1_name}"
        )

    s1_acquisition = "_".join(
        s1_parts[:-3]
    )

    s1_base = (
        "BigEarthNet-S1/"
        f"{s1_acquisition}/"
        f"{s1_name}/"
    )

    s1_members.extend(
        [
            s1_base + f"{s1_name}_VH.tif",
            s1_base + f"{s1_name}_VV.tif",
        ]
    )

    # Sentinel-2 archive member paths.
    s2_parts = s2_name.split("_")

    if len(s2_parts) < 3:
        raise RuntimeError(
            f"Unexpected S2 name: {s2_name}"
        )

    s2_acquisition = "_".join(
        s2_parts[:-2]
    )

    s2_base = (
        "BigEarthNet-S2/"
        f"{s2_acquisition}/"
        f"{s2_name}/"
    )

    for band in S2_BANDS:
        s2_members.append(
            s2_base
            + f"{patch_id}_{band}.tif"
        )


# Remove duplicates while preserving order.
s1_members = list(dict.fromkeys(s1_members))
s2_members = list(dict.fromkeys(s2_members))


S1_LIST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

S1_LIST.write_text(
    "\n".join(s1_members) + "\n",
    encoding="utf-8",
)

S2_LIST.write_text(
    "\n".join(s2_members) + "\n",
    encoding="utf-8",
)


print(f"Evaluation pairs: {len(pairs):,}")
print(f"S1 files required: {len(s1_members):,}")
print(f"S2 files required: {len(s2_members):,}")
print()
print(f"S1 list: {S1_LIST}")
print(f"S2 list: {S2_LIST}")
print()
print("Expected:")
print("S1 = 3,082 × 2 = 6,164 files")
print("S2 = 3,082 × 12 = 36,984 files")


if len(s1_members) != 3_082 * 2:
    raise RuntimeError(
        "Unexpected S1 extraction-list size."
    )

if len(s2_members) != 3_082 * 12:
    raise RuntimeError(
        "Unexpected S2 extraction-list size."
    )

print()
print("SUCCESS: extraction lists are ready.")