from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

# ============================================================
# SATQUERYAI — POINT GROUNDING PATCH-DISJOINT SPLIT
# ============================================================

SEED = 42
TRAIN_FRACTION = 0.80

SOURCE = Path(
    r"D:\SatQueryAI\data\processed\training_annotations.parquet"
)

OUT = Path(
    r"D:\SatQueryAI\data\processed\point_grounding_split"
)

OUT.mkdir(parents=True, exist_ok=True)

POINT_RE = re.compile(
    r"<point>\(\s*"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))\s*,\s*"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))"
    r"\s*\)</point>",
    re.IGNORECASE,
)

BOX_RE = re.compile(
    r"\[\s*"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))\s+"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))\s*,\s*"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))\s+"
    r"([-+]?(?:\d*\.\d+|\d+\.?\d*))"
    r"\s*\]"
)


def parse_point(text):
    if not isinstance(text, str):
        return None

    m = POINT_RE.search(text)

    if m is None:
        return None

    return (
        float(m.group(1)),
        float(m.group(2)),
    )


def parse_box(text):
    if not isinstance(text, str):
        return None

    m = BOX_RE.fullmatch(text.strip())

    if m is None:
        return None

    return tuple(float(v) for v in m.groups())


print("=" * 80)
print("SATQUERYAI — POINT GROUNDING SPLIT")
print("=" * 80)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source file does not exist:\n{SOURCE}"
    )

df = pd.read_parquet(SOURCE)

print(f"\nOriginal rows: {len(df):,}")

required = {
    "type",
    "category",
    "input",
    "output",
    "patch_id",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

# ------------------------------------------------------------
# GROUNDING SUBSET
# ------------------------------------------------------------

grounding = df[
    df["type"].astype(str).str.lower() == "bounding box"
].copy()

grounding = grounding.reset_index(
    drop=False
)

grounding.rename(
    columns={"index": "annotation_index"},
    inplace=True
)

grounding["grounding_row_index"] = np.arange(
    len(grounding),
    dtype=np.int64
)

print(
    f"Grounding rows: {len(grounding):,}"
)

if len(grounding) != 45_883:
    raise RuntimeError(
        f"Expected 45,883 grounding rows, "
        f"found {len(grounding):,}"
    )

# ------------------------------------------------------------
# POINT-ONLY SUBSET
# ------------------------------------------------------------

point = grounding[
    grounding["category"].astype(str).str.lower() == "point"
].copy()

print(
    f"Point rows: {len(point):,}"
)

if len(point) != 23_778:
    raise RuntimeError(
        f"Expected 23,778 point rows, "
        f"found {len(point):,}"
    )

# ------------------------------------------------------------
# PARSE POINTS
# ------------------------------------------------------------

point["point_xy"] = point["input"].apply(
    parse_point
)

failed_point = point["point_xy"].isna().sum()

print(
    f"Point parse failures: {failed_point:,}"
)

if failed_point:
    print(
        point.loc[
            point["point_xy"].isna(),
            ["patch_id", "input"]
        ].head(10).to_string(index=False)
    )
    raise RuntimeError(
        "Point parsing failed."
    )

# ------------------------------------------------------------
# PARSE BOXES
# ------------------------------------------------------------

point["target_box"] = point["output"].apply(
    parse_box
)

failed_box = point["target_box"].isna().sum()

print(
    f"Box parse failures: {failed_box:,}"
)

if failed_box:
    print(
        point.loc[
            point["target_box"].isna(),
            ["patch_id", "output"]
        ].head(10).to_string(index=False)
    )
    raise RuntimeError(
        "Bounding-box parsing failed."
    )

# ------------------------------------------------------------
# RANGE CHECK
# ------------------------------------------------------------

points = np.asarray(
    point["point_xy"].tolist(),
    dtype=np.float64
)

px = points[:, 0]
py = points[:, 1]

bad_range = (
    (px < 0) | (px > 1) |
    (py < 0) | (py > 1)
)

print(
    f"Out-of-range points: {bad_range.sum():,}"
)

if bad_range.any():
    raise RuntimeError(
        "Invalid point coordinates detected."
    )

# ------------------------------------------------------------
# POINT-IN-BOX CHECK
# ------------------------------------------------------------

inside = np.zeros(
    len(point),
    dtype=bool
)

for i, box in enumerate(
    point["target_box"]
):
    x1, y1, x2, y2 = box
    x, y = point["point_xy"].iloc[i]

    inside[i] = (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )

print(
    f"Points inside target box: "
    f"{inside.sum():,}/{len(point):,}"
)

if not inside.all():
    bad = point.loc[
        ~inside,
        [
            "patch_id",
            "input",
            "point_xy",
            "target_box"
        ]
    ]

    print(
        bad.head(10).to_string(index=False)
    )

    raise RuntimeError(
        "Point-in-box validation failed."
    )

# ------------------------------------------------------------
# PATCH-DISJOINT SPLIT
# ------------------------------------------------------------

point["patch_id"] = (
    point["patch_id"]
    .astype(str)
)

patches = (
    point["patch_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(SEED)

rng.shuffle(patches)

n_train = int(
    len(patches) * TRAIN_FRACTION
)

train_patch_ids = set(
    patches[:n_train]
)

val_patch_ids = set(
    patches[n_train:]
)

train = point[
    point["patch_id"].isin(
        train_patch_ids
    )
].copy()

val = point[
    point["patch_id"].isin(
        val_patch_ids
    )
].copy()

# ------------------------------------------------------------
# LEAKAGE CHECK
# ------------------------------------------------------------

train_patches = set(
    train["patch_id"]
)

val_patches = set(
    val["patch_id"]
)

patch_overlap = (
    train_patches &
    val_patches
)

print("\n" + "=" * 80)
print("PATCH SPLIT")
print("=" * 80)

print(
    f"Unique patches: {len(patches):,}"
)

print(
    f"Train patches:  {len(train_patches):,}"
)

print(
    f"Val patches:    {len(val_patches):,}"
)

print(
    f"Patch overlap:  {len(patch_overlap):,}"
)

if patch_overlap:
    raise RuntimeError(
        "PATCH LEAKAGE DETECTED."
    )

# ------------------------------------------------------------
# ROW CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ROW SPLIT")
print("=" * 80)

print(
    f"Total point rows: {len(point):,}"
)

print(
    f"Train rows:       {len(train):,}"
)

print(
    f"Val rows:         {len(val):,}"
)

print(
    f"Train fraction:   {len(train)/len(point):.4f}"
)

print(
    f"Val fraction:     {len(val)/len(point):.4f}"
)

if len(train) + len(val) != len(point):
    raise RuntimeError(
        "Train + val row count mismatch."
    )

# ------------------------------------------------------------
# GROUNDING-ROW CHECK
# ------------------------------------------------------------

train_indices = set(
    train["grounding_row_index"]
)

val_indices = set(
    val["grounding_row_index"]
)

index_overlap = (
    train_indices &
    val_indices
)

print(
    f"Grounding-row overlap: "
    f"{len(index_overlap):,}"
)

if index_overlap:
    raise RuntimeError(
        "Grounding-row leakage detected."
    )

# ------------------------------------------------------------
# KEEP DOWNSTREAM FIELDS
# ------------------------------------------------------------

columns = [
    "grounding_row_index",
    "annotation_index",
    "patch_id",
    "input",
    "output",
    "point_xy",
    "target_box",
]

train = train[columns].reset_index(
    drop=True
)

val = val[columns].reset_index(
    drop=True
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

train_path = OUT / "point_train.parquet"
val_path = OUT / "point_val.parquet"
meta_path = OUT / "split_metadata.json"

train.to_parquet(
    train_path,
    index=False
)

val.to_parquet(
    val_path,
    index=False
)

metadata = {
    "seed": SEED,
    "train_fraction_by_patch": TRAIN_FRACTION,
    "source": str(SOURCE),
    "grounding_rows": int(len(grounding)),
    "point_rows": int(len(point)),
    "train_rows": int(len(train)),
    "val_rows": int(len(val)),
    "unique_point_patches": int(len(patches)),
    "train_unique_patches": int(len(train_patches)),
    "val_unique_patches": int(len(val_patches)),
    "patch_overlap": int(len(patch_overlap)),
    "grounding_row_overlap": int(len(index_overlap)),
    "point_parse_failures": int(failed_point),
    "box_parse_failures": int(failed_box),
    "out_of_range_points": int(bad_range.sum()),
    "points_inside_box": int(inside.sum()),
}

with open(
    meta_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(train_path)
print(val_path)
print(meta_path)

print("\n" + "=" * 80)
print("POINT GROUNDING SPLIT COMPLETE")
print("=" * 80)