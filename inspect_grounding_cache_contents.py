from pathlib import Path
import sys
import torch


PROJECT = Path(r"D:\SatQueryAI")

DOFA_DIR = (
    PROJECT
    / "kaggle"
    / "reference_grounding"
    / "dofa_features"
)

TEXT_DIR = (
    PROJECT
    / "kaggle"
    / "reference_grounding"
    / "grounding_text_embeddings"
)

POINT_VAL = (
    PROJECT
    / "data"
    / "processed"
    / "point_grounding_split"
    / "point_val.parquet"
)

REFERENCE_VAL = (
    PROJECT
    / "kaggle"
    / "reference_grounding_end_to_end"
    / "reference_val.parquet"
)


def inspect_object(name, obj, depth=0, max_depth=3):
    indent = "  " * depth

    print(
        f"{indent}{name}: "
        f"type={type(obj).__name__}"
    )

    if depth >= max_depth:
        return

    if isinstance(obj, dict):

        print(
            f"{indent}  keys={list(obj.keys())[:30]}"
        )

        for key, value in list(obj.items())[:10]:
            inspect_object(
                f"[{key!r}]",
                value,
                depth + 1,
                max_depth,
            )

    elif isinstance(obj, (list, tuple)):

        print(
            f"{indent}  length={len(obj)}"
        )

        for index, value in enumerate(obj[:5]):
            inspect_object(
                f"[{index}]",
                value,
                depth + 1,
                max_depth,
            )

    elif torch.is_tensor(obj):

        print(
            f"{indent}  shape={tuple(obj.shape)}"
        )
        print(
            f"{indent}  dtype={obj.dtype}"
        )

        if obj.numel() > 0:
            print(
                f"{indent}  device={obj.device}"
            )

    elif hasattr(obj, "shape"):

        try:
            print(
                f"{indent}  shape={obj.shape}"
            )
        except Exception:
            pass


def load_and_inspect(path, label):

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print(f"PATH: {path}")
    print(f"EXISTS: {path.exists()}")

    if not path.exists():
        return

    print(
        f"SIZE: {path.stat().st_size:,} bytes"
    )

    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        print(
            f"LOAD FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    inspect_object(
        "ROOT",
        obj,
        max_depth=4,
    )


# ============================================================
# 1. DOFA SHARD
# ============================================================

load_and_inspect(
    DOFA_DIR / "shard_0000.pt",
    "DOFA SHARD 0000",
)


# ============================================================
# 2. TEXT SHARD
# ============================================================

load_and_inspect(
    TEXT_DIR / "grounding_text_shard_000.pt",
    "GROUNDING TEXT SHARD 000",
)


# ============================================================
# 3. POINT VALIDATION SAMPLE
# ============================================================

print()
print("=" * 72)
print("POINT VALIDATION SAMPLE")
print("=" * 72)

try:
    import pandas as pd

    point_df = pd.read_parquet(
        POINT_VAL,
        engine="pyarrow",
    )

    print(
        f"Rows: {len(point_df):,}"
    )

    for index, row in point_df.head(5).iterrows():

        print()
        print(f"ROW {index}")

        print(
            f"  grounding_row_index: "
            f"{row['grounding_row_index']}"
        )

        print(
            f"  annotation_index: "
            f"{row['annotation_index']}"
        )

        print(
            f"  patch_id: "
            f"{row['patch_id']}"
        )

        print(
            f"  point_xy: "
            f"{row['point_xy']}"
        )

        print(
            f"  target_box: "
            f"{row['target_box']}"
        )

except Exception as exc:

    print(
        f"POINT PARQUET READ FAILED: "
        f"{type(exc).__name__}: {exc}"
    )


# ============================================================
# 4. REFERENCE VALIDATION SAMPLE
# ============================================================

print()
print("=" * 72)
print("REFERENCE VALIDATION SAMPLE")
print("=" * 72)

try:
    import pandas as pd

    reference_df = pd.read_parquet(
        REFERENCE_VAL,
        engine="pyarrow",
    )

    print(
        f"Rows: {len(reference_df):,}"
    )

    for index, row in reference_df.head(5).iterrows():

        print()
        print(f"ROW {index}")

        print(
            f"  ID: "
            f"{row['ID']}"
        )

        print(
            f"  original_row_index: "
            f"{row['original_row_index']}"
        )

        print(
            f"  patch_id: "
            f"{row['patch_id']}"
        )

        print(
            f"  input: "
            f"{row['input']}"
        )

        print(
            f"  output: "
            f"{row['output']}"
        )

except Exception as exc:

    print(
        f"REFERENCE PARQUET READ FAILED: "
        f"{type(exc).__name__}: {exc}"
    )


# ============================================================
# 5. CHECK SHARD COUNTS
# ============================================================

print()
print("=" * 72)
print("CACHE SHARD COUNTS")
print("=" * 72)

dofa_shards = sorted(
    DOFA_DIR.glob("shard_*.pt")
)

text_shards = sorted(
    TEXT_DIR.glob("grounding_text_shard_*.pt")
)

print(
    f"DOFA shards: {len(dofa_shards)}"
)

print(
    f"Text shards: {len(text_shards)}"
)

print()

print(
    "Expected from project registry:"
)

print(
    "  DOFA shards = 40"
)

print(
    "  Text shards = 180"
)

print()
print("=" * 72)
print("CACHE CONTENT INSPECTION COMPLETE")
print("=" * 72)
print("No model modified.")
print("No checkpoint modified.")
print("No training performed.")
print("Kaggle not used.")
print("GPU off.")