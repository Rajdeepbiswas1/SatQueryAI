from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

TRAIN_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

MAP_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_semantic_targets"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

VOCAB_JSON = (
    OUTPUT_ROOT
    / "clc_vocabulary.json"
)


# ============================================================
# LOAD PATCH IDS
# ============================================================

def load_patch_ids(
    parquet_path: Path,
) -> set[str]:

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing parquet:\n{parquet_path}"
        )

    df = pd.read_parquet(
        parquet_path
    )

    if "patch_id" not in df.columns:
        raise ValueError(
            f"'patch_id' missing from:\n{parquet_path}"
        )

    return set(
        df["patch_id"]
        .dropna()
        .astype(str)
        .unique()
    )


# ============================================================
# BUILD REFERENCE-MAP INDEX
# ============================================================

def build_map_index() -> dict[str, Path]:

    if not MAP_ROOT.exists():
        raise FileNotFoundError(
            f"Reference-map root missing:\n{MAP_ROOT}"
        )

    index = {}

    for path in MAP_ROOT.rglob(
        "*_reference_map.tif"
    ):

        suffix = "_reference_map.tif"

        if not path.name.endswith(
            suffix
        ):
            continue

        patch_id = path.name[
            :-len(suffix)
        ]

        if patch_id in index:
            raise RuntimeError(
                "Duplicate reference map:\n"
                f"{patch_id}\n"
                f"{index[patch_id]}\n"
                f"{path}"
            )

        index[patch_id] = path

    return index


# ============================================================
# COLLECT CLC VALUES FROM MAPS
# ============================================================

def collect_clc_values(
    patch_ids: set[str],
    map_index: dict[str, Path],
    split_name: str,
) -> tuple[set[int], dict[int, int]]:

    missing = (
        patch_ids
        - set(map_index.keys())
    )

    if missing:

        print(
            f"{split_name}: missing maps = "
            f"{len(missing)}"
        )

        for patch_id in sorted(
            missing
        )[:20]:

            print(
                patch_id
            )

        raise RuntimeError(
            f"Missing reference maps for "
            f"{split_name}."
        )

    values: set[int] = set()

    # Count how many patches contain each CLC value.
    patch_presence: dict[int, int] = {}

    total = len(
        patch_ids
    )

    print(
        f"\nScanning {split_name} maps: "
        f"{total}"
    )

    for number, patch_id in enumerate(
        sorted(patch_ids),
        start=1,
    ):

        path = map_index[
            patch_id
        ]

        with rasterio.open(
            path
        ) as src:

            arr = src.read(1)

        if arr.shape != (
            120,
            120,
        ):
            raise RuntimeError(
                "Unexpected reference-map shape:\n"
                f"Patch: {patch_id}\n"
                f"Shape: {arr.shape}"
            )

        unique = np.unique(
            arr
        )

        for value in unique:

            value = int(value)

            values.add(
                value
            )

            patch_presence[value] = (
                patch_presence.get(
                    value,
                    0,
                )
                + 1
            )

        if (
            number <= 5
            or number % 500 == 0
            or number == total
        ):

            print(
                f"  {number}/{total}"
            )

    return (
        values,
        patch_presence,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("LOCK C3 CLC VOCABULARY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load split patch IDs.
    # --------------------------------------------------------

    train_patch_ids = load_patch_ids(
        TRAIN_PARQUET
    )

    val_patch_ids = load_patch_ids(
        VAL_PARQUET
    )

    overlap = (
        train_patch_ids
        & val_patch_ids
    )

    print(
        f"Train patches: "
        f"{len(train_patch_ids)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_patch_ids)}"
    )

    print(
        f"Train/validation overlap: "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train/validation patch overlap detected."
        )

    # --------------------------------------------------------
    # Map index.
    # --------------------------------------------------------

    print(
        "\nIndexing reference maps..."
    )

    map_index = build_map_index()

    print(
        f"Reference maps indexed: "
        f"{len(map_index)}"
    )

    # --------------------------------------------------------
    # Training vocabulary.
    # --------------------------------------------------------

    train_values, train_presence = (
        collect_clc_values(
            train_patch_ids,
            map_index,
            "TRAIN",
        )
    )

    # --------------------------------------------------------
    # Validation vocabulary.
    # --------------------------------------------------------

    val_values, val_presence = (
        collect_clc_values(
            val_patch_ids,
            map_index,
            "VALIDATION",
        )
    )

    unseen_in_val = (
        val_values
        - train_values
    )

    # --------------------------------------------------------
    # Union for reporting only.
    # --------------------------------------------------------

    union_values = (
        train_values
        | val_values
    )

    print("\n" + "=" * 70)
    print("CLC VOCABULARY RESULT")
    print("=" * 70)

    print(
        f"TRAIN unique CLC values: "
        f"{len(train_values)}"
    )

    print(
        f"VALIDATION unique CLC values: "
        f"{len(val_values)}"
    )

    print(
        f"UNION unique CLC values: "
        f"{len(union_values)}"
    )

    print(
        f"VALIDATION CLC values unseen in TRAIN: "
        f"{len(unseen_in_val)}"
    )

    print(
        "\nTRAIN CLC VALUES:"
    )

    print(
        sorted(train_values)
    )

    print(
        "\nVALIDATION CLC VALUES:"
    )

    print(
        sorted(val_values)
    )

    if unseen_in_val:

        print(
            "\nCLC VALUES PRESENT ONLY IN VALIDATION:"
        )

        print(
            sorted(unseen_in_val)
        )

    else:

        print(
            "\nNo validation-only CLC values."
        )

    # --------------------------------------------------------
    # Build frozen vocabulary.
    # --------------------------------------------------------

    vocabulary = sorted(
        train_values
    )

    clc_to_index = {
        str(clc): index
        for index, clc
        in enumerate(vocabulary)
    }

    index_to_clc = {
        str(index): clc
        for index, clc
        in enumerate(vocabulary)
    }

    manifest = {
        "vocabulary_source": "TRAIN_REFERENCE_MAPS_ONLY",
        "train_patch_count": len(
            train_patch_ids
        ),
        "validation_patch_count": len(
            val_patch_ids
        ),
        "train_validation_patch_overlap": len(
            overlap
        ),
        "train_clc_count": len(
            train_values
        ),
        "validation_clc_count": len(
            val_values
        ),
        "union_clc_count": len(
            union_values
        ),
        "validation_only_clc_count": len(
            unseen_in_val
        ),
        "validation_only_clc_values": sorted(
            unseen_in_val
        ),
        "clc_vocabulary": vocabulary,
        "clc_to_index": clc_to_index,
        "index_to_clc": index_to_clc,
        "train_patch_presence": {
            str(k): int(v)
            for k, v
            in sorted(
                train_presence.items()
            )
        },
        "validation_patch_presence": {
            str(k): int(v)
            for k, v
            in sorted(
                val_presence.items()
            )
        },
    }

    with VOCAB_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"\nFrozen vocabulary:\n"
        f"{VOCAB_JSON}"
    )

    # --------------------------------------------------------
    # Final check.
    # --------------------------------------------------------

    if unseen_in_val:

        print(
            "\nWARNING:"
        )

        print(
            "Validation contains CLC values that "
            "are absent from training."
        )

        print(
            "We should NOT generate the final semantic "
            "targets until this is resolved."
        )

        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("C3 CLC VOCABULARY LOCKED")
    print("=" * 70)

    print(
        f"Frozen semantic classes: "
        f"{len(vocabulary)}"
    )

    print(
        "Validation has no unseen CLC classes."
    )


if __name__ == "__main__":
    main()